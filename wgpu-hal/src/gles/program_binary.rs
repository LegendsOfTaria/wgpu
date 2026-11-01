//! Linked programs kept between runs as pipeline cache data: the driver's binaries (`glGetProgramBinary`), by the
//! GLSL each was linked from. A driver without a shader cache of its own otherwise compiles and links every program
//! again at every launch, seconds of a phone's start.

use alloc::{boxed::Box, vec::Vec};
use core::fmt;

use glow::HasContext;
use naga::FastHashMap;
use parking_lot::Mutex;

/// Program binaries by the GLSL they were linked from.
#[derive(Default)]
pub struct PipelineCache {
    programs: Mutex<FastHashMap<Box<[u8]>, glow::ProgramBinary>>,
}

impl fmt::Debug for PipelineCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipelineCache")
            .field("programs", &self.programs.lock().len())
            .finish()
    }
}

impl crate::DynPipelineCache for PipelineCache {}

/// What a program is linked from, as its cache key: each stage and its GLSL, in order.
pub(super) fn program_key<'a>(
    stages: impl IntoIterator<Item = (naga::ShaderStage, &'a str)>,
) -> Box<[u8]> {
    let mut key = Vec::new();
    for (stage, source) in stages {
        key.push(stage as u8);
        key.extend_from_slice(&(source.len() as u32).to_le_bytes());
        key.extend_from_slice(source.as_bytes());
    }
    key.into_boxed_slice()
}

impl PipelineCache {
    /// Reads `data` from [`Self::data`]. Anything unreadable starts the cache empty.
    pub(super) fn from_data(data: &[u8]) -> Self {
        let mut programs = FastHashMap::default();
        let mut reader = Reader(data);
        let count = reader.u32().unwrap_or(0);
        for _ in 0..count {
            let entry = (|| {
                let key = reader.bytes()?;
                let format = reader.u32()?;
                let buffer = reader.bytes()?;
                Some((key, format, buffer))
            })();
            let Some((key, format, buffer)) = entry else {
                log::warn!("Program binary cache is truncated; starting it empty");
                programs.clear();
                break;
            };
            programs.insert(
                key.into(),
                glow::ProgramBinary {
                    buffer: buffer.to_vec(),
                    format,
                },
            );
        }
        log::debug!("Program binary cache: {} programs", programs.len());
        Self {
            programs: Mutex::new(programs),
        }
    }

    /// Every binary, for [`Self::from_data`] in a later run.
    pub(super) fn data(&self) -> Vec<u8> {
        let programs = self.programs.lock();
        let mut data = Vec::new();
        data.extend_from_slice(&(programs.len() as u32).to_le_bytes());
        for (key, binary) in programs.iter() {
            data.extend_from_slice(&(key.len() as u32).to_le_bytes());
            data.extend_from_slice(key);
            data.extend_from_slice(&binary.format.to_le_bytes());
            data.extend_from_slice(&(binary.buffer.len() as u32).to_le_bytes());
            data.extend_from_slice(&binary.buffer);
        }
        data
    }

    /// A new program linked from the binary kept for `key`, if there is one the driver still takes.
    pub(super) unsafe fn load(&self, gl: &glow::Context, key: &[u8]) -> Option<glow::Program> {
        let programs = self.programs.lock();
        let binary = programs.get(key)?;
        let program = unsafe { gl.create_program() }.ok()?;
        unsafe { gl.program_binary(program, binary) };
        if unsafe { gl.get_program_link_status(program) } {
            return Some(program);
        }
        // An updated driver may refuse an older driver's binary; the program compiles from its GLSL instead.
        log::debug!("Driver refused a cached program binary");
        unsafe { gl.delete_program(program) };
        None
    }

    /// Keeps `program`'s binary for `key`; it must have been linked with the retrievable hint.
    pub(super) unsafe fn store(&self, gl: &glow::Context, key: Box<[u8]>, program: glow::Program) {
        match unsafe { gl.get_program_binary(program) } {
            Some(binary) if !binary.buffer.is_empty() => {
                self.programs.lock().insert(key, binary);
            }
            _ => log::debug!("Driver gave no binary for program {program:?}"),
        }
    }
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        if self.0.len() < len {
            return None;
        }
        let (taken, rest) = self.0.split_at(len);
        self.0 = rest;
        Some(taken)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn bytes(&mut self) -> Option<&'a [u8]> {
        let len = self.u32()? as usize;
        self.take(len)
    }
}

/// The key of a cache of this driver's program binaries, which tells it apart from one another driver (or driver
/// version) made; `None` where binaries can't be kept between runs. `formats` is how many binary formats the
/// driver has (`GL_NUM_PROGRAM_BINARY_FORMATS`).
pub(super) fn validation_key(
    vendor: &str,
    renderer: &str,
    version: &str,
    formats: i32,
) -> Option<[u8; 16]> {
    log::debug!("Program binary formats: {formats}");
    if formats <= 0 || !drivers_keep_binaries(vendor, renderer) {
        return None;
    }
    // FNV-1a, twice with different starting values: stable from one run (and build) to the next.
    let hash = |mut hash: u64| {
        for text in [vendor, renderer, version] {
            for &byte in text.as_bytes().iter().chain(&[0]) {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    };
    let mut key = [0; 16];
    key[..8].copy_from_slice(&hash(0xcbf2_9ce4_8422_2325).to_le_bytes());
    key[8..].copy_from_slice(&hash(0x8422_2325_cbf2_9ce4).to_le_bytes());
    Some(key)
}

/// Drivers whose program binaries break, as Chromium and ANGLE found (`gpu_driver_bug_list.json`,
/// `disableProgramBinary`): Imagination's mix up cache entries (crbug.com/555781989) and PowerVR Rogue's need
/// working around (crbug.com/480992), Adreno 3xx binaries lose bound attribute locations (crbug.com/510637),
/// Adreno 4xx and 5xx drivers crash in `glProgramBinary` (crbug.com/699122), and Vivante's break rendering
/// (crbug.com/780498).
fn drivers_keep_binaries(vendor: &str, renderer: &str) -> bool {
    let vendor = vendor.to_lowercase();
    let renderer = renderer.to_lowercase();
    let broken = vendor.contains("imagination")
        || renderer.contains("powervr")
        || ["adreno (tm) 3", "adreno (tm) 4", "adreno (tm) 5"]
            .iter()
            .any(|adreno| renderer.contains(adreno))
        || vendor.contains("vivante")
        || renderer.contains("vivante");
    !broken
}
