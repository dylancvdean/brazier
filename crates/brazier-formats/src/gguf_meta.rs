//! Read string metadata keys from GGUF files (e.g. `tokenizer.chat_template`).

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

/// GGUF value type for a UTF-8 string.
const GGUF_TYPE_STRING: u32 = 8;
/// GGUF value type for a homogeneous array.
const GGUF_TYPE_ARRAY: u32 = 9;

/// Maximum bytes we will load for a single string metadata value.
const MAX_STRING_BYTES: usize = 2 * 1024 * 1024;

/// Read a string metadata value from a GGUF file.
///
/// Returns `Ok(None)` when the key is missing or not a string. Walks the
/// metadata section from the start of the file so large templates still work
/// without loading the whole GGUF into memory.
pub fn read_string_kv(path: &Path, key: &str) -> anyhow::Result<Option<String>> {
    let mut file =
        File::open(path).map_err(|error| anyhow::anyhow!("open {}: {error}", path.display()))?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .map_err(|error| anyhow::anyhow!("read GGUF magic: {error}"))?;
    anyhow::ensure!(&magic == b"GGUF", "{} is not a GGUF file", path.display());
    let version = read_u32(&mut file)?;
    anyhow::ensure!(
        (1..=3).contains(&version),
        "unsupported GGUF version {version}"
    );
    let _tensor_count = read_u64(&mut file)?;
    let kv_count = read_u64(&mut file)?;
    for _ in 0..kv_count {
        let entry_key = read_string(&mut file)?;
        let value_type = read_u32(&mut file)?;
        if entry_key == key {
            return match value_type {
                GGUF_TYPE_STRING => Ok(Some(read_string(&mut file)?)),
                _ => Ok(None),
            };
        }
        skip_value(&mut file, value_type)?;
    }
    Ok(None)
}

/// Chat template Jinja source embedded in a GGUF, when present.
pub fn read_chat_template(path: &Path) -> anyhow::Result<Option<String>> {
    // Most instruct GGUFs use tokenizer.chat_template; a few older converters
    // nested it under tokenizer.ggml.*.
    for key in [
        "tokenizer.chat_template",
        "tokenizer.ggml.chat_template",
        "chat_template",
    ] {
        if let Some(template) = read_string_kv(path, key)? {
            return Ok(Some(template));
        }
    }
    Ok(None)
}

fn read_u32(file: &mut File) -> anyhow::Result<u32> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)
        .map_err(|error| anyhow::anyhow!("read u32: {error}"))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(file: &mut File) -> anyhow::Result<u64> {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf)
        .map_err(|error| anyhow::anyhow!("read u64: {error}"))?;
    Ok(u64::from_le_bytes(buf))
}

fn read_string(file: &mut File) -> anyhow::Result<String> {
    let length = usize::try_from(read_u64(file)?)
        .map_err(|_| anyhow::anyhow!("GGUF string length does not fit in memory"))?;
    anyhow::ensure!(
        length <= MAX_STRING_BYTES,
        "GGUF string exceeds {MAX_STRING_BYTES} bytes"
    );
    let mut buf = vec![0u8; length];
    if length > 0 {
        file.read_exact(&mut buf)
            .map_err(|error| anyhow::anyhow!("read string: {error}"))?;
    }
    String::from_utf8(buf).map_err(|error| anyhow::anyhow!("GGUF string is not UTF-8: {error}"))
}

fn skip_value(file: &mut File, value_type: u32) -> anyhow::Result<()> {
    let size = match value_type {
        0 | 1 | 7 => 1, // u8 / i8 / bool
        2 | 3 => 2,     // u16 / i16
        4..=6 => 4,     // u32 / i32 / f32
        10..=12 => 8,   // u64 / i64 / f64
        GGUF_TYPE_STRING => {
            let length = read_u64(file)?;
            skip_bytes(file, length, "string")?;
            return Ok(());
        }
        GGUF_TYPE_ARRAY => {
            let element_type = read_u32(file)?;
            let count = read_u64(file)?;
            for _ in 0..count {
                skip_value(file, element_type)?;
            }
            return Ok(());
        }
        other => anyhow::bail!("unsupported GGUF value type {other}"),
    };
    skip_bytes(file, size as u64, "value")?;
    Ok(())
}

fn skip_bytes(file: &mut File, length: u64, kind: &str) -> anyhow::Result<()> {
    let offset = i64::try_from(length)
        .map_err(|_| anyhow::anyhow!("GGUF {kind} is too large to seek past"))?;
    file.seek(SeekFrom::Current(offset))
        .map_err(|error| anyhow::anyhow!("skip {kind}: {error}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn write_string(out: &mut Vec<u8>, value: &str) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }

    fn write_gguf(entries: &[(&str, &str)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (key, value) in entries {
            write_string(&mut out, key);
            out.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
            write_string(&mut out, value);
        }
        out
    }

    #[test]
    fn reads_chat_template_string() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        let bytes = write_gguf(&[
            ("general.architecture", "qwen3"),
            ("tokenizer.chat_template", "{% raw %}hello{% endraw %}"),
        ]);
        File::create(&path).unwrap().write_all(&bytes).unwrap();
        assert_eq!(
            read_chat_template(&path).unwrap().as_deref(),
            Some("{% raw %}hello{% endraw %}")
        );
    }
    #[test]
    fn missing_key_returns_none() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        let bytes = write_gguf(&[("general.architecture", "llama")]);
        File::create(&path).unwrap().write_all(&bytes).unwrap();
        assert_eq!(read_chat_template(&path).unwrap(), None);
    }

    #[test]
    fn rejects_unseekable_string_lengths() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        write_string(&mut bytes, "ignored");
        bytes.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        File::create(&path).unwrap().write_all(&bytes).unwrap();

        let error = read_string_kv(&path, "wanted").unwrap_err();
        assert!(error.to_string().contains("too large to seek past"));
    }
}
