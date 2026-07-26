//! WAV inspection and in-process conversion to what whisper.cpp accepts.
//!
//! whisper.cpp reads exactly one shape of audio: 16 kHz, mono, 16-bit PCM. It
//! does not resample, it refuses. Everything upstream of it here produced
//! something else — the voice capture graph runs at 24 kHz, because that is what
//! the Moshi protocol wants — and a `.wav` extension was taken as proof the file
//! was already right, so a spoken turn sent to batch whisper failed on every
//! utterance while streaming ASR worked, since its Python worker resamples.
//!
//! Conversion happens here rather than through ffmpeg because the voice path
//! should not require a system ffmpeg to transcribe a microphone: it is a
//! resample of a few seconds of PCM, and the alternative is an install step
//! between the user and their first spoken word. ffmpeg remains the fallback for
//! containers this cannot read (mp3, m4a, ogg, video soundtracks).

use anyhow::Context;

/// What whisper.cpp requires of its input, and what conversion targets.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

const FORMAT_PCM: u16 = 1;
const FORMAT_FLOAT: u16 = 3;
const FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// The parts of a WAV header that decide whether a file needs converting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavInfo {
    pub format: u16,
    pub channels: u16,
    pub sample_rate: u32,
    pub bits_per_sample: u16,
    data_offset: usize,
    data_len: usize,
}

impl WavInfo {
    /// Whether whisper.cpp can read this file as it stands.
    pub fn is_whisper_ready(&self) -> bool {
        self.format == FORMAT_PCM
            && self.channels == 1
            && self.sample_rate == TARGET_SAMPLE_RATE
            && self.bits_per_sample == 16
    }

    /// Duration of the sample data, for logs and diagnostics.
    pub fn duration_seconds(&self) -> f32 {
        let frame_bytes = (self.bits_per_sample as usize / 8) * self.channels.max(1) as usize;
        if frame_bytes == 0 || self.sample_rate == 0 {
            return 0.0;
        }
        (self.data_len / frame_bytes) as f32 / self.sample_rate as f32
    }
}

/// Read a RIFF/WAVE header, or `None` when this is not a WAV at all.
///
/// Chunks other than `fmt ` and `data` are skipped rather than rejected: real
/// files carry `LIST`, `fact`, and metadata chunks, and a parser that only
/// accepts the minimal layout would send perfectly readable audio to ffmpeg.
pub fn inspect(bytes: &[u8]) -> Option<WavInfo> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let mut offset = 12;
    let mut format = None::<(u16, u16, u32, u16)>;
    let mut data = None::<(usize, usize)>;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().ok()?) as usize;
        let body = offset + 8;
        // A truncated final chunk is still usable: take what is there rather
        // than losing the whole recording to a missing byte.
        let end = body.saturating_add(size).min(bytes.len());
        if id == b"fmt " && end - body >= 16 {
            let chunk = &bytes[body..end];
            let mut code = u16::from_le_bytes(chunk[0..2].try_into().ok()?);
            let channels = u16::from_le_bytes(chunk[2..4].try_into().ok()?);
            let sample_rate = u32::from_le_bytes(chunk[4..8].try_into().ok()?);
            let bits = u16::from_le_bytes(chunk[14..16].try_into().ok()?);
            // WAVE_FORMAT_EXTENSIBLE names the real encoding in its GUID, whose
            // first two bytes are the ordinary format tag.
            if code == FORMAT_EXTENSIBLE && chunk.len() >= 26 {
                code = u16::from_le_bytes(chunk[24..26].try_into().ok()?);
            }
            format = Some((code, channels, sample_rate, bits));
        } else if id == b"data" {
            data = Some((body, end - body));
        }
        // Chunks are word-aligned, so an odd size is followed by a pad byte.
        offset = body + size + (size & 1);
        if size == 0 && id != b"data" {
            break;
        }
    }
    let (format, channels, sample_rate, bits_per_sample) = format?;
    let (data_offset, data_len) = data?;
    if channels == 0 || sample_rate == 0 || bits_per_sample == 0 {
        return None;
    }
    Some(WavInfo {
        format,
        channels,
        sample_rate,
        bits_per_sample,
        data_offset,
        data_len,
    })
}

/// Decode the sample data as mono float, averaging channels.
fn decode_mono(bytes: &[u8], info: &WavInfo) -> anyhow::Result<Vec<f32>> {
    let data = &bytes[info.data_offset..info.data_offset + info.data_len];
    let channels = info.channels as usize;
    let width = info.bits_per_sample as usize / 8;
    anyhow::ensure!(width > 0, "unsupported WAV sample width");
    let frame = width * channels;
    anyhow::ensure!(frame > 0, "unsupported WAV frame size");
    let frames = data.len() / frame;
    let mut mono = Vec::with_capacity(frames);
    for index in 0..frames {
        let mut sum = 0.0_f32;
        for channel in 0..channels {
            let start = index * frame + channel * width;
            let sample = &data[start..start + width];
            sum += match (info.format, info.bits_per_sample) {
                (FORMAT_PCM, 8) => (sample[0] as f32 - 128.0) / 128.0,
                (FORMAT_PCM, 16) => i16::from_le_bytes([sample[0], sample[1]]) as f32 / 32768.0,
                (FORMAT_PCM, 24) => {
                    let value = i32::from_le_bytes([0, sample[0], sample[1], sample[2]]) >> 8;
                    value as f32 / 8_388_608.0
                }
                (FORMAT_PCM, 32) => {
                    i32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]) as f32
                        / 2_147_483_648.0
                }
                (FORMAT_FLOAT, 32) => {
                    f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]])
                }
                (FORMAT_FLOAT, 64) => f64::from_le_bytes([
                    sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6],
                    sample[7],
                ]) as f32,
                (format, bits) => {
                    anyhow::bail!("unsupported WAV encoding: format {format}, {bits}-bit")
                }
            };
        }
        mono.push(sum / channels as f32);
    }
    Ok(mono)
}

/// Half-width of the resampling kernel, in output-side sample periods.
///
/// Sixteen lobes of a windowed sinc is far more than speech recognition needs
/// and still costs microseconds on an utterance: the whole point is that this
/// runs while someone waits for their turn to be heard.
const KERNEL_LOBES: usize = 16;

/// Resample mono float audio with a windowed-sinc kernel.
///
/// Plain decimation would be simpler and wrong: dropping 24 kHz to 16 kHz
/// without a low-pass folds everything above 8 kHz back into the speech band as
/// noise the recogniser then has to hear through. The kernel's cutoff follows
/// the lower of the two rates, so the same code upsamples cleanly too.
pub fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || input.is_empty() {
        return input.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = ((input.len() as f64) * ratio).round() as usize;
    // Cutoff just under the lower Nyquist, expressed in input samples.
    let cutoff = if ratio < 1.0 { ratio } else { 1.0 } * 0.95;
    let half_width = (KERNEL_LOBES as f64 / cutoff).ceil() as isize;
    let mut output = Vec::with_capacity(output_len);
    for index in 0..output_len {
        let center = index as f64 / ratio;
        let first = (center - half_width as f64).ceil() as isize;
        let last = (center + half_width as f64).floor() as isize;
        let mut sum = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for tap in first..=last {
            if tap < 0 || tap as usize >= input.len() {
                continue;
            }
            let distance = center - tap as f64;
            let weight = sinc(distance * cutoff) * blackman(distance / half_width as f64);
            sum += weight * input[tap as usize] as f64;
            weight_sum += weight;
        }
        // Normalising by the weights actually used keeps the ends of the signal
        // at their original level instead of fading where the kernel runs off.
        output.push(if weight_sum.abs() > f64::EPSILON {
            (sum / weight_sum) as f32
        } else {
            0.0
        });
    }
    output
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        return 1.0;
    }
    let pi_x = std::f64::consts::PI * x;
    pi_x.sin() / pi_x
}

fn blackman(normalized: f64) -> f64 {
    if normalized.abs() >= 1.0 {
        return 0.0;
    }
    let x = (normalized + 1.0) / 2.0;
    let two_pi_x = 2.0 * std::f64::consts::PI * x;
    0.42 - 0.5 * two_pi_x.cos() + 0.08 * (2.0 * two_pi_x).cos()
}

/// Encode mono float samples as a 16-bit PCM WAV.
pub fn encode_mono_pcm16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = samples.len() * 2;
    let mut bytes = Vec::with_capacity(44 + data_len);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16_u32.to_le_bytes());
    bytes.extend_from_slice(&FORMAT_PCM.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2_u16.to_le_bytes());
    bytes.extend_from_slice(&16_u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(data_len as u32).to_le_bytes());
    for sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        bytes.extend_from_slice(&((clamped * 32767.0).round() as i16).to_le_bytes());
    }
    bytes
}

/// Convert a WAV to 16 kHz mono 16-bit PCM, or report that it already is.
///
/// `Ok(None)` means the input is usable as it stands, so the caller can hand the
/// original file over without writing a copy.
pub fn to_whisper_wav(bytes: &[u8]) -> anyhow::Result<Option<Vec<u8>>> {
    let info = inspect(bytes).context("not a readable RIFF/WAVE file")?;
    if info.is_whisper_ready() {
        return Ok(None);
    }
    let mono = decode_mono(bytes, &info)?;
    let resampled = resample(&mono, info.sample_rate, TARGET_SAMPLE_RATE);
    Ok(Some(encode_mono_pcm16(&resampled, TARGET_SAMPLE_RATE)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(frequency: f32, sample_rate: u32, seconds: f32) -> Vec<f32> {
        let count = (sample_rate as f32 * seconds) as usize;
        (0..count)
            .map(|index| {
                let t = index as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
            })
            .collect()
    }

    /// Energy at one frequency, by the Goertzel algorithm.
    fn energy_at(samples: &[f32], frequency: f32, sample_rate: u32) -> f32 {
        let omega = 2.0 * std::f32::consts::PI * frequency / sample_rate as f32;
        let coefficient = 2.0 * omega.cos();
        let (mut s1, mut s2) = (0.0_f32, 0.0_f32);
        for sample in samples {
            let s0 = sample + coefficient * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        ((s1 * s1 + s2 * s2 - coefficient * s1 * s2).max(0.0)).sqrt() / samples.len() as f32
    }

    #[test]
    fn reads_the_header_the_voice_path_produces() {
        let wav = encode_mono_pcm16(&tone(440.0, 24_000, 0.25), 24_000);
        let info = inspect(&wav).expect("24 kHz capture WAV must parse");
        assert_eq!(info.sample_rate, 24_000);
        assert_eq!(info.channels, 1);
        assert_eq!(info.bits_per_sample, 16);
        assert!(
            !info.is_whisper_ready(),
            "24 kHz is not what whisper accepts"
        );
        assert!((info.duration_seconds() - 0.25).abs() < 0.01);
    }

    #[test]
    fn leaves_audio_whisper_can_already_read_alone() {
        let wav = encode_mono_pcm16(&tone(440.0, 16_000, 0.1), 16_000);
        assert!(inspect(&wav).unwrap().is_whisper_ready());
        assert!(
            to_whisper_wav(&wav).unwrap().is_none(),
            "a conforming file must not be rewritten"
        );
    }

    #[test]
    fn converts_a_spoken_turn_to_sixteen_kilohertz() {
        let wav = encode_mono_pcm16(&tone(440.0, 24_000, 0.5), 24_000);
        let converted = to_whisper_wav(&wav).unwrap().expect("must convert");
        let info = inspect(&converted).unwrap();
        assert!(info.is_whisper_ready());
        assert!(
            (info.duration_seconds() - 0.5).abs() < 0.01,
            "duration must survive the resample: {}",
            info.duration_seconds()
        );
        let samples = decode_mono(&converted, &info).unwrap();
        let at_tone = energy_at(&samples, 440.0, TARGET_SAMPLE_RATE);
        let elsewhere = energy_at(&samples, 3000.0, TARGET_SAMPLE_RATE);
        assert!(
            at_tone > 0.2 && at_tone > elsewhere * 20.0,
            "the tone must survive: {at_tone} vs {elsewhere}"
        );
    }

    /// Naive decimation would fold this back into the middle of the speech band.
    #[test]
    fn filters_out_what_sixteen_kilohertz_cannot_carry() {
        let wav = encode_mono_pcm16(&tone(10_000.0, 24_000, 0.5), 24_000);
        let converted = to_whisper_wav(&wav).unwrap().unwrap();
        let info = inspect(&converted).unwrap();
        let samples = decode_mono(&converted, &info).unwrap();
        // 10 kHz aliases to 6 kHz at a 16 kHz rate.
        let alias = energy_at(&samples, 6000.0, TARGET_SAMPLE_RATE);
        assert!(alias < 0.02, "aliased image must be suppressed: {alias}");
    }

    #[test]
    fn downmixes_stereo_and_reads_float_samples() {
        let left = tone(440.0, 48_000, 0.2);
        let mut bytes = Vec::new();
        let data_len = left.len() * 2 * 4;
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&FORMAT_FLOAT.to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&48_000_u32.to_le_bytes());
        bytes.extend_from_slice(&(48_000_u32 * 8).to_le_bytes());
        bytes.extend_from_slice(&8_u16.to_le_bytes());
        bytes.extend_from_slice(&32_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data_len as u32).to_le_bytes());
        for sample in &left {
            bytes.extend_from_slice(&sample.to_le_bytes());
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        let converted = to_whisper_wav(&bytes).unwrap().expect("must convert");
        let info = inspect(&converted).unwrap();
        assert!(info.is_whisper_ready());
        let samples = decode_mono(&converted, &info).unwrap();
        assert!(energy_at(&samples, 440.0, TARGET_SAMPLE_RATE) > 0.2);
    }

    #[test]
    fn skips_chunks_that_are_not_format_or_data() {
        let inner = encode_mono_pcm16(&tone(440.0, 16_000, 0.05), 16_000);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&inner[0..12]);
        // An odd-sized LIST chunk, which must be skipped including its pad byte.
        bytes.extend_from_slice(b"LIST");
        bytes.extend_from_slice(&5_u32.to_le_bytes());
        bytes.extend_from_slice(b"INFO\0");
        bytes.push(0);
        bytes.extend_from_slice(&inner[12..]);
        let patched_size = (bytes.len() - 8) as u32;
        bytes[4..8].copy_from_slice(&patched_size.to_le_bytes());
        let info = inspect(&bytes).expect("metadata chunks must not defeat the parser");
        assert_eq!(info.sample_rate, 16_000);
        assert!(info.is_whisper_ready());
    }

    #[test]
    fn reports_what_is_not_a_wav_at_all() {
        assert!(inspect(b"ID3\x04\x00junk").is_none());
        assert!(to_whisper_wav(b"not audio").is_err());
    }
}
