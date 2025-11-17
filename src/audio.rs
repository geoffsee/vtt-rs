//! Audio capture and encoding utilities.
//!
//! This module provides low-level audio capture from system input devices
//! and encoding to WAV format for API transmission.

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use hound::{SampleFormat as HoundSampleFormat, WavSpec, WavWriter};
use std::io::Cursor;

/// Audio stream configuration parameters.
///
/// Contains the sample rate and channel count detected from the input device.
pub struct AudioConfig {
    /// Sample rate in Hz (e.g., 44100, 48000)
    pub sample_rate: u32,
    /// Number of audio channels (1 for mono, 2 for stereo)
    pub channels: u16,
}

/// Starts capturing audio from the default system input device.
///
/// This function initializes the audio input stream and begins sending captured
/// audio samples to the provided channel. All samples are normalized to f32
/// values in the range [-1.0, 1.0] regardless of the input device's native format.
///
/// The returned [`Stream`] must be kept alive for audio capture to continue.
/// Dropping the stream will stop audio capture.
///
/// # Examples
///
/// ```ignore
/// use tokio::sync::mpsc;
///
/// let (tx, mut rx) = mpsc::unbounded_channel();
/// let (stream, config) = start_audio_capture(tx)?;
///
/// println!("Capturing at {} Hz with {} channels",
///          config.sample_rate, config.channels);
///
/// // Keep stream alive
/// drop(stream);
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - No default input device is available
/// - The input device configuration cannot be retrieved
/// - The audio stream cannot be created or started
/// - The input device uses an unsupported sample format
///
/// # Panics
///
/// May panic if the audio system is not properly initialized (rare).
pub(crate) fn start_audio_capture(
    sender: tokio::sync::mpsc::UnboundedSender<Vec<f32>>,
) -> Result<(Stream, AudioConfig)> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("Unable to find a default input device")?;

    let input_config = device
        .default_input_config()
        .context("Failed to fetch the default input configuration")?;

    let stream_config: StreamConfig = input_config.clone().into();
    let sample_format = input_config.sample_format();

    let audio_config = AudioConfig {
        sample_rate: stream_config.sample_rate.0,
        channels: stream_config.channels,
    };

    let stream = match sample_format {
        SampleFormat::F32 => {
            let sender = sender.clone();
            let err_fn = move |err| eprintln!("Input stream error: {err}");
            device.build_input_stream(
                &stream_config,
                move |data: &[f32], _| send_samples_f32(data, &sender),
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let sender = sender.clone();
            let err_fn = move |err| eprintln!("Input stream error: {err}");
            device.build_input_stream(
                &stream_config,
                move |data: &[i16], _| send_samples_i16(data, &sender),
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let sender = sender.clone();
            let err_fn = move |err| eprintln!("Input stream error: {err}");
            device.build_input_stream(
                &stream_config,
                move |data: &[u16], _| send_samples_u16(data, &sender),
                err_fn,
                None,
            )?
        }
        _ => {
            return Err(anyhow!(
                "input sample format {:?} is not supported",
                sample_format
            ));
        }
    };

    stream
        .play()
        .context("Failed to start the audio input stream")?;

    Ok((stream, audio_config))
}

fn send_samples_f32(input: &[f32], sender: &tokio::sync::mpsc::UnboundedSender<Vec<f32>>) {
    let _ = sender.send(input.to_vec());
}

fn send_samples_i16(input: &[i16], sender: &tokio::sync::mpsc::UnboundedSender<Vec<f32>>) {
    let mut buffer = Vec::with_capacity(input.len());
    let scale = 1.0 / i16::MAX as f32;
    for sample in input {
        buffer.push((*sample as f32) * scale);
    }
    let _ = sender.send(buffer);
}

fn send_samples_u16(input: &[u16], sender: &tokio::sync::mpsc::UnboundedSender<Vec<f32>>) {
    let mut buffer = Vec::with_capacity(input.len());
    const MIDPOINT: f32 = 32768.0;
    for sample in input {
        buffer.push(((*sample as f32) - MIDPOINT) / MIDPOINT);
    }
    let _ = sender.send(buffer);
}

/// Encodes audio samples into WAV format suitable for API transmission.
///
/// Converts normalized f32 samples (range [-1.0, 1.0]) into a complete WAV file
/// with 16-bit integer samples. The output is returned as a byte vector ready
/// to be sent via HTTP multipart requests.
///
/// # Examples
///
/// ```ignore
/// let samples = vec![0.0, 0.5, -0.5, 0.0]; // Simple waveform
/// let wav_bytes = encode_wav(&samples, 44100, 1)?;
///
/// assert!(wav_bytes.len() > 44); // At minimum, WAV header + samples
/// ```
///
/// # Errors
///
/// Returns an error if the WAV encoding fails (very rare, typically only
/// on out-of-memory conditions).
pub(crate) fn encode_wav(samples: &[f32], sample_rate: u32, channels: u16) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: HoundSampleFormat::Int,
    };
    let mut buffer = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut buffer, spec)?;
        for &sample in samples {
            let clipped = sample.clamp(-1.0, 1.0);
            let amplitude = (clipped * i16::MAX as f32) as i16;
            writer.write_sample(amplitude)?;
        }
        writer.finalize()?;
    }
    Ok(buffer.into_inner())
}
