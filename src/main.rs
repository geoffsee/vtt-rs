use anyhow::{Context, Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use hound::{SampleFormat as HoundSampleFormat, WavSpec, WavWriter};
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use std::{
    fs,
    io::Cursor,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{fs::OpenOptions, io::AsyncWriteExt, signal, sync::Mutex};

const DEFAULT_CONFIG_PATH: &str = "vtt.config.json";
const DEFAULT_CHUNK_DURATION_SECS: usize = 5;
const DEFAULT_MODEL: &str = "whisper-1";
const DEFAULT_ENDPOINT: &str = "https://api.openai.com/v1/audio/transcriptions";

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
struct Config {
    chunk_duration_secs: usize,
    model: String,
    endpoint: String,
    out_file: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            chunk_duration_secs: DEFAULT_CHUNK_DURATION_SECS,
            model: DEFAULT_MODEL.to_string(),
            endpoint: DEFAULT_ENDPOINT.to_string(),
            out_file: None,
        }
    }
}

fn load_config(arg_path: Option<String>) -> Result<(Config, Option<PathBuf>)> {
    if let Some(path) = arg_path {
        let path = PathBuf::from(path);
        let config = read_config_file(&path)?;
        return Ok((config, Some(path)));
    }

    let default_path = PathBuf::from(DEFAULT_CONFIG_PATH);
    if default_path.exists() {
        let config = read_config_file(&default_path)?;
        return Ok((config, Some(default_path)));
    }

    Ok((Config::default(), None))
}

fn read_config_file(path: &Path) -> Result<Config> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("reading config from {}", path.display()))?;
    serde_json::from_str(&contents)
        .with_context(|| format!("parsing config from {}", path.display()))
}

fn resolve_out_path(out_file: &Option<PathBuf>, source: Option<&Path>) -> Option<PathBuf> {
    out_file.as_ref().map(|path| {
        if path.is_absolute() {
            path.clone()
        } else if let Some(source) = source {
            source
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."))
                .join(path)
        } else {
            path.clone()
        }
    })
}

async fn open_output_file(path: Option<PathBuf>) -> Result<Option<Arc<Mutex<tokio::fs::File>>>> {
    if let Some(path) = path {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
            .with_context(|| format!("opening output file {}", path.display()))?;
        Ok(Some(Arc::new(Mutex::new(file))))
    } else {
        Ok(None)
    }
}

async fn append_transcript(
    writer: Arc<Mutex<tokio::fs::File>>,
    chunk_id: usize,
    text: &str,
) -> Result<()> {
    let mut guard = writer.lock().await;
    let entry = format!("Chunk {chunk_id}: {text}\n");
    guard.write_all(entry.as_bytes()).await?;
    guard.flush().await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .context("Set OPENAI_API_KEY to call OpenAI transcription")?;

    let config_arg = std::env::args().nth(1);
    let (config, config_path) = load_config(config_arg)?;
    if let Some(path) = config_path.as_deref() {
        println!("Loaded configuration from {}", path.display());
    } else {
        println!("Using default configuration");
    }
    let resolved_out_path = resolve_out_path(&config.out_file, config_path.as_deref());
    if let Some(path) = resolved_out_path.as_ref() {
        println!("Saving transcripts to {}", path.display());
    }
    let transcript_sink = open_output_file(resolved_out_path.clone()).await?;

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("Unable to find a default input device")?;
    let input_config = device
        .default_input_config()
        .context("Failed to fetch the default input configuration")?;
    let stream_config: StreamConfig = input_config.clone().into();
    let sample_format = input_config.sample_format();

    println!(
        "Capturing audio from \"{}\" ({:?}, {} Hz, {} channels)",
        device
            .name()
            .unwrap_or_else(|_| "Unknown input device".to_string()),
        sample_format,
        stream_config.sample_rate.0,
        stream_config.channels
    );

    let (sample_tx, mut sample_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<f32>>();

    let stream = match sample_format {
        SampleFormat::F32 => {
            let sender = sample_tx.clone();
            let err_fn = move |err| eprintln!("Input stream error: {err}");
            device.build_input_stream(
                &stream_config,
                move |data: &[f32], _| send_samples_f32(data, &sender),
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let sender = sample_tx.clone();
            let err_fn = move |err| eprintln!("Input stream error: {err}");
            device.build_input_stream(
                &stream_config,
                move |data: &[i16], _| send_samples_i16(data, &sender),
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let sender = sample_tx.clone();
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

    println!("Listening for speech. Press Ctrl+C to stop.");

    let client = Client::new();
    let chunk_duration_secs = config.chunk_duration_secs.max(1);
    let samples_per_chunk = (stream_config.sample_rate.0 as usize)
        .saturating_mul(chunk_duration_secs)
        .saturating_mul(stream_config.channels as usize);
    let mut buffer = Vec::with_capacity(samples_per_chunk * 2);
    let mut chunk_id = 0usize;
    let ctrl_c = signal::ctrl_c();
    tokio::pin!(ctrl_c);
    let model = Arc::new(config.model);
    let endpoint = Arc::new(config.endpoint);

    loop {
        tokio::select! {
            _ = &mut ctrl_c => {
                println!("Stopping capture...");
                break;
            }
            maybe_chunk = sample_rx.recv() => {
                let data = match maybe_chunk {
                    Some(chunk) => chunk,
                    None => break,
                };
                buffer.extend(data);
                while buffer.len() >= samples_per_chunk {
                    let chunk_samples = buffer.drain(..samples_per_chunk).collect::<Vec<_>>();
                    let client = client.clone();
                    let api_key = api_key.clone();
                    let sample_rate = stream_config.sample_rate.0;
                    let channels = stream_config.channels;
                    let current_chunk = chunk_id;
                    let chunk_model = model.clone();
                    let chunk_endpoint = endpoint.clone();
                    let chunk_sink = transcript_sink.clone();
                    tokio::spawn(async move {
                        match transcribe_chunk(
                            client,
                            api_key,
                            sample_rate,
                            channels,
                            chunk_samples,
                            current_chunk,
                            chunk_model,
                            chunk_endpoint,
                        )
                        .await
                        {
                            Ok(text) => {
                                if text.is_empty() {
                                    println!("Chunk {current_chunk}: <silence>");
                                } else {
                                    println!("Chunk {current_chunk}: {text}");
                                }
                                if let Some(writer) = chunk_sink {
                                    let record_text =
                                        if text.is_empty() { "<silence>" } else { text.as_str() };
                                    if let Err(err) = append_transcript(
                                        writer,
                                        current_chunk,
                                        record_text,
                                    )
                                    .await
                                    {
                                        eprintln!(
                                            "Chunk {current_chunk} file write failed: {err}"
                                        );
                                    }
                                }
                            }
                            Err(err) => {
                                eprintln!(
                                    "Chunk {current_chunk} transcription failed: {err}"
                                );
                            }
                        }
                    });
                    chunk_id += 1;
                }
            }
        }
    }

    Ok(())
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

async fn transcribe_chunk(
    client: Client,
    api_key: String,
    sample_rate: u32,
    channels: u16,
    samples: Vec<f32>,
    chunk_id: usize,
    model: Arc<String>,
    endpoint: Arc<String>,
) -> Result<String> {
    let wav = encode_wav(&samples, sample_rate, channels)?;
    let part = reqwest::multipart::Part::bytes(wav)
        .file_name(format!("chunk-{chunk_id}.wav"))
        .mime_str("audio/wav")?;
    let form = reqwest::multipart::Form::new()
        .text("model", model.as_ref().clone())
        .part("file", part);

    let response = client
        .post(endpoint.as_str())
        .bearer_auth(api_key)
        .multipart(form)
        .send()
        .await?
        .error_for_status()?;

    let payload: Value = response.json().await?;
    let text = payload
        .get("text")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .unwrap_or_default()
        .to_string();

    Ok(text)
}

fn encode_wav(samples: &[f32], sample_rate: u32, channels: u16) -> Result<Vec<u8>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

    struct DirGuard(PathBuf);
    impl Drop for DirGuard {
        fn drop(&mut self) {
            let _ = env::set_current_dir(&self.0);
        }
    }

    #[test]
    fn load_default_config_without_file() {
        let orig = env::current_dir().unwrap();
        let temp = tempdir().unwrap();
        env::set_current_dir(temp.path()).unwrap();
        let _guard = DirGuard(orig.clone());

        let (config, source) = load_config(None).unwrap();
        assert!(source.is_none());
        assert_eq!(config.chunk_duration_secs, DEFAULT_CHUNK_DURATION_SECS);
        assert_eq!(config.model, DEFAULT_MODEL);
        assert_eq!(config.endpoint, DEFAULT_ENDPOINT);
        assert!(config.out_file.is_none());
        drop(_guard);
    }

    #[test]
    fn load_config_from_explicit_path() {
        let temp = tempdir().unwrap();
        let config_path = temp.path().join("custom.json");
        let contents = r#"{"chunk_duration_secs":1,"model":"custom","endpoint":"https://example.com","out_file":"log.txt"}"#;
        fs::write(&config_path, contents).unwrap();

        let (config, source) =
            load_config(Some(config_path.to_string_lossy().to_string())).unwrap();
        assert_eq!(source.unwrap(), config_path);
        assert_eq!(config.chunk_duration_secs, 1);
        assert_eq!(config.model, "custom");
        assert_eq!(config.endpoint, "https://example.com");
        assert_eq!(config.out_file.unwrap(), PathBuf::from("log.txt"));
    }

    #[test]
    fn resolve_out_path_relative_with_source() {
        let source = Path::new("/tmp/vtt").join("vtt.config.json");
        let out = resolve_out_path(&Some(PathBuf::from("logs/output.log")), Some(&source));
        assert_eq!(out.unwrap(), PathBuf::from("/tmp/vtt/logs/output.log"));
    }

    #[test]
    fn resolve_out_path_absolute_returns_same() {
        let out = resolve_out_path(&Some(PathBuf::from("/var/log/vtt.log")), None);
        assert_eq!(out.unwrap(), PathBuf::from("/var/log/vtt.log"));
    }
}
