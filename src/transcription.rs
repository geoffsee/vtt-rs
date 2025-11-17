//! Transcription service and event types.
//!
//! This module contains the main [`TranscriptionService`] that coordinates
//! audio capture and transcription, as well as the [`TranscriptionEvent`] enum
//! for receiving results.

use crate::audio::{encode_wav, start_audio_capture};
use crate::config::Config;
use anyhow::{Context, Result};
use cpal::Stream;
use reqwest::Client;
use serde_json::Value;
use std::sync::Arc;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};

/// Events emitted by the transcription service.
///
/// These events are sent through the channel returned by [`TranscriptionService::start`]
/// and represent either successful transcriptions or errors that occurred during processing.
///
/// # Examples
///
/// ```no_run
/// use vtt_rs::{TranscriptionService, TranscriptionEvent, Config};
///
/// # #[tokio::main]
/// # async fn main() -> anyhow::Result<()> {
/// # let api_key = "test".to_string();
/// # let config = Config::default();
/// let mut service = TranscriptionService::new(config, api_key)?;
/// let (mut receiver, _stream) = service.start().await?;
///
/// while let Some(event) = receiver.recv().await {
///     match event {
///         TranscriptionEvent::Transcription { chunk_id, text } => {
///             if !text.is_empty() {
///                 println!("[{}] {}", chunk_id, text);
///             }
///         }
///         TranscriptionEvent::Error { chunk_id, error } => {
///             eprintln!("[{}] Error: {}", chunk_id, error);
///         }
///     }
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub enum TranscriptionEvent {
    /// A successful transcription.
    ///
    /// Contains the chunk ID and the transcribed text. An empty string
    /// indicates that silence was detected in the audio chunk.
    Transcription {
        /// The chunk ID (incremental counter starting from 0)
        chunk_id: usize,
        /// The transcribed text (empty string for silence)
        text: String,
    },
    /// An error occurred during transcription.
    ///
    /// This can happen due to network failures, API errors, or audio
    /// processing issues. The chunk ID helps identify which audio segment failed.
    Error {
        /// The chunk ID that failed
        chunk_id: usize,
        /// The error message describing what went wrong
        error: String,
    },
}

/// The main transcription service.
///
/// This service manages the entire transcription pipeline: capturing audio from
/// the system's default input device, chunking it into segments, sending those
/// segments to an OpenAI-compatible transcription API, and emitting events with
/// the results.
///
/// # Examples
///
/// ## Basic usage
///
/// ```no_run
/// use vtt_rs::{TranscriptionService, Config};
///
/// # #[tokio::main]
/// # async fn main() -> anyhow::Result<()> {
/// let config = Config::default();
/// let api_key = std::env::var("OPENAI_API_KEY")?;
///
/// let mut service = TranscriptionService::new(config, api_key)?;
/// let (mut receiver, _stream) = service.start().await?;
///
/// // Process events...
/// # Ok(())
/// # }
/// ```
///
/// ## With custom configuration
///
/// ```no_run
/// use vtt_rs::{TranscriptionService, Config};
///
/// # #[tokio::main]
/// # async fn main() -> anyhow::Result<()> {
/// let config = Config {
///     chunk_duration_secs: 3,
///     model: "whisper-1".to_string(),
///     endpoint: "https://api.openai.com/v1/audio/transcriptions".to_string(),
///     out_file: None,
/// };
///
/// let api_key = std::env::var("OPENAI_API_KEY")?;
/// let mut service = TranscriptionService::new(config, api_key)?;
/// # Ok(())
/// # }
/// ```
pub struct TranscriptionService {
    config: Config,
    api_key: String,
}

impl TranscriptionService {
    /// Creates a new transcription service with the specified configuration and API key.
    ///
    /// This doesn't start audio capture yet; call [`start`](Self::start) to begin
    /// transcription.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use vtt_rs::{TranscriptionService, Config};
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let config = Config::default();
    /// let api_key = std::env::var("OPENAI_API_KEY")?;
    /// let service = TranscriptionService::new(config, api_key)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Currently always succeeds, but returns [`Result`] for future extensibility.
    pub fn new(config: Config, api_key: String) -> Result<Self> {
        Ok(Self { config, api_key })
    }

    /// Starts the transcription service and returns a receiver for events.
    ///
    /// This method begins capturing audio from the default input device and spawns
    /// background tasks to process and transcribe audio chunks. Events are delivered
    /// through the returned [`UnboundedReceiver<TranscriptionEvent>`].
    ///
    /// The returned [`Stream`] must be kept alive for audio capture to continue.
    /// Dropping either the receiver or the stream will stop transcription.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use vtt_rs::{TranscriptionService, Config, TranscriptionEvent};
    ///
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// let config = Config::default();
    /// let api_key = std::env::var("OPENAI_API_KEY")?;
    ///
    /// let mut service = TranscriptionService::new(config, api_key)?;
    /// let (mut receiver, _stream) = service.start().await?;
    ///
    /// // Process transcription events
    /// while let Some(event) = receiver.recv().await {
    ///     match event {
    ///         TranscriptionEvent::Transcription { chunk_id, text } => {
    ///             println!("Chunk {}: {}", chunk_id, text);
    ///         }
    ///         TranscriptionEvent::Error { chunk_id, error } => {
    ///             eprintln!("Error in chunk {}: {}", chunk_id, error);
    ///         }
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No default audio input device is available
    /// - The audio device cannot be configured
    /// - The configured output file cannot be opened or created
    ///
    /// # Panics
    ///
    /// May panic if the audio system is not properly initialized (rare).
    pub async fn start(&mut self) -> Result<(UnboundedReceiver<TranscriptionEvent>, Stream)> {
        let (event_tx, event_rx) = unbounded_channel::<TranscriptionEvent>();
        let (sample_tx, mut sample_rx) = unbounded_channel::<Vec<f32>>();

        // Start audio capture
        let (_stream, audio_config) = start_audio_capture(sample_tx)?;

        // Open output file if configured
        let transcript_sink = if let Some(path) = &self.config.out_file {
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .await
                .with_context(|| format!("opening output file {}", path.display()))?;
            Some(Arc::new(tokio::sync::Mutex::new(file)))
        } else {
            None
        };

        let client = Client::new();
        let chunk_duration_secs = self.config.chunk_duration_secs.max(1);
        let samples_per_chunk = (audio_config.sample_rate as usize)
            .saturating_mul(chunk_duration_secs)
            .saturating_mul(audio_config.channels as usize);

        let model = Arc::new(self.config.model.clone());
        let endpoint = Arc::new(self.config.endpoint.clone());
        let api_key = self.api_key.clone();

        // Spawn the processing task
        tokio::spawn(async move {
            let mut buffer = Vec::with_capacity(samples_per_chunk * 2);
            let mut chunk_id = 0usize;

            while let Some(data) = sample_rx.recv().await {
                buffer.extend(data);

                while buffer.len() >= samples_per_chunk {
                    let chunk_samples = buffer.drain(..samples_per_chunk).collect::<Vec<_>>();
                    let client = client.clone();
                    let api_key = api_key.clone();
                    let sample_rate = audio_config.sample_rate;
                    let channels = audio_config.channels;
                    let current_chunk = chunk_id;
                    let chunk_model = model.clone();
                    let chunk_endpoint = endpoint.clone();
                    let chunk_sink = transcript_sink.clone();
                    let event_sender = event_tx.clone();

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
                                // Send event
                                let _ = event_sender.send(TranscriptionEvent::Transcription {
                                    chunk_id: current_chunk,
                                    text: text.clone(),
                                });

                                // Write to file if configured
                                if let Some(writer) = chunk_sink {
                                    let record_text =
                                        if text.is_empty() { "<silence>" } else { text.as_str() };
                                    if let Err(err) =
                                        append_transcript(writer, current_chunk, record_text).await
                                    {
                                        let _ = event_sender.send(TranscriptionEvent::Error {
                                            chunk_id: current_chunk,
                                            error: format!("File write failed: {err}"),
                                        });
                                    }
                                }
                            }
                            Err(err) => {
                                let _ = event_sender.send(TranscriptionEvent::Error {
                                    chunk_id: current_chunk,
                                    error: err.to_string(),
                                });
                            }
                        }
                    });

                    chunk_id += 1;
                }
            }
        });

        Ok((event_rx, _stream))
    }
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

async fn append_transcript(
    writer: Arc<tokio::sync::Mutex<tokio::fs::File>>,
    chunk_id: usize,
    text: &str,
) -> Result<()> {
    let mut guard = writer.lock().await;
    let entry = format!("Chunk {chunk_id}: {text}\n");
    guard.write_all(entry.as_bytes()).await?;
    guard.flush().await?;
    Ok(())
}
