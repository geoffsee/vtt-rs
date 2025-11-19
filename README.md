# vtt-rs

A Rust library and command-line utility for real-time audio transcription using OpenAI-compatible APIs. Perfect for adding situational awareness to AI agents through speech recognition.

[![CI](https://github.com/geoffsee/vtt-rs/workflows/CI/badge.svg)](https://github.com/geoffsee/vtt-rs/actions/workflows/ci.yml)
[![Documentation](https://github.com/geoffsee/vtt-rs/workflows/Documentation/badge.svg)](https://github.com/geoffsee/vtt-rs/actions/workflows/docs.yml)
[![API Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://geoffsee.github.io/vtt-rs/vtt_rs/)

## Documentation

- **[API Documentation](https://geoffsee.github.io/vtt-rs/vtt_rs/)** - Full API reference (auto-generated from code)
- **[Integration Guide](INTEGRATION_GUIDE.md)** - Comprehensive guide for AI agent integration

Or build locally:
```bash
cargo doc --no-deps --open
```

## Configuration

- Set `OPENAI_API_KEY` in your environment before running anything.
- The binary expects an optional JSON configuration file (default `vtt.config.json` in the current directory, or pass an alternate path as the first argument).
- Supported keys (all optional; sensible defaults exist):
  ```json
  {
    "chunk_duration_secs": 5,
    "model": "whisper-1",
    "endpoint": "https://api.openai.com/v1/audio/transcriptions",
    "out_file": "transcripts.log",
    "on_device": {
      "enabled": false,
      "model": "tiny.en",
      "cpu": true
    }
  }
  ```

  * `chunk_duration_secs`: duration of each captured audio block that is transcribed.
  * `model`: which OpenAI transcription model to hit.
  * `endpoint`: custom transcription endpoint for e.g. a proxy service.
  * `out_file`: path to append every transcription (chunk ID + contents).
  * `on_device`: optional block to turn on the bundled Candle Whisper runner.

### On-Device Whisper

Set `on_device.enabled` to `true` in your config to run Whisper locally without
calling the OpenAI API. You can pick from the built-in model shortcuts
(`"tiny"`, `"small"`, etc.), force CPU execution, and optionally select a
specific input device.

## Usage as a Library

Add vtt-rs to your `Cargo.toml`:

```toml
[dependencies]
vtt-rs = { git = "https://github.com/geoffsee/vtt-rs" }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

### Basic Example

```rust
use vtt_rs::{Config, TranscriptionEvent, TranscriptionService};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Config::default();
    let api_key = std::env::var("OPENAI_API_KEY")?;

    let mut service = TranscriptionService::new(config, api_key)?;
    let (mut receiver, _stream) = service.start().await?;

    // Process transcription events
    while let Some(event) = receiver.recv().await {
        match event {
            TranscriptionEvent::Transcription { chunk_id, text } => {
                println!("Heard: {}", text);
                // Feed this to your AI agent for situational awareness
            }
            TranscriptionEvent::Error { chunk_id, error } => {
                eprintln!("Error: {}", error);
            }
        }
    }

    Ok(())
}
```

### AI Agent Integration

The library is designed to give AI agents "ears" - the ability to perceive and respond to their audio environment. Check out the examples:

- `examples/ai_agent.rs` - Basic AI agent with audio awareness
- `examples/streaming_agent.rs` - Advanced agent with temporal context

Run examples with:
```bash
OPENAI_API_KEY=sk-... cargo run --example ai_agent
```

## Usage as a CLI

```bash
OPENAI_API_KEY=sk-... cargo run -- vtt.config.json
```

- Omit the CLI argument to let the tool load `vtt.config.json` from the current directory if it exists, otherwise it runs with defaults.
- Transcripts are printed live and, when `out_file` is set, appended to that file in addition to the console output.

## Features

- **Real-time transcription**: Continuously captures and transcribes audio
- **Event-driven API**: React to transcriptions as they happen
- **Configurable chunking**: Adjust audio chunk duration for your needs
- **OpenAI compatible**: Works with OpenAI Whisper and compatible APIs
- **Async/await**: Built on Tokio for efficient async processing
- **Type-safe**: Strongly typed events and configuration
