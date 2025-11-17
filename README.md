# vtt-rs

Command-line utility that streams microphone input to OpenAI compatible transcription APIs

## Configuration

- Set `OPENAI_API_KEY` in your environment before running anything.
- The binary expects an optional JSON configuration file (default `vtt.config.json` in the current directory, or pass an alternate path as the first argument).
- Supported keys (all optional; sensible defaults exist):
  ```json
  {
    "chunk_duration_secs": 5,
    "model": "whisper-1",
    "endpoint": "https://api.openai.com/v1/audio/transcriptions",
    "out_file": "transcripts.log"
  }
  ```

  * `chunk_duration_secs`: duration of each captured audio block that is transcribed.
  * `model`: which OpenAI transcription model to hit.
  * `endpoint`: custom transcription endpoint for e.g. a proxy service.
  * `out_file`: path to append every transcription (chunk ID + contents).

## Running

```bash
OPENAI_API_KEY=sk-... cargo run -- vtt.config.json
```

- Omit the CLI argument to let the tool load `vtt.config.json` from the current directory if it exists, otherwise it runs with defaults.
- Transcripts are printed live and, when `out_file` is set, appended to that file in addition to the console output.
