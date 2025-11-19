use super::{TranscriptionEvent, append_transcript};
use crate::config::OnDeviceConfig;
use anyhow::{Context, Error as E, Result, anyhow};
use candle::utils::{cuda_is_available, metal_is_available};
use candle::{D, Device, IndexOp, Tensor};
use candle_nn::{VarBuilder, ops::softmax};
use hf_hub::{Repo, RepoType, api::sync::Api};
use rand::{SeedableRng, distr::Distribution};
use tokenizers::Tokenizer;
use tokio::runtime::Handle;
use tokio::sync::{Mutex, mpsc::UnboundedSender};

use std::str::FromStr;
use std::sync::{Arc, mpsc};
use std::thread;

use rubato::{FastFixedIn, PolynomialDegree, Resampler};

mod multilingual;

use candle_transformers::models::whisper::{self as m, Config, audio};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

pub enum Model {
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

// Maybe we should use some traits rather than doing the dispatch for all these.
impl Model {
    pub fn config(&self) -> &Config {
        match self {
            Self::Normal(m) => &m.config,
            Self::Quantized(m) => &m.config,
        }
    }

    pub fn encoder_forward(&mut self, x: &Tensor, flush: bool) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.encoder.forward(x.into(), flush),
            Self::Quantized(m) => m.encoder.forward(x, flush),
        }
    }

    pub fn decoder_forward(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
        flush: bool,
    ) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.forward(x, xa, flush),
            Self::Quantized(m) => m.decoder.forward(x, xa, flush),
        }
    }

    pub fn decoder_final_linear(&self, x: &Tensor) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.final_linear(x),
            Self::Quantized(m) => m.decoder.final_linear(x),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct DecodingResult {
    tokens: Vec<u32>,
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Segment {
    start: f64,
    duration: f64,
    dr: DecodingResult,
}

struct Decoder {
    model: Model,
    rng: rand::rngs::StdRng,
    task: Option<Task>,
    timestamps: bool,
    verbose: bool,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        device: &Device,
        language_token: Option<u32>,
        task: Option<Task>,
        timestamps: bool,
        verbose: bool,
    ) -> Result<Self> {
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i)
                    || timestamps && i == no_timestamps_token
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, m::TRANSLATE_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok());
        let no_speech_token = match no_speech_token {
            None => anyhow::bail!("unable to find any non-speech token"),
            Some(n) => n,
        };
        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            task,
            timestamps,
            verbose,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
        })
    }

    fn decode(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult> {
        let model = &mut self.model;
        let audio_features = model.encoder_forward(mel, true)?;
        if self.verbose {
            println!("audio features: {:?}", audio_features.dims());
        }
        let sample_len = model.config().max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        match self.task {
            None | Some(Task::Transcribe) => tokens.push(self.transcribe_token),
            Some(Task::Translate) => tokens.push(self.translate_token),
        }
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = model.decoder_forward(&tokens_t, &audio_features, i == 0)?;

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            // TODO: Besides suppress tokens, we should apply the heuristics from
            // ApplyTimestampRules, i.e.:
            // - Timestamps come in pairs, except before EOT.
            // - Timestamps should be non-decreasing.
            // - If the sum of the probabilities of timestamps is higher than any other tokens,
            //   only consider timestamps when sampling.
            // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L439
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = rand::distr::weighted::WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.eot_token || tokens.len() > model.config().max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    fn decode_with_fallback(&mut self, segment: &Tensor) -> Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(segment, t);
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    println!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    fn run(&mut self, mel: &Tensor, times: Option<(f64, f64)>) -> Result<Vec<Segment>> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];
        while seek < content_frames {
            let start = std::time::Instant::now();
            let time_offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment)?;
            seek += segment_size;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                println!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            if self.timestamps {
                println!(
                    "{:.1}s -- {:.1}s",
                    segment.start,
                    segment.start + segment.duration,
                );
                let mut tokens_to_decode = vec![];
                let mut prev_timestamp_s = 0f32;
                for &token in segment.dr.tokens.iter() {
                    if token == self.sot_token || token == self.eot_token {
                        continue;
                    }
                    // The no_timestamp_token is the last before the timestamp ones.
                    if token > self.no_timestamps_token {
                        let timestamp_s = (token - self.no_timestamps_token + 1) as f32 / 50.;
                        if !tokens_to_decode.is_empty() {
                            let text = self
                                .tokenizer
                                .decode(&tokens_to_decode, true)
                                .map_err(E::msg)?;
                            println!("  {:.1}s-{:.1}s: {}", prev_timestamp_s, timestamp_s, text);
                            tokens_to_decode.clear()
                        }
                        prev_timestamp_s = timestamp_s;
                    } else {
                        tokens_to_decode.push(token)
                    }
                }
                if !tokens_to_decode.is_empty() {
                    let text = self
                        .tokenizer
                        .decode(&tokens_to_decode, true)
                        .map_err(E::msg)?;
                    if !text.is_empty() {
                        println!("  {:.1}s-...: {}", prev_timestamp_s, text);
                    }
                    tokens_to_decode.clear()
                }
            } else {
                match times {
                    Some((start, end)) => {
                        println!("{:.1}s -- {:.1}s: {}", start, end, segment.dr.text)
                    }
                    None => {
                        println!(
                            "{:.1}s -- {:.1}s: {}",
                            segment.start,
                            segment.start + segment.duration,
                            segment.dr.text,
                        )
                    }
                }
            }
            if self.verbose {
                println!("{seek}: {segment:?}, in {:?}", start.elapsed());
            }
            segments.push(segment)
        }
        Ok(segments)
    }

    fn set_language_token(&mut self, language_token: Option<u32>) {
        self.language_token = language_token;
    }

    #[allow(dead_code)]
    fn reset_kv_cache(&mut self) {
        match &mut self.model {
            Model::Normal(m) => m.reset_kv_cache(),
            Model::Quantized(m) => m.reset_kv_cache(),
        }
    }

    fn model(&mut self) -> &mut Model {
        &mut self.model
    }
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Task {
    Transcribe,
    Translate,
}

impl FromStr for Task {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            v if v.eq_ignore_ascii_case("transcribe") => Ok(Task::Transcribe),
            v if v.eq_ignore_ascii_case("translate") => Ok(Task::Translate),
            other => Err(anyhow!("Unknown task '{other}'")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WhichModel {
    Tiny,
    TinyEn,
    Base,
    BaseEn,
    Small,
    SmallEn,
    Medium,
    MediumEn,
    Large,
    LargeV2,
    LargeV3,
    LargeV3Turbo,
    DistilMediumEn,
    DistilLargeV2,
}

impl FromStr for WhichModel {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "tiny" => Ok(WhichModel::Tiny),
            "tiny.en" => Ok(WhichModel::TinyEn),
            "base" => Ok(WhichModel::Base),
            "base.en" => Ok(WhichModel::BaseEn),
            "small" => Ok(WhichModel::Small),
            "small.en" => Ok(WhichModel::SmallEn),
            "medium" => Ok(WhichModel::Medium),
            "medium.en" => Ok(WhichModel::MediumEn),
            "large" => Ok(WhichModel::Large),
            "large-v2" => Ok(WhichModel::LargeV2),
            "large-v3" => Ok(WhichModel::LargeV3),
            "large-v3-turbo" => Ok(WhichModel::LargeV3Turbo),
            "distil-medium.en" => Ok(WhichModel::DistilMediumEn),
            "distil-large-v2" => Ok(WhichModel::DistilLargeV2),
            other => Err(anyhow!("Unknown Whisper model '{other}'")),
        }
    }
}

impl WhichModel {
    fn is_multilingual(&self) -> bool {
        match self {
            Self::Tiny
            | Self::Base
            | Self::Small
            | Self::Medium
            | Self::Large
            | Self::LargeV2
            | Self::LargeV3
            | Self::LargeV3Turbo
            | Self::DistilLargeV2 => true,
            Self::TinyEn | Self::BaseEn | Self::SmallEn | Self::MediumEn | Self::DistilMediumEn => {
                false
            }
        }
    }

    fn model_and_revision(&self) -> (&'static str, &'static str) {
        match self {
            Self::Tiny => ("openai/whisper-tiny", "main"),
            Self::TinyEn => ("openai/whisper-tiny.en", "refs/pr/15"),
            Self::Base => ("openai/whisper-base", "refs/pr/22"),
            Self::BaseEn => ("openai/whisper-base.en", "refs/pr/13"),
            Self::Small => ("openai/whisper-small", "main"),
            Self::SmallEn => ("openai/whisper-small.en", "refs/pr/10"),
            Self::Medium => ("openai/whisper-medium", "main"),
            Self::MediumEn => ("openai/whisper-medium.en", "main"),
            Self::Large => ("openai/whisper-large", "refs/pr/36"),
            Self::LargeV2 => ("openai/whisper-large-v2", "refs/pr/57"),
            Self::LargeV3 => ("openai/whisper-large-v3", "main"),
            Self::LargeV3Turbo => ("openai/whisper-large-v3-turbo", "main"),
            Self::DistilMediumEn => ("distil-whisper/distil-medium.en", "main"),
            Self::DistilLargeV2 => ("distil-whisper/distil-large-v2", "main"),
        }
    }

    fn quantized_tag(&self) -> Option<&'static str> {
        match self {
            Self::Tiny => Some("tiny"),
            Self::TinyEn => Some("tiny-en"),
            _ => None,
        }
    }
}

pub(super) fn start_on_device_transcription(
    cfg: OnDeviceConfig,
    event_sender: UnboundedSender<TranscriptionEvent>,
    transcript_sink: Option<Arc<Mutex<tokio::fs::File>>>,
    runtime: Handle,
) -> Result<cpal::Stream> {
    let audio_device = select_audio_device(cfg.audio_device.as_deref())?;
    let audio_config = audio_device
        .default_input_config()
        .context("Failed to get default audio input config")?;
    let channel_count = audio_config.channels() as usize;
    let in_sample_rate = audio_config.sample_rate().0 as usize;
    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let stream = audio_device.build_input_stream(
        &audio_config.config(),
        move |pcm: &[f32], _: &cpal::InputCallbackInfo| {
            let chunk = pcm
                .iter()
                .step_by(channel_count)
                .copied()
                .collect::<Vec<f32>>();
            if !chunk.is_empty() {
                let _ = tx.send(chunk);
            }
        },
        move |err| eprintln!("an error occurred on stream: {err}"),
        None,
    )?;
    stream.play()?;

    let thread_cfg = cfg.clone();
    let runtime_clone = runtime.clone();
    thread::spawn(move || {
        if let Err(err) = run_on_device(
            thread_cfg,
            in_sample_rate,
            event_sender,
            transcript_sink,
            runtime_clone,
            rx,
        ) {
            eprintln!("on-device whisper loop exited: {err:?}");
        }
    });

    Ok(stream)
}

fn run_on_device(
    cfg: OnDeviceConfig,
    in_sample_rate: usize,
    event_sender: UnboundedSender<TranscriptionEvent>,
    transcript_sink: Option<Arc<Mutex<tokio::fs::File>>>,
    runtime: Handle,
    rx: mpsc::Receiver<Vec<f32>>,
) -> Result<()> {
    let which_model = WhichModel::from_str(cfg.model.trim())?;
    if !which_model.is_multilingual() && cfg.language.is_some() {
        anyhow::bail!("a language cannot be set for non-multilingual models");
    }
    let task = parse_task_option(&cfg.task)?;
    let device = build_device(cfg.cpu)?;
    let (config, tokenizer, model) = load_model_artifacts(&cfg, which_model, &device)?;
    let mel_filters = load_mel_filters(&config)?;
    let decoder = Decoder::new(
        model,
        tokenizer.clone(),
        cfg.seed,
        &device,
        None,
        task,
        cfg.timestamps,
        cfg.verbose,
    )?;

    process_pcm_stream(
        cfg,
        which_model,
        decoder,
        tokenizer,
        config,
        mel_filters,
        device,
        in_sample_rate,
        event_sender,
        transcript_sink,
        runtime,
        rx,
    )
}

#[allow(clippy::too_many_arguments)]
fn process_pcm_stream(
    cfg: OnDeviceConfig,
    which_model: WhichModel,
    mut decoder: Decoder,
    tokenizer: Tokenizer,
    config: Config,
    mel_filters: Vec<f32>,
    device: Device,
    in_sample_rate: usize,
    event_sender: UnboundedSender<TranscriptionEvent>,
    transcript_sink: Option<Arc<Mutex<tokio::fs::File>>>,
    runtime: Handle,
    rx: mpsc::Receiver<Vec<f32>>,
) -> Result<()> {
    let mut buffered_pcm = Vec::new();
    let mut language_token_set = false;
    let mut chunk_id = 0usize;
    let mut resampler = FastFixedIn::new(
        16000.0 / in_sample_rate as f64,
        10.,
        PolynomialDegree::Septic,
        1024,
        1,
    )?;

    while let Ok(pcm) = rx.recv() {
        buffered_pcm.extend_from_slice(&pcm);
        if buffered_pcm.len() < 10 * in_sample_rate {
            continue;
        }

        let mut resampled_pcm = Vec::new();
        let full_chunks = buffered_pcm.len() / 1024;
        let remainder = buffered_pcm.len() % 1024;
        for chunk in 0..full_chunks {
            let slice = &buffered_pcm[chunk * 1024..(chunk + 1) * 1024];
            match resampler.process(&[slice], None) {
                Ok(samples) => resampled_pcm.extend_from_slice(&samples[0]),
                Err(err) => {
                    send_error(&event_sender, chunk_id, format!("resampler error: {err}"));
                    resampled_pcm.clear();
                    break;
                }
            }
        }

        if remainder == 0 {
            buffered_pcm.clear();
        } else {
            buffered_pcm.copy_within(full_chunks * 1024.., 0);
            buffered_pcm.truncate(remainder);
        }

        if resampled_pcm.is_empty() {
            continue;
        }

        let mel = audio::pcm_to_mel(&config, &resampled_pcm, &mel_filters);
        let mel_len = mel.len();
        let mel = match Tensor::from_vec(
            mel,
            (1, config.num_mel_bins, mel_len / config.num_mel_bins),
            &device,
        ) {
            Ok(tensor) => tensor,
            Err(err) => {
                send_error(
                    &event_sender,
                    chunk_id,
                    format!("tensor build failed: {err}"),
                );
                continue;
            }
        };

        if !language_token_set {
            let language_token =
                resolve_language_token(&cfg, which_model, decoder.model(), &tokenizer, &mel)?;
            decoder.set_language_token(language_token);
            language_token_set = true;
        }

        match decoder.run(&mel, None) {
            Ok(segments) => {
                for segment in segments {
                    let text = segment.dr.text.trim().to_string();
                    let current_chunk = chunk_id;
                    chunk_id += 1;
                    let _ = event_sender.send(TranscriptionEvent::Transcription {
                        chunk_id: current_chunk,
                        text: text.clone(),
                    });
                    if let Some(writer) = transcript_sink.clone() {
                        spawn_append_task(&runtime, writer, current_chunk, text, &event_sender);
                    }
                }
            }
            Err(err) => {
                send_error(&event_sender, chunk_id, err.to_string());
                chunk_id += 1;
            }
        }
        decoder.reset_kv_cache();
    }

    Ok(())
}

fn parse_task_option(value: &Option<String>) -> Result<Option<Task>> {
    value.as_ref().map(|task| Task::from_str(task)).transpose()
}

fn build_device(cpu: bool) -> candle::Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Device::new_cuda(0)
    } else if metal_is_available() {
        Device::new_metal(0)
    } else {
        Ok(Device::Cpu)
    }
}

fn resolve_model_sources(cfg: &OnDeviceConfig, which_model: WhichModel) -> (String, String) {
    let (default_model, default_revision) = if cfg.quantized {
        ("lmz/candle-whisper", "main")
    } else {
        which_model.model_and_revision()
    };
    let default_model = default_model.to_string();
    let default_revision = default_revision.to_string();
    match (&cfg.model_id, &cfg.revision) {
        (Some(model_id), Some(revision)) => (model_id.clone(), revision.clone()),
        (Some(model_id), None) => (model_id.clone(), "main".to_string()),
        (None, Some(revision)) => (default_model, revision.clone()),
        (None, None) => (default_model, default_revision),
    }
}

fn load_model_artifacts(
    cfg: &OnDeviceConfig,
    which_model: WhichModel,
    device: &Device,
) -> Result<(Config, Tokenizer, Model)> {
    let (model_id, revision) = resolve_model_sources(cfg, which_model);
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
    let (config_path, tokenizer_path, weights_path) = if cfg.quantized {
        let ext = which_model
            .quantized_tag()
            .ok_or_else(|| anyhow!("no quantized weights for {:?}", which_model))?;
        (
            repo.get(&format!("config-{ext}.json"))?,
            repo.get(&format!("tokenizer-{ext}.json"))?,
            repo.get(&format!("model-{ext}-q80.gguf"))?,
        )
    } else {
        (
            repo.get("config.json")?,
            repo.get("tokenizer.json")?,
            repo.get("model.safetensors")?,
        )
    };
    let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;
    let model = if cfg.quantized {
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            &weights_path,
            device,
        )?;
        Model::Quantized(m::quantized_model::Whisper::load(&vb, config.clone())?)
    } else {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], m::DTYPE, device)? };
        Model::Normal(m::model::Whisper::load(&vb, config.clone())?)
    };
    Ok((config, tokenizer, model))
}

fn load_mel_filters(config: &Config) -> Result<Vec<f32>> {
    let mel_bytes = match config.num_mel_bins {
        80 => include_bytes!("../whisper/melfilters.bytes").as_slice(),
        128 => include_bytes!("../whisper/melfilters128.bytes").as_slice(),
        nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);
    Ok(mel_filters)
}

fn select_audio_device(preferred: Option<&str>) -> Result<cpal::Device> {
    let host = cpal::default_host();
    match preferred {
        None => host
            .default_input_device()
            .context("failed to find the default audio input device"),
        Some(name) => host
            .input_devices()?
            .find(|dev| dev.name().map_or(false, |n| n == name))
            .ok_or_else(|| anyhow!("failed to find audio input device '{name}'")),
    }
}

fn resolve_language_token(
    cfg: &OnDeviceConfig,
    which_model: WhichModel,
    model: &mut Model,
    tokenizer: &Tokenizer,
    mel: &Tensor,
) -> Result<Option<u32>> {
    match (which_model.is_multilingual(), cfg.language.as_ref()) {
        (true, None) => Ok(Some(multilingual::detect_language(model, tokenizer, mel)?)),
        (false, None) => Ok(None),
        (true, Some(language)) => {
            let token_id = token_id(tokenizer, &format!("<|{language}|>"))?;
            Ok(Some(token_id))
        }
        (false, Some(_)) => anyhow::bail!("a language cannot be set for non-multilingual models"),
    }
}

fn spawn_append_task(
    runtime: &Handle,
    writer: Arc<Mutex<tokio::fs::File>>,
    chunk_id: usize,
    text: String,
    event_sender: &UnboundedSender<TranscriptionEvent>,
) {
    let handle = runtime.clone();
    let event_sender = event_sender.clone();
    handle.spawn(async move {
        let entry = if text.is_empty() {
            "<silence>".to_string()
        } else {
            text
        };
        if let Err(err) = append_transcript(writer, chunk_id, &entry).await {
            let _ = event_sender.send(TranscriptionEvent::Error {
                chunk_id,
                error: format!("File write failed: {err}"),
            });
        }
    });
}

fn send_error(sender: &UnboundedSender<TranscriptionEvent>, chunk_id: usize, error: String) {
    let _ = sender.send(TranscriptionEvent::Error { chunk_id, error });
}
