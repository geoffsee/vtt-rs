use anyhow::{Context, Result};
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
};

const DEFAULT_CONFIG_PATH: &str = "vtt.config.json";
const DEFAULT_CHUNK_DURATION_SECS: usize = 5;
const DEFAULT_MODEL: &str = "whisper-1";
const DEFAULT_ENDPOINT: &str = "https://api.openai.com/v1/audio/transcriptions";

/// Configuration for the transcription service.
///
/// This struct controls how audio is captured, chunked, and sent to the transcription API.
/// All fields have sensible defaults and can be loaded from a JSON configuration file.
///
/// # Examples
///
/// ## Using defaults
///
/// ```
/// use vtt_rs::Config;
///
/// let config = Config::default();
/// assert_eq!(config.chunk_duration_secs, 5);
/// assert_eq!(config.model, "whisper-1");
/// ```
///
/// ## Custom configuration
///
/// ```
/// use vtt_rs::Config;
/// use std::path::PathBuf;
///
/// let config = Config {
///     chunk_duration_secs: 3,
///     model: "whisper-1".to_string(),
///     endpoint: "https://api.openai.com/v1/audio/transcriptions".to_string(),
///     out_file: Some(PathBuf::from("output.log")),
/// };
/// ```
///
/// ## Loading from JSON file
///
/// ```no_run
/// use vtt_rs::Config;
///
/// # fn main() -> anyhow::Result<()> {
/// let config = Config::from_file("config.json")?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct Config {
    /// Duration of each audio chunk in seconds.
    ///
    /// Smaller values (2-3 seconds) provide faster response times but may reduce accuracy.
    /// Larger values (5-10 seconds) improve accuracy but increase latency.
    pub chunk_duration_secs: usize,

    /// OpenAI model to use for transcription.
    ///
    /// Common values: `"whisper-1"` for OpenAI's Whisper model.
    pub model: String,

    /// API endpoint for transcription requests.
    ///
    /// Defaults to OpenAI's endpoint but can be changed to support local Whisper
    /// instances or other OpenAI-compatible APIs.
    pub endpoint: String,

    /// Optional file path to append transcription logs.
    ///
    /// When set, all transcriptions (including silence markers) will be written
    /// to this file in addition to being sent via events. Use [`None`] to disable
    /// file logging.
    pub out_file: Option<PathBuf>,

    /// Optional on-device configuration.
    ///
    /// When present and `enabled`, the service will run Whisper locally using
    /// the bundled Candle integration instead of calling a remote API.
    pub on_device: Option<OnDeviceConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            chunk_duration_secs: DEFAULT_CHUNK_DURATION_SECS,
            model: DEFAULT_MODEL.to_string(),
            endpoint: DEFAULT_ENDPOINT.to_string(),
            out_file: None,
            on_device: None,
        }
    }
}

impl Config {
    /// Loads configuration from a JSON file at the specified path.
    ///
    /// The JSON file should contain fields matching the [`Config`] struct.
    /// Missing fields will use their default values via serde's `#[serde(default)]`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use vtt_rs::Config;
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let config = Config::from_file("my_config.json")?;
    /// println!("Using model: {}", config.model);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be read
    /// - The file contains invalid JSON
    /// - The JSON structure doesn't match the expected format
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path)
            .with_context(|| format!("reading config from {}", path.display()))?;
        serde_json::from_str(&contents)
            .with_context(|| format!("parsing config from {}", path.display()))
    }

    /// Attempts to load configuration from `vtt.config.json` in the current directory.
    ///
    /// If the default config file exists, it will be loaded. Otherwise, returns
    /// default configuration values.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use vtt_rs::Config;
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let (config, source_path) = Config::load_or_default()?;
    ///
    /// if let Some(path) = source_path {
    ///     println!("Loaded config from: {}", path.display());
    /// } else {
    ///     println!("Using default configuration");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the default config file exists but cannot be read or parsed.
    pub fn load_or_default() -> Result<(Self, Option<PathBuf>)> {
        let default_path = PathBuf::from(DEFAULT_CONFIG_PATH);
        if default_path.exists() {
            let config = Self::from_file(&default_path)?;
            Ok((config, Some(default_path)))
        } else {
            Ok((Self::default(), None))
        }
    }

    /// Resolves the output file path relative to the configuration file's location.
    ///
    /// This method handles both absolute and relative paths in `out_file`:
    /// - Absolute paths are returned as-is
    /// - Relative paths are resolved relative to the config file's directory
    /// - If no source path is provided, relative paths remain unchanged
    ///
    /// # Examples
    ///
    /// ```
    /// use vtt_rs::Config;
    /// use std::path::{Path, PathBuf};
    ///
    /// let mut config = Config::default();
    /// config.out_file = Some(PathBuf::from("logs/output.log"));
    ///
    /// let source = Path::new("/etc/vtt/config.json");
    /// let resolved = config.resolve_out_path(Some(source));
    ///
    /// assert_eq!(resolved, Some(PathBuf::from("/etc/vtt/logs/output.log")));
    /// ```
    pub fn resolve_out_path(&self, source: Option<&Path>) -> Option<PathBuf> {
        self.out_file.as_ref().map(|path| {
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

    /// Returns true when on-device transcription is enabled.
    pub fn uses_on_device(&self) -> bool {
        self.on_device
            .as_ref()
            .map(|cfg| cfg.enabled)
            .unwrap_or(false)
    }

    /// Returns the enabled on-device configuration if present.
    pub fn on_device_config(&self) -> Option<&OnDeviceConfig> {
        self.on_device.as_ref().filter(|cfg| cfg.enabled)
    }
}

/// Configuration for the on-device Whisper backend.
#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct OnDeviceConfig {
    /// Whether to enable the on-device backend.
    pub enabled: bool,
    /// Force CPU execution instead of GPU.
    pub cpu: bool,
    /// Predefined Whisper model identifier (e.g. `"tiny.en"`).
    pub model: String,
    /// Optional custom Hugging Face model id.
    pub model_id: Option<String>,
    /// Optional revision for the custom model id.
    pub revision: Option<String>,
    /// Use quantized weights where available (tiny/tiny.en).
    pub quantized: bool,
    /// RNG seed passed to the decoder.
    pub seed: u64,
    /// Optional forced language token (e.g. `"en"`).
    pub language: Option<String>,
    /// Optional decoding task (`"transcribe"` or `"translate"`).
    pub task: Option<String>,
    /// Emit timestamped segments.
    pub timestamps: bool,
    /// Print verbose logs from the decoder.
    pub verbose: bool,
    /// Preferred audio input device name.
    pub audio_device: Option<String>,
}

impl Default for OnDeviceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cpu: true,
            model: "tiny.en".to_string(),
            model_id: None,
            revision: None,
            quantized: false,
            seed: 299_792_458,
            language: None,
            task: None,
            timestamps: false,
            verbose: false,
            audio_device: None,
        }
    }
}
