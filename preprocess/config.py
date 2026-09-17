from dataclasses import dataclass, field


@dataclass
class PreprocessConfig:
    sample_rate: int = 16_000
    panns_sample_rate: int = 32_000
    segment_samples: int = 64_600
    silence_rms: float = 1e-5
    audio_extensions: tuple[str, ...] = (
        ".aac",
        ".flac",
        ".m4a",
        ".mp3",
        ".ogg",
        ".opus",
        ".wav",
        ".wma",
    )
    extras: dict = field(default_factory=dict)
