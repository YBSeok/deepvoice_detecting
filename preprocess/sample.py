from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import PreprocessConfig


@dataclass
class AudioSample:
    path: Path
    waveform: Any = None
    sample_rate: Optional[int] = None
    segments: Any = None
    extras: dict = field(default_factory=dict)
    config: Optional[PreprocessConfig] = None
