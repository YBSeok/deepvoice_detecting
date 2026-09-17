from pathlib import Path

from .config import PreprocessConfig
from .sample import AudioSample
from .steps import build_steps


class PreprocessPipeline:
    def __init__(self, steps=None, config=None):
        self.config = config or PreprocessConfig()
        self.steps = steps if steps is not None else build_steps(self.config)

    def run(self, path) -> AudioSample:
        sample = AudioSample(path=Path(path), config=self.config)
        for step in self.steps:
            sample = step.run(sample)
        return sample
