from .base import PreprocessStep
from ..sample import AudioSample


class SilenceStep(PreprocessStep):
    name = "silence"

    def run(self, sample: AudioSample) -> AudioSample:
        # TODO: 침묵 구간 표시. RMS < silence_rms 이면 fake 0.0.
        return sample
