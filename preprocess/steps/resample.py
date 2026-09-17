from .base import PreprocessStep
from ..sample import AudioSample


class ResampleStep(PreprocessStep):
    name = "resample"

    def run(self, sample: AudioSample) -> AudioSample:
        # TODO: 모델별 샘플링레이트로 맞춘다.
        return sample
