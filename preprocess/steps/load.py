from .base import PreprocessStep
from ..sample import AudioSample


class LoadStep(PreprocessStep):
    name = "load"

    def run(self, sample: AudioSample) -> AudioSample:
        # TODO: data/test 오디오를 읽는다.
        return sample
