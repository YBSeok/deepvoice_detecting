from .base import PreprocessStep
from ..sample import AudioSample


class TelephoneStep(PreprocessStep):
    name = "telephone"

    def run(self, sample: AudioSample) -> AudioSample:
        # TODO: 전화채널 오디오 보정.
        return sample
