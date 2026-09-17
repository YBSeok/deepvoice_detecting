from .base import PreprocessStep
from ..sample import AudioSample


class ChannelStep(PreprocessStep):
    name = "channel"

    def run(self, sample: AudioSample) -> AudioSample:
        # TODO: 모노/스테레오를 헤드 입력 형식에 맞춘다.
        return sample
