from .base import PreprocessStep
from ..sample import AudioSample


class SegmentStep(PreprocessStep):
    name = "segment"

    def run(self, sample: AudioSample) -> AudioSample:
        # TODO: SEGMENT_SAMPLES 단위로 자른다.
        return sample
