from ..sample import AudioSample


class PreprocessStep:
    name = "base"

    def __init__(self, config=None):
        self.config = config

    def run(self, sample: AudioSample) -> AudioSample:
        return sample
