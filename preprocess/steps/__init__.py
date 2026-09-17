from .base import PreprocessStep
from .channel import ChannelStep
from .load import LoadStep
from .resample import ResampleStep
from .segment import SegmentStep
from .silence import SilenceStep
from .telephone import TelephoneStep

STEP_ORDER = (
    LoadStep,
    ResampleStep,
    ChannelStep,
    TelephoneStep,
    SegmentStep,
    SilenceStep,
)


def build_steps(config=None):
    return [step_class(config) for step_class in STEP_ORDER]
