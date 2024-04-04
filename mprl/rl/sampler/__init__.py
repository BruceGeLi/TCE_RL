from typing import Literal

from .abstract_sampler import *
from .black_box_sampler import *
from .temporal_correlated_sampler import *


def sampler_factory(typ: Literal["TemporalCorrelatedSampler"],
                    **kwargs):
    """
    Factory methods to instantiate a sampler
    Args:
        typ: sampler class type
        **kwargs: keyword arguments

    Returns:

    """
    return eval(typ + "(**kwargs)")
