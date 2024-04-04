from typing import Literal

from .abstract_policy import *
from .black_box_policy import *
from .temporal_correlated_policy import *


def policy_factory(typ: Literal["TemporalCorrelatedPolicy", "BlackBoxPolicy"],
                   **kwargs):
    """
    Factory methods to instantiate a policy
    Args:
        typ: policy class type
        **kwargs: keyword arguments

    Returns:

    """
    return eval(typ + "(**kwargs)")
