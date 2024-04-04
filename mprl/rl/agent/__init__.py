from typing import Literal

from .abstract_agent import *
from .temporal_correlated_agent import *
from .black_box_agent import *


def agent_factory(typ: Literal["TemporalCorrelatedAgent"],
                  **kwargs):
    """
    Factory methods to instantiate an agent
    Args:
        typ: agent class type
        **kwargs: keyword arguments

    Returns:

    """
    return eval(typ + "(**kwargs)")
