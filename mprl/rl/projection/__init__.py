from typing import Literal
from trust_region_projections.projections.base_projection_layer \
    import BaseProjectionLayer
from trust_region_projections.projections.frob_projection_layer \
    import FrobeniusProjectionLayer
from trust_region_projections.projections.kl_projection_layer\
    import KLProjectionLayer
from trust_region_projections.projections.papi_projection \
    import PAPIProjection
from trust_region_projections.projections.w2_projection_layer \
    import WassersteinProjectionLayer
from trust_region_projections.projections.w2_projection_layer_non_com \
    import WassersteinProjectionLayerNonCommuting
from mprl.util import parse_dtype_device
import torch


def projection_factory(typ: Literal["BaseProjectionLayer",
                                    "FrobeniusProjectionLayer",
                                    "KLProjectionLayer",
                                    "PAPIProjection",
                                    "WassersteinProjectionLayer",
                                    "WassersteinProjectionLayerNonCommuting"],
                       **kwargs):
    """
    Factory methods to instantiate a projector
    Args:
        typ: projection class type
        **kwargs: keyword arguments

    Returns:

    """
    dtype, device = parse_dtype_device(kwargs["dtype"], kwargs["device"])

    kwargs["cpu"] = True if device == torch.device("cpu") else False
    kwargs["dtype"] = dtype
    del kwargs["device"]

    return eval(typ + "(**kwargs)")
