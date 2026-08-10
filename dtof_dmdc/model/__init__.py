import importlib
from typing import *

if TYPE_CHECKING:
    from .moge_v1 import MoGeModel as MoGeModelV1
    from .moge_v2 import MoGeModel as MoGeModelV2
    from .dmdc import MoGeModel as DtofDMDCModel

# version string -> module (same package). Each module exposes a class named `MoGeModel`.
# Add future output configurations of our model here (e.g. 'dmdc_v2': '.dmdc_v2').
_MODEL_VERSIONS = {
    'moge_v1': '.moge_v1',   # MoGe   (baseline)
    'moge_v2': '.moge_v2',   # MoGe   (baseline)
    'dmdc':    '.dmdc',      # dToF-DMDC (ours)
}


def import_model_class_by_version(version: str) -> Type[Union['MoGeModelV1', 'MoGeModelV2', 'DtofDMDCModel']]:
    if version not in _MODEL_VERSIONS:
        raise ValueError(f'Unsupported model version: {version}. Choose from {list(_MODEL_VERSIONS)}')
    module = importlib.import_module(_MODEL_VERSIONS[version], __package__)
    return getattr(module, 'MoGeModel')
