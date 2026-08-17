"""CardioSense — Phase 1.

Three independent AI pipelines for cardiovascular decision support:

* ``cardiosense.clinical`` — tabular risk prediction (UCI Heart Disease)
* ``cardiosense.ecg``      — 12-lead ECG interpretation (PTB-XL)
* ``cardiosense.xray``     — chest X-ray cardiomegaly detection (NIH ChestX-ray14)

Multimodal fusion, RAG, LLM report generation and the dashboard are Phase 2 and
are intentionally absent from this package.
"""

__version__ = "0.1.0"
__phase__ = 1

from .common.config import Config, load_config  # noqa: F401
from .common.env import get_device, print_environment  # noqa: F401
from .common.paths import PATHS  # noqa: F401
from .common.seeding import set_seed  # noqa: F401

__all__ = ["Config", "load_config", "get_device", "print_environment", "PATHS", "set_seed",
           "__version__", "__phase__"]
