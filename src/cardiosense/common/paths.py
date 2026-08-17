"""Project path resolution.

Everything in CardioSense addresses files relative to the project root, which is
discovered at import time by walking up from this file until a marker
(``pyproject.toml`` or ``.git``) is found. That means no absolute path is ever
hard-coded, and the identical code runs on a laptop, on Colab and in CI.

The one path that is genuinely machine-specific is the raw-data root, because
the datasets live on Google Drive during Colab work. Override it with the
``CARDIOSENSE_DATA_ROOT`` environment variable::

    import os
    os.environ["CARDIOSENSE_DATA_ROOT"] = "/content/drive/MyDrive/CardioSense/data"
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ProjectPaths", "PATHS", "get_project_root", "resolve_path", "ensure_dir"]

_ROOT_MARKERS = ("pyproject.toml", ".git")
_DATA_ROOT_ENV = "CARDIOSENSE_DATA_ROOT"
_PROJECT_ROOT_ENV = "CARDIOSENSE_PROJECT_ROOT"


def get_project_root(start: Path | str | None = None) -> Path:
    """Return the project root directory.

    Resolution order:

    1. ``$CARDIOSENSE_PROJECT_ROOT`` if set (useful for odd Colab layouts).
    2. The first ancestor of *start* containing ``pyproject.toml`` or ``.git``.
    3. Three levels above this file (``src/cardiosense/common`` -> root),
       which is correct for a plain source checkout.

    Args:
        start: Directory to begin searching from. Defaults to this file's dir.

    Returns:
        Absolute path to the project root.
    """
    env_root = os.environ.get(_PROJECT_ROOT_ENV)
    if env_root:
        return Path(env_root).expanduser().resolve()

    here = Path(start).expanduser().resolve() if start else Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate

    # src/cardiosense/common/paths.py -> src/cardiosense/common -> src/cardiosense
    # -> src -> root
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical locations for every artifact CardioSense reads or writes."""

    root: Path
    data: Path
    models: Path
    results: Path
    configs: Path
    docs: Path
    notebooks: Path
    experiments: Path

    @classmethod
    def create(cls, root: Path | str | None = None) -> "ProjectPaths":
        """Build a :class:`ProjectPaths`, honouring ``$CARDIOSENSE_DATA_ROOT``."""
        root_path = Path(root).expanduser().resolve() if root else get_project_root()

        data_env = os.environ.get(_DATA_ROOT_ENV)
        data_path = Path(data_env).expanduser().resolve() if data_env else root_path / "data"

        return cls(
            root=root_path,
            data=data_path,
            models=root_path / "models",
            results=root_path / "results",
            configs=root_path / "configs",
            docs=root_path / "docs",
            notebooks=root_path / "notebooks",
            experiments=root_path / "results" / "experiments",
        )

    # -- convenience accessors ------------------------------------------------
    def data_for(self, modality: str) -> Path:
        """Raw/derived data directory for ``clinical`` | ``ecg`` | ``xray``."""
        return self.data / modality

    def models_for(self, modality: str) -> Path:
        return self.models / modality

    def results_for(self, modality: str) -> Path:
        return self.results / modality

    def ensure_all(self) -> "ProjectPaths":
        """Create every standard directory. Safe to call repeatedly."""
        for base in (self.data, self.models, self.results):
            for modality in ("clinical", "ecg", "xray"):
                (base / modality).mkdir(parents=True, exist_ok=True)
        self.experiments.mkdir(parents=True, exist_ok=True)
        (self.results / "ecg" / "explanations").mkdir(parents=True, exist_ok=True)
        (self.results / "xray" / "gradcam").mkdir(parents=True, exist_ok=True)
        return self

    def describe(self) -> str:
        return "\n".join(
            f"  {name:<12} {getattr(self, name)}"
            for name in ("root", "data", "models", "results", "configs", "docs", "experiments")
        )


#: Module-level singleton. Import this rather than re-deriving paths.
PATHS = ProjectPaths.create()


def resolve_path(path: Path | str, base: Path | None = None) -> Path:
    """Resolve *path* against the project root unless it is already absolute.

    Args:
        path: Absolute or project-relative path.
        base: Base directory. Defaults to the project root.

    Returns:
        An absolute :class:`~pathlib.Path`.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return ((base or PATHS.root) / candidate).resolve()


def ensure_dir(path: Path | str) -> Path:
    """Create *path* (and parents) if needed and return it as a ``Path``."""
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory
