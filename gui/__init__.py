# gui/__init__.py
from __future__ import annotations

# digits
from .precision_input import DigitFormat, DigitNumberEdit

# layer widgets + state
from .layer import UnitsSpec, ParamField, LayerState, LayerWidget

# panels
from .layers_panel import LayersPanel

# plots
from .plots import PlotTabsWidget, PlotCanvasQt  # PlotCanvasQt нужен dialogs.py

# dialogs
from .dialogs import (
    PlotDisplaySettingsDialog,
    CRIDialog,
    show_cri_dialog,
    GradientProfileDialog,
    show_gradient_profile_dialog,
)

__all__ = [
    "DigitFormat",
    "DigitNumberEdit",
    "UnitsSpec",
    "ParamField",
    "LayerState",
    "LayerWidget",
    "LayersPanel",
    "PlotTabsWidget",
    "PlotCanvasQt",
    "PlotDisplaySettingsDialog",
    "CRIDialog",
    "show_cri_dialog",
    "GradientProfileDialog",
    "show_gradient_profile_dialog",
]
