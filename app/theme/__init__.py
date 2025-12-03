# BucketsVision Theme System
from .colors import COLORS, SPACING, TYPOGRAPHY
from .mui_theme import MUI_THEME, get_theme_provider
from .styles import (
    inject_main_styles,
    inject_all_styles,
    render_header,
    render_footer,
    MAIN_CSS,
)

__all__ = [
    'COLORS',
    'SPACING',
    'TYPOGRAPHY',
    'MUI_THEME',
    'get_theme_provider',
    'inject_main_styles',
    'inject_all_styles',
    'render_header',
    'render_footer',
    'MAIN_CSS',
]
