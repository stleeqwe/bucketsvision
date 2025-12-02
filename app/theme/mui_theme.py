# BucketsVision MUI Theme Configuration
from .colors import COLORS

MUI_THEME = {
    "palette": {
        "mode": "dark",
        "primary": {
            "main": COLORS["home"],
            "light": "#60a5fa",
            "dark": "#2563eb",
        },
        "secondary": {
            "main": COLORS["away"],
            "light": "#f87171",
            "dark": "#dc2626",
        },
        "background": {
            "default": COLORS["bg_primary"],
            "paper": COLORS["bg_secondary"],
        },
        "text": {
            "primary": COLORS["text_primary"],
            "secondary": COLORS["text_secondary"],
            "disabled": COLORS["text_muted"],
        },
        "success": {
            "main": COLORS["success"],
            "light": "#4ade80",
            "dark": "#16a34a",
        },
        "error": {
            "main": COLORS["error"],
            "light": "#f87171",
            "dark": "#dc2626",
        },
        "warning": {
            "main": COLORS["warning"],
            "light": "#facc15",
            "dark": "#ca8a04",
        },
        "info": {
            "main": COLORS["info"],
            "light": "#60a5fa",
            "dark": "#2563eb",
        },
        "divider": COLORS["border"],
    },
    "shape": {
        "borderRadius": 8,
    },
    "typography": {
        "fontFamily": "'Inter', 'SF Pro', -apple-system, BlinkMacSystemFont, sans-serif",
        "h1": {"fontSize": "2rem", "fontWeight": 700},
        "h2": {"fontSize": "1.5rem", "fontWeight": 600},
        "h3": {"fontSize": "1.25rem", "fontWeight": 600},
        "h4": {"fontSize": "1.125rem", "fontWeight": 600},
        "body1": {"fontSize": "1rem"},
        "body2": {"fontSize": "0.875rem"},
        "caption": {"fontSize": "0.75rem"},
    },
    "components": {
        "MuiPaper": {
            "styleOverrides": {
                "root": {
                    "backgroundImage": "none",
                    "border": f"1px solid {COLORS['border']}",
                },
            },
        },
        "MuiChip": {
            "styleOverrides": {
                "root": {
                    "borderRadius": "4px",
                    "fontWeight": 500,
                },
            },
        },
        "MuiButton": {
            "styleOverrides": {
                "root": {
                    "textTransform": "none",
                    "fontWeight": 600,
                },
            },
        },
        "MuiTableCell": {
            "styleOverrides": {
                "root": {
                    "borderBottom": f"1px solid {COLORS['border_muted']}",
                },
            },
        },
    },
}

# Nivo chart theme (matches MUI dark theme)
NIVO_THEME = {
    "background": "transparent",
    "textColor": COLORS["text_secondary"],
    "fontSize": 12,
    "axis": {
        "domain": {
            "line": {
                "stroke": COLORS["border"],
                "strokeWidth": 1,
            },
        },
        "ticks": {
            "line": {
                "stroke": COLORS["border"],
                "strokeWidth": 1,
            },
            "text": {
                "fill": COLORS["text_secondary"],
            },
        },
    },
    "grid": {
        "line": {
            "stroke": COLORS["border_muted"],
            "strokeWidth": 1,
        },
    },
    "tooltip": {
        "container": {
            "background": COLORS["bg_tertiary"],
            "color": COLORS["text_primary"],
            "fontSize": 12,
            "borderRadius": 4,
            "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.3)",
        },
    },
}

def get_theme_provider():
    """Returns theme configuration for streamlit-elements"""
    return {
        "theme": MUI_THEME,
        "nivo": NIVO_THEME,
    }
