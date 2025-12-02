# BucketsVision Color System
# Dark Theme Design Tokens

COLORS = {
    # Background
    "bg_primary": "#0d1117",      # Main background
    "bg_secondary": "#161b22",    # Card background
    "bg_tertiary": "#21262d",     # Section background

    # Text (밝게 조정)
    "text_primary": "#ffffff",    # Primary text - 흰색
    "text_secondary": "#e6edf3",  # Secondary text - 밝은 회색
    "text_muted": "#c9d1d9",      # Muted text - 중간 밝기 회색

    # Team Colors
    "home": "#3b82f6",            # Home team (blue)
    "away": "#ef4444",            # Away team (red)

    # Status
    "success": "#22c55e",         # Win/Correct
    "error": "#ef4444",           # Lose/Fail
    "warning": "#eab308",         # Warning/GTD
    "info": "#3b82f6",            # Info

    # Border (밝게 조정)
    "border": "#ffffff",          # 흰색 구분선
    "border_muted": "#c9d1d9",    # 밝은 회색

    # Gradient Backgrounds (for results)
    "success_gradient": "linear-gradient(145deg, #1a2e1a 0%, #161b22 100%)",
    "error_gradient": "linear-gradient(145deg, #2e1a1a 0%, #161b22 100%)",
}

SPACING = {
    "xs": "4px",
    "sm": "8px",
    "md": "16px",
    "lg": "24px",
    "xl": "32px",
}

TYPOGRAPHY = {
    "h1": {"fontSize": "2rem", "fontWeight": 700},
    "h2": {"fontSize": "1.5rem", "fontWeight": 600},
    "h3": {"fontSize": "1.25rem", "fontWeight": 600},
    "body1": {"fontSize": "1rem", "fontWeight": 400},
    "body2": {"fontSize": "0.875rem", "fontWeight": 400},
    "caption": {"fontSize": "0.75rem", "fontWeight": 400},
    "score": {"fontSize": "2.5rem", "fontWeight": 800},
    "percentage": {"fontSize": "1.5rem", "fontWeight": 700},
}

# Helper functions
def get_team_color(is_home: bool) -> str:
    return COLORS["home"] if is_home else COLORS["away"]

def get_result_color(is_correct: bool) -> str:
    return COLORS["success"] if is_correct else COLORS["error"]

def get_result_gradient(is_correct: bool) -> str:
    return COLORS["success_gradient"] if is_correct else COLORS["error_gradient"]

def get_edge_color(edge: float) -> str:
    if edge >= 5:
        return COLORS["success"]
    elif edge >= 2:
        return "#4ade80"  # Light green
    elif edge <= -5:
        return COLORS["error"]
    elif edge <= -2:
        return "#f87171"  # Light red
    return COLORS["text_secondary"]
