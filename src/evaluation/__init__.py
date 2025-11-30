"""
Model Evaluation Module.

Model evaluation, cross-validation, and performance analysis.
"""

from src.evaluation.metrics import (
    calculate_metrics,
    calculate_betting_metrics,
    calculate_calibration_metrics,
    MetricsReport
)
from src.evaluation.cross_validation import (
    TimeSeriesCV,
    SeasonBasedCV,
    WalkForwardCV,
    cross_validate_model
)
from src.evaluation.analysis import (
    ErrorAnalyzer,
    PredictionAnalyzer,
    FeatureImportanceAnalyzer
)

__all__ = [
    # Metrics
    "calculate_metrics",
    "calculate_betting_metrics",
    "calculate_calibration_metrics",
    "MetricsReport",
    # Cross Validation
    "TimeSeriesCV",
    "SeasonBasedCV",
    "WalkForwardCV",
    "cross_validate_model",
    # Analysis
    "ErrorAnalyzer",
    "PredictionAnalyzer",
    "FeatureImportanceAnalyzer"
]
