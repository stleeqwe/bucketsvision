"""
NBA Score Prediction Models.

Point differential prediction ML model module.
"""

from src.models.base import (
    BaseModel,
    EnsembleModel,
    ModelMetrics,
    TrainingResult,
    compare_models
)
from src.models.xgboost_model import (
    XGBoostModel,
    XGBoostModelWithCV
)
from src.models.lightgbm_model import (
    LightGBMModel,
    LightGBMModelWithCV,
    LightGBMDARTModel
)
from src.models.ridge_model import (
    RidgeModel,
    ElasticNetModel,
    HuberRegressionModel
)

__all__ = [
    # Base
    "BaseModel",
    "EnsembleModel",
    "ModelMetrics",
    "TrainingResult",
    "compare_models",
    # XGBoost
    "XGBoostModel",
    "XGBoostModelWithCV",
    # LightGBM
    "LightGBMModel",
    "LightGBMModelWithCV",
    "LightGBMDARTModel",
    # Linear
    "RidgeModel",
    "ElasticNetModel",
    "HuberRegressionModel",
]
