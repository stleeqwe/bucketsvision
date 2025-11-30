"""
Base model interfaces and utilities.

모든 모델의 기반 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.utils.logger import logger


@dataclass
class ModelMetrics:
    """모델 평가 지표"""
    rmse: float
    mae: float
    win_accuracy: float
    within_5_accuracy: float
    within_10_accuracy: float
    mean_error: float = 0.0  # Bias
    std_error: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "win_accuracy": self.win_accuracy,
            "within_5_accuracy": self.within_5_accuracy,
            "within_10_accuracy": self.within_10_accuracy,
            "mean_error": self.mean_error,
            "std_error": self.std_error
        }

    @classmethod
    def from_predictions(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> "ModelMetrics":
        """예측값으로부터 메트릭 계산"""
        errors = y_pred - y_true

        return cls(
            rmse=np.sqrt(mean_squared_error(y_true, y_pred)),
            mae=mean_absolute_error(y_true, y_pred),
            win_accuracy=np.mean(np.sign(y_true) == np.sign(y_pred)),
            within_5_accuracy=np.mean(np.abs(errors) <= 5),
            within_10_accuracy=np.mean(np.abs(errors) <= 10),
            mean_error=np.mean(errors),
            std_error=np.std(errors)
        )

    def summary(self) -> str:
        return (
            f"RMSE: {self.rmse:.3f}, MAE: {self.mae:.3f}, "
            f"Win Acc: {self.win_accuracy:.1%}, "
            f"Within 5: {self.within_5_accuracy:.1%}"
        )


@dataclass
class TrainingResult:
    """학습 결과"""
    model_name: str
    params: Dict[str, Any]
    train_metrics: ModelMetrics
    val_metrics: Optional[ModelMetrics] = None
    feature_importance: Optional[Dict[str, float]] = None
    training_time: float = 0.0
    n_iterations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "params": self.params,
            "train_metrics": self.train_metrics.to_dict(),
            "val_metrics": self.val_metrics.to_dict() if self.val_metrics else None,
            "training_time": self.training_time,
            "n_iterations": self.n_iterations
        }


class BaseModel(ABC):
    """
    모델 기본 추상 클래스.

    모든 모델은 이 클래스를 상속받아 구현합니다.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """모델 이름"""
        pass

    @property
    def params(self) -> Dict[str, Any]:
        """현재 파라미터"""
        return {}

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> "BaseModel":
        """
        모델 학습.

        Args:
            X: 피처 DataFrame
            y: 타겟 Series
            eval_set: 검증 데이터 (X_val, y_val)
            **kwargs: 추가 파라미터

        Returns:
            학습된 모델 (self)
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        예측.

        Args:
            X: 피처 DataFrame

        Returns:
            예측값 배열
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        피처 중요도 반환.

        Returns:
            피처명 -> 중요도 딕셔너리
        """
        pass

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> ModelMetrics:
        """
        모델 평가.

        Args:
            X: 피처 DataFrame
            y: 타겟 Series

        Returns:
            ModelMetrics 객체
        """
        y_pred = self.predict(X)
        return ModelMetrics.from_predictions(y.values, y_pred)

    def save(self, path: Path) -> None:
        """
        모델 저장.

        Args:
            path: 저장 경로 (디렉토리)
        """
        path.mkdir(parents=True, exist_ok=True)

        # 모델 객체 저장
        model_path = path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self, f)

        # 메타데이터 저장
        metadata = {
            "name": self.name,
            "params": self.params,
            "saved_at": datetime.now().isoformat()
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """
        모델 로드.

        Args:
            path: 모델 경로 (디렉토리)

        Returns:
            로드된 모델
        """
        model_path = path / "model.pkl"
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        logger.info(f"Model loaded from {path}")
        return model


class EnsembleModel(BaseModel):
    """
    앙상블 모델.

    여러 모델의 예측을 가중 평균합니다.
    """

    def __init__(
        self,
        models: List[BaseModel],
        weights: Optional[List[float]] = None
    ):
        """
        Args:
            models: 앙상블할 모델 리스트
            weights: 가중치 (None이면 균등)
        """
        self.models = models
        self._weights = weights
        self._fitted = False

    @property
    def name(self) -> str:
        model_names = "+".join(m.name for m in self.models)
        return f"ensemble({model_names})"

    @property
    def weights(self) -> List[float]:
        if self._weights:
            return self._weights
        return [1.0 / len(self.models)] * len(self.models)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> "EnsembleModel":
        """모든 모델 학습"""
        for model in self.models:
            logger.info(f"Training {model.name}...")
            model.fit(X, y, eval_set, **kwargs)

        # 가중치 미지정 시 검증셋 RMSE 기반 계산
        if self._weights is None and eval_set is not None:
            self._calculate_weights(eval_set[0], eval_set[1])

        self._fitted = True
        return self

    def _calculate_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """검증 RMSE 역수 기반 가중치 계산"""
        rmses = []
        for model in self.models:
            metrics = model.evaluate(X_val, y_val)
            rmses.append(metrics.rmse)
            logger.info(f"{model.name} validation RMSE: {metrics.rmse:.3f}")

        # 역수 비례 가중치
        inv_rmses = [1.0 / r for r in rmses]
        total = sum(inv_rmses)
        self._weights = [w / total for w in inv_rmses]

        logger.info(f"Ensemble weights: {dict(zip([m.name for m in self.models], self._weights))}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """가중 평균 예측"""
        predictions = np.zeros(len(X))

        for model, weight in zip(self.models, self.weights):
            predictions += weight * model.predict(X)

        return predictions

    def get_feature_importance(self) -> Dict[str, float]:
        """가중 평균 피처 중요도"""
        combined = {}

        for model, weight in zip(self.models, self.weights):
            importance = model.get_feature_importance()
            for feature, value in importance.items():
                combined[feature] = combined.get(feature, 0) + weight * value

        return combined


# ===================
# Utility Functions
# ===================

def compare_models(
    models: List[BaseModel],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    모델 비교.

    Args:
        models: 비교할 모델 리스트
        X_test: 테스트 피처
        y_test: 테스트 타겟

    Returns:
        비교 결과 DataFrame
    """
    results = []

    for model in models:
        metrics = model.evaluate(X_test, y_test)
        results.append({
            "model": model.name,
            **metrics.to_dict()
        })

    return pd.DataFrame(results).sort_values("rmse")
