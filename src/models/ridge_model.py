"""
Ridge Regression Model Implementation.

Ridge 회귀 기반 점수차 예측 모델.
L2 정규화를 통해 다중공선성 문제를 해결하고 안정적인 예측을 제공합니다.
"""

from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.models.base import BaseModel, ModelMetrics, TrainingResult
from src.utils.logger import logger
from config.settings import ModelConfig


class RidgeModel(BaseModel):
    """
    Ridge 회귀 모델.

    특징:
    - L2 정규화로 과적합 방지
    - 다중공선성에 강건
    - 해석 가능한 계수
    - 빠른 학습 속도
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        normalize_features: bool = True,
        alphas_for_cv: Optional[List[float]] = None
    ):
        """
        Args:
            alpha: L2 정규화 강도
            fit_intercept: 절편 학습 여부
            normalize_features: 피처 정규화 여부
            alphas_for_cv: CV로 탐색할 alpha 값들 (None이면 단일 alpha 사용)
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize_features = normalize_features
        self.alphas_for_cv = alphas_for_cv

        self.model: Optional[Ridge] = None
        self.scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = []
        self._training_result: Optional[TrainingResult] = None
        self._best_alpha: float = alpha

    @property
    def name(self) -> str:
        return "ridge"

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "alpha": self._best_alpha,
            "fit_intercept": self.fit_intercept,
            "normalize_features": self.normalize_features
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> "RidgeModel":
        """
        모델 학습.

        Args:
            X: 피처 DataFrame
            y: 타겟 Series
            eval_set: 검증 데이터 (X_val, y_val)

        Returns:
            학습된 모델
        """
        start_time = time.time()

        self._feature_names = X.columns.tolist()

        # 스케일링
        X_scaled = X.copy()
        if self.normalize_features:
            self.scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )

        # CV로 최적 alpha 찾기
        if self.alphas_for_cv is not None:
            self.model = RidgeCV(
                alphas=self.alphas_for_cv,
                fit_intercept=self.fit_intercept,
                cv=5,
                scoring="neg_mean_squared_error"
            )
            self.model.fit(X_scaled, y)
            self._best_alpha = self.model.alpha_
            logger.info(f"Best alpha from CV: {self._best_alpha}")
        else:
            self.model = Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept
            )
            self.model.fit(X_scaled, y)
            self._best_alpha = self.alpha

        training_time = time.time() - start_time

        # 학습 결과 저장
        train_metrics = self.evaluate(X, y)

        val_metrics = None
        if eval_set is not None:
            val_metrics = self.evaluate(eval_set[0], eval_set[1])

        self._training_result = TrainingResult(
            model_name=self.name,
            params=self.params,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            feature_importance=self.get_feature_importance(),
            training_time=training_time,
            n_iterations=1
        )

        logger.info(f"Ridge trained in {training_time:.3f}s")
        logger.info(f"  Train: {train_metrics.summary()}")
        if val_metrics:
            logger.info(f"  Val: {val_metrics.summary()}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측"""
        if self.model is None:
            raise RuntimeError("Model not fitted")

        X_scaled = X.copy()
        if self.normalize_features and self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )

        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        피처 중요도 (계수의 절대값 기반).

        Ridge 회귀의 경우 정규화된 계수의 절대값이 중요도를 나타냅니다.
        """
        if self.model is None:
            return {}

        # 계수의 절대값을 중요도로 사용
        importance = np.abs(self.model.coef_)

        # 정규화 (합이 1이 되도록)
        if importance.sum() > 0:
            importance = importance / importance.sum()

        return dict(zip(self._feature_names, importance))

    def get_coefficients(self) -> Dict[str, float]:
        """
        실제 회귀 계수 반환.

        중요도와 달리 부호가 포함된 실제 계수값입니다.
        """
        if self.model is None:
            return {}

        return dict(zip(self._feature_names, self.model.coef_))

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """상위 N개 피처"""
        importance = self.get_feature_importance()
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]

    @property
    def training_result(self) -> Optional[TrainingResult]:
        """학습 결과"""
        return self._training_result

    @property
    def intercept(self) -> float:
        """절편"""
        if self.model is None:
            return 0.0
        return self.model.intercept_


class ElasticNetModel(BaseModel):
    """
    ElasticNet 회귀 모델.

    L1 + L2 정규화 조합으로 피처 선택과 정규화를 동시에 수행합니다.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        normalize_features: bool = True
    ):
        """
        Args:
            alpha: 전체 정규화 강도
            l1_ratio: L1 비율 (0=Ridge, 1=Lasso)
            fit_intercept: 절편 학습 여부
            normalize_features: 피처 정규화 여부
        """
        from sklearn.linear_model import ElasticNet

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.normalize_features = normalize_features

        self.model: Optional[ElasticNet] = None
        self.scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = []
        self._training_result: Optional[TrainingResult] = None

    @property
    def name(self) -> str:
        return f"elasticnet_l1_{self.l1_ratio}"

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "fit_intercept": self.fit_intercept,
            "normalize_features": self.normalize_features
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> "ElasticNetModel":
        """모델 학습"""
        from sklearn.linear_model import ElasticNet

        start_time = time.time()

        self._feature_names = X.columns.tolist()

        # 스케일링
        X_scaled = X.copy()
        if self.normalize_features:
            self.scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )

        # 학습
        self.model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            max_iter=10000
        )
        self.model.fit(X_scaled, y)

        training_time = time.time() - start_time

        # 학습 결과 저장
        train_metrics = self.evaluate(X, y)

        val_metrics = None
        if eval_set is not None:
            val_metrics = self.evaluate(eval_set[0], eval_set[1])

        self._training_result = TrainingResult(
            model_name=self.name,
            params=self.params,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            feature_importance=self.get_feature_importance(),
            training_time=training_time,
            n_iterations=1
        )

        logger.info(f"ElasticNet trained in {training_time:.3f}s")
        logger.info(f"  Train: {train_metrics.summary()}")
        if val_metrics:
            logger.info(f"  Val: {val_metrics.summary()}")

        # 선택된 피처 수 로깅
        n_selected = np.sum(np.abs(self.model.coef_) > 1e-6)
        logger.info(f"  Selected features: {n_selected}/{len(self._feature_names)}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측"""
        if self.model is None:
            raise RuntimeError("Model not fitted")

        X_scaled = X.copy()
        if self.normalize_features and self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )

        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> Dict[str, float]:
        """피처 중요도"""
        if self.model is None:
            return {}

        importance = np.abs(self.model.coef_)

        if importance.sum() > 0:
            importance = importance / importance.sum()

        return dict(zip(self._feature_names, importance))

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """상위 N개 피처"""
        importance = self.get_feature_importance()
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]

    @property
    def training_result(self) -> Optional[TrainingResult]:
        """학습 결과"""
        return self._training_result


class HuberRegressionModel(BaseModel):
    """
    Huber 회귀 모델.

    이상치에 강건한 선형 회귀로, MAE와 MSE의 장점을 결합합니다.
    """

    def __init__(
        self,
        epsilon: float = 1.35,
        alpha: float = 0.0001,
        fit_intercept: bool = True,
        normalize_features: bool = True
    ):
        """
        Args:
            epsilon: Huber loss 임계값 (작을수록 이상치에 강건)
            alpha: L2 정규화 강도
            fit_intercept: 절편 학습 여부
            normalize_features: 피처 정규화 여부
        """
        from sklearn.linear_model import HuberRegressor

        self.epsilon = epsilon
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize_features = normalize_features

        self.model: Optional[HuberRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = []
        self._training_result: Optional[TrainingResult] = None

    @property
    def name(self) -> str:
        return "huber_regression"

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept,
            "normalize_features": self.normalize_features
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> "HuberRegressionModel":
        """모델 학습"""
        from sklearn.linear_model import HuberRegressor

        start_time = time.time()

        self._feature_names = X.columns.tolist()

        # 스케일링
        X_scaled = X.copy()
        if self.normalize_features:
            self.scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )

        # 학습
        self.model = HuberRegressor(
            epsilon=self.epsilon,
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            max_iter=1000
        )
        self.model.fit(X_scaled, y)

        training_time = time.time() - start_time

        # 학습 결과 저장
        train_metrics = self.evaluate(X, y)

        val_metrics = None
        if eval_set is not None:
            val_metrics = self.evaluate(eval_set[0], eval_set[1])

        self._training_result = TrainingResult(
            model_name=self.name,
            params=self.params,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            feature_importance=self.get_feature_importance(),
            training_time=training_time,
            n_iterations=self.model.n_iter_
        )

        logger.info(f"Huber Regression trained in {training_time:.3f}s")
        logger.info(f"  Train: {train_metrics.summary()}")
        if val_metrics:
            logger.info(f"  Val: {val_metrics.summary()}")

        # 이상치로 감지된 샘플 수
        n_outliers = np.sum(self.model.outliers_)
        logger.info(f"  Detected outliers: {n_outliers}/{len(y)} ({100*n_outliers/len(y):.1f}%)")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측"""
        if self.model is None:
            raise RuntimeError("Model not fitted")

        X_scaled = X.copy()
        if self.normalize_features and self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )

        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> Dict[str, float]:
        """피처 중요도"""
        if self.model is None:
            return {}

        importance = np.abs(self.model.coef_)

        if importance.sum() > 0:
            importance = importance / importance.sum()

        return dict(zip(self._feature_names, importance))

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """상위 N개 피처"""
        importance = self.get_feature_importance()
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]

    @property
    def training_result(self) -> Optional[TrainingResult]:
        """학습 결과"""
        return self._training_result
