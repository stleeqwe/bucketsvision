"""
XGBoost Model Implementation.

XGBoost 기반 점수차 예측 모델.
"""

from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.base import BaseModel, ModelMetrics, TrainingResult
from src.utils.logger import logger
from config.settings import ModelConfig


class XGBoostModel(BaseModel):
    """
    XGBoost 회귀 모델.

    Huber loss를 사용하여 이상치에 robust한 학습을 수행합니다.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        use_huber: bool = True,
        huber_delta: float = 10.0
    ):
        """
        Args:
            params: XGBoost 파라미터
            use_huber: Huber loss 사용 여부
            huber_delta: Huber delta 값
        """
        self._params = params or self._default_params(use_huber, huber_delta)
        self.model: Optional[xgb.XGBRegressor] = None
        self._feature_names: List[str] = []
        self._training_result: Optional[TrainingResult] = None

    @property
    def name(self) -> str:
        return "xgboost"

    @property
    def params(self) -> Dict[str, Any]:
        return self._params.copy()

    def _default_params(self, use_huber: bool, huber_delta: float) -> Dict[str, Any]:
        """기본 파라미터"""
        params = ModelConfig.XGBOOST_DEFAULT_PARAMS.copy()

        if use_huber:
            params["objective"] = "reg:pseudohubererror"
            params["huber_slope"] = huber_delta
        else:
            params["objective"] = "reg:squarederror"

        return params

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = False,
        **kwargs
    ) -> "XGBoostModel":
        """
        모델 학습.

        Args:
            X: 피처 DataFrame
            y: 타겟 Series
            eval_set: 검증 데이터 (X_val, y_val)
            early_stopping_rounds: Early stopping rounds
            verbose: 상세 출력 여부

        Returns:
            학습된 모델
        """
        start_time = time.time()

        self._feature_names = X.columns.tolist()

        # 모델 파라미터에 early_stopping_rounds 추가
        model_params = self._params.copy()
        if eval_set is not None:
            model_params["early_stopping_rounds"] = early_stopping_rounds

        # 모델 생성
        self.model = xgb.XGBRegressor(**model_params)

        # 학습 설정
        fit_params = {"verbose": verbose}

        if eval_set is not None:
            fit_params["eval_set"] = [eval_set]

        # 학습
        self.model.fit(X, y, **fit_params)

        training_time = time.time() - start_time

        # 학습 결과 저장
        train_metrics = self.evaluate(X, y)

        val_metrics = None
        if eval_set is not None:
            val_metrics = self.evaluate(eval_set[0], eval_set[1])

        self._training_result = TrainingResult(
            model_name=self.name,
            params=self._params,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            feature_importance=self.get_feature_importance(),
            training_time=training_time,
            n_iterations=self.model.best_iteration if hasattr(self.model, 'best_iteration') else self._params.get('n_estimators', 0)
        )

        logger.info(f"XGBoost trained in {training_time:.1f}s")
        logger.info(f"  Train: {train_metrics.summary()}")
        if val_metrics:
            logger.info(f"  Val: {val_metrics.summary()}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측"""
        if self.model is None:
            raise RuntimeError("Model not fitted")

        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """피처 중요도 (gain 기준)"""
        if self.model is None:
            return {}

        importance = self.model.feature_importances_

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


class XGBoostModelWithCV(XGBoostModel):
    """
    교차검증 기반 XGBoost 모델.

    xgb.cv를 사용하여 최적 iterations을 찾습니다.
    """

    def fit_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
        early_stopping_rounds: int = 50,
        **kwargs
    ) -> "XGBoostModelWithCV":
        """
        교차검증으로 학습.

        Args:
            X: 피처 DataFrame
            y: 타겟 Series
            n_folds: Fold 수
            early_stopping_rounds: Early stopping rounds

        Returns:
            학습된 모델
        """
        self._feature_names = X.columns.tolist()

        # DMatrix 생성
        dtrain = xgb.DMatrix(X, label=y)

        # CV 파라미터
        params = self._params.copy()
        n_estimators = params.pop("n_estimators", 1000)

        # 교차검증
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=n_estimators,
            nfold=n_folds,
            early_stopping_rounds=early_stopping_rounds,
            metrics="rmse",
            as_pandas=True,
            seed=params.get("random_state", 42)
        )

        # 최적 iterations
        best_iteration = len(cv_results)
        logger.info(f"CV best iteration: {best_iteration}")
        logger.info(f"CV RMSE: {cv_results['test-rmse-mean'].iloc[-1]:.3f} +/- {cv_results['test-rmse-std'].iloc[-1]:.3f}")

        # 전체 데이터로 재학습
        self._params["n_estimators"] = best_iteration
        self.model = xgb.XGBRegressor(**self._params)
        self.model.fit(X, y)

        return self
