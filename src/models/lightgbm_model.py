"""
LightGBM Model Implementation.

LightGBM 기반 점수차 예측 모델.
XGBoost 대비 빠른 학습 속도와 메모리 효율성이 특징입니다.
"""

from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.models.base import BaseModel, ModelMetrics, TrainingResult
from src.utils.logger import logger
from config.settings import ModelConfig


class LightGBMModel(BaseModel):
    """
    LightGBM 회귀 모델.

    Huber loss를 사용하여 이상치에 robust한 학습을 수행합니다.
    GOSS(Gradient-based One-Side Sampling) 기반 효율적 학습.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        use_huber: bool = True,
        huber_delta: float = 10.0
    ):
        """
        Args:
            params: LightGBM 파라미터
            use_huber: Huber loss 사용 여부
            huber_delta: Huber alpha 값
        """
        self._params = params or self._default_params(use_huber, huber_delta)
        self.model: Optional[lgb.LGBMRegressor] = None
        self._feature_names: List[str] = []
        self._training_result: Optional[TrainingResult] = None

    @property
    def name(self) -> str:
        return "lightgbm"

    @property
    def params(self) -> Dict[str, Any]:
        return self._params.copy()

    def _default_params(self, use_huber: bool, huber_delta: float) -> Dict[str, Any]:
        """기본 파라미터"""
        params = ModelConfig.LIGHTGBM_DEFAULT_PARAMS.copy()

        if use_huber:
            params["objective"] = "huber"
            params["alpha"] = huber_delta
        else:
            params["objective"] = "regression"

        return params

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = False,
        **kwargs
    ) -> "LightGBMModel":
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

        # 모델 생성
        self.model = lgb.LGBMRegressor(**self._params)

        # 학습 설정
        callbacks = []
        if not verbose:
            callbacks.append(lgb.log_evaluation(period=0))

        fit_params = {
            "callbacks": callbacks
        }

        if eval_set is not None:
            fit_params["eval_set"] = [eval_set]
            fit_params["eval_metric"] = "rmse"

            # Early stopping callback
            callbacks.append(
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose)
            )

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
            n_iterations=self.model.best_iteration_ if hasattr(self.model, 'best_iteration_') and self.model.best_iteration_ else self._params.get('n_estimators', 0)
        )

        logger.info(f"LightGBM trained in {training_time:.1f}s")
        logger.info(f"  Train: {train_metrics.summary()}")
        if val_metrics:
            logger.info(f"  Val: {val_metrics.summary()}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측"""
        if self.model is None:
            raise RuntimeError("Model not fitted")

        return self.model.predict(X)

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """
        피처 중요도.

        Args:
            importance_type: 'gain', 'split', 'weight' 중 선택

        Returns:
            피처명 -> 중요도 딕셔너리
        """
        if self.model is None:
            return {}

        # LightGBM importance_type 매핑
        lgb_importance_type = importance_type
        if importance_type == "weight":
            lgb_importance_type = "split"

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


class LightGBMModelWithCV(LightGBMModel):
    """
    교차검증 기반 LightGBM 모델.

    lgb.cv를 사용하여 최적 iterations을 찾습니다.
    """

    def fit_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
        early_stopping_rounds: int = 50,
        stratified: bool = False,
        **kwargs
    ) -> "LightGBMModelWithCV":
        """
        교차검증으로 학습.

        Args:
            X: 피처 DataFrame
            y: 타겟 Series
            n_folds: Fold 수
            early_stopping_rounds: Early stopping rounds
            stratified: Stratified split 사용 여부

        Returns:
            학습된 모델
        """
        self._feature_names = X.columns.tolist()

        # Dataset 생성
        train_data = lgb.Dataset(X, label=y)

        # CV 파라미터
        params = self._params.copy()
        n_estimators = params.pop("n_estimators", 1000)

        # 교차검증
        cv_results = lgb.cv(
            params,
            train_data,
            num_boost_round=n_estimators,
            nfold=n_folds,
            stratified=stratified,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0)
            ],
            return_cvbooster=False
        )

        # 최적 iterations
        # cv_results keys: 'valid rmse-mean', 'valid rmse-stdv' 등
        metric_key = None
        for key in cv_results.keys():
            if 'mean' in key:
                metric_key = key
                break

        if metric_key:
            best_iteration = len(cv_results[metric_key])
            std_key = metric_key.replace('mean', 'stdv')

            logger.info(f"CV best iteration: {best_iteration}")
            logger.info(f"CV RMSE: {cv_results[metric_key][-1]:.3f} +/- {cv_results.get(std_key, [0])[-1]:.3f}")
        else:
            best_iteration = n_estimators

        # 전체 데이터로 재학습
        self._params["n_estimators"] = best_iteration
        self.model = lgb.LGBMRegressor(**self._params)
        self.model.fit(X, y, callbacks=[lgb.log_evaluation(period=0)])

        return self


class LightGBMDARTModel(LightGBMModel):
    """
    DART(Dropouts meet Multiple Additive Regression Trees) 기반 LightGBM.

    Dropout을 적용하여 과적합을 방지합니다.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        drop_rate: float = 0.1,
        max_drop: int = 50,
        skip_drop: float = 0.5
    ):
        """
        Args:
            params: LightGBM 파라미터
            drop_rate: 드롭아웃 비율
            max_drop: 최대 드롭 트리 수
            skip_drop: 드롭 스킵 확률
        """
        base_params = params or ModelConfig.LIGHTGBM_DEFAULT_PARAMS.copy()

        # DART 설정
        base_params["boosting_type"] = "dart"
        base_params["drop_rate"] = drop_rate
        base_params["max_drop"] = max_drop
        base_params["skip_drop"] = skip_drop

        super().__init__(params=base_params, use_huber=True)

    @property
    def name(self) -> str:
        return "lightgbm_dart"
