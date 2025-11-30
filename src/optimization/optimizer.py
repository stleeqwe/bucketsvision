"""
Optuna Hyperparameter Optimizer.

Optuna를 활용한 체계적인 하이퍼파라미터 최적화.
시계열 교차검증과 베이지안 최적화를 결합합니다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from pathlib import Path
import json

import numpy as np
import pandas as pd
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.models.base import BaseModel, ModelMetrics
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.ridge_model import RidgeModel
from src.utils.logger import logger


@dataclass
class OptimizationResult:
    """최적화 결과"""
    model_name: str
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    cv_scores: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "cv_scores": self.cv_scores
        }

    def save(self, path: Path) -> None:
        """결과 저장"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class OptunaOptimizer(ABC):
    """
    Optuna 기반 하이퍼파라미터 최적화 기본 클래스.

    특징:
    - TPE(Tree-structured Parzen Estimator) 기반 베이지안 최적화
    - 시계열 교차검증 지원
    - 조기 종료(Pruning) 지원
    - 재현 가능한 결과를 위한 시드 설정
    """

    def __init__(
        self,
        n_trials: int = 100,
        cv_folds: int = 5,
        scoring: str = "neg_root_mean_squared_error",
        direction: str = "minimize",
        seed: int = 42,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None
    ):
        """
        Args:
            n_trials: 최적화 시도 횟수
            cv_folds: 교차검증 fold 수
            scoring: 평가 지표 ('neg_root_mean_squared_error', 'neg_mean_absolute_error')
            direction: 최적화 방향 ('minimize', 'maximize')
            seed: 랜덤 시드
            timeout: 최대 실행 시간(초)
            study_name: Optuna study 이름
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.direction = direction
        self.seed = seed
        self.timeout = timeout
        self.study_name = study_name or self.model_name

        self._study: Optional[optuna.Study] = None
        self._X: Optional[pd.DataFrame] = None
        self._y: Optional[pd.Series] = None

    @property
    @abstractmethod
    def model_name(self) -> str:
        """모델 이름"""
        pass

    @abstractmethod
    def _suggest_params(self, trial: Trial) -> Dict[str, Any]:
        """
        Optuna trial에서 파라미터 제안.

        Args:
            trial: Optuna trial 객체

        Returns:
            파라미터 딕셔너리
        """
        pass

    @abstractmethod
    def _create_model(self, params: Dict[str, Any]) -> BaseModel:
        """
        파라미터로 모델 생성.

        Args:
            params: 하이퍼파라미터

        Returns:
            모델 인스턴스
        """
        pass

    def _objective(self, trial: Trial) -> float:
        """
        Optuna 목적 함수.

        Args:
            trial: Optuna trial 객체

        Returns:
            CV 점수 (RMSE의 경우 낮을수록 좋음)
        """
        params = self._suggest_params(trial)

        # 모델 생성
        model = self._create_model(params)

        # 시계열 교차검증
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)

        cv_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(self._X)):
            X_train = self._X.iloc[train_idx]
            y_train = self._y.iloc[train_idx]
            X_val = self._X.iloc[val_idx]
            y_val = self._y.iloc[val_idx]

            # 학습
            model.fit(X_train, y_train)

            # 평가
            metrics = model.evaluate(X_val, y_val)
            cv_scores.append(metrics.rmse)

            # Pruning 체크 (중간 결과가 나쁘면 조기 종료)
            trial.report(np.mean(cv_scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_rmse = np.mean(cv_scores)

        # 추가 메트릭 기록
        trial.set_user_attr("cv_scores", cv_scores)
        trial.set_user_attr("cv_std", np.std(cv_scores))

        return mean_rmse

    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        show_progress: bool = True
    ) -> OptimizationResult:
        """
        하이퍼파라미터 최적화 실행.

        Args:
            X: 피처 DataFrame
            y: 타겟 Series
            show_progress: 진행 상황 표시

        Returns:
            OptimizationResult 객체
        """
        self._X = X
        self._y = y

        # Optuna Study 생성
        sampler = TPESampler(seed=self.seed)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=2)

        self._study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=sampler,
            pruner=pruner
        )

        # 최적화 실행
        verbosity = optuna.logging.INFO if show_progress else optuna.logging.WARNING
        optuna.logging.set_verbosity(verbosity)

        logger.info(f"Starting {self.model_name} optimization with {self.n_trials} trials...")

        self._study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=show_progress,
            catch=(Exception,)
        )

        # 결과 정리
        best_trial = self._study.best_trial

        result = OptimizationResult(
            model_name=self.model_name,
            best_params=best_trial.params,
            best_score=best_trial.value,
            n_trials=len(self._study.trials),
            cv_scores=best_trial.user_attrs.get("cv_scores"),
            optimization_history=[
                {
                    "trial": t.number,
                    "value": t.value,
                    "params": t.params
                }
                for t in self._study.trials
                if t.value is not None
            ]
        )

        logger.info(f"Optimization complete!")
        logger.info(f"  Best RMSE: {result.best_score:.4f}")
        logger.info(f"  Best params: {result.best_params}")

        return result

    def get_best_model(self) -> Optional[BaseModel]:
        """최적 파라미터로 모델 생성"""
        if self._study is None:
            return None

        return self._create_model(self._study.best_params)

    def get_importance(self) -> Dict[str, float]:
        """파라미터 중요도"""
        if self._study is None:
            return {}

        try:
            importance = optuna.importance.get_param_importances(self._study)
            return dict(importance)
        except Exception:
            return {}


class XGBoostOptimizer(OptunaOptimizer):
    """XGBoost 하이퍼파라미터 최적화"""

    @property
    def model_name(self) -> str:
        return "xgboost"

    def _suggest_params(self, trial: Trial) -> Dict[str, Any]:
        """XGBoost 파라미터 범위 정의"""
        return {
            # Tree 파라미터
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

            # Learning 파라미터
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),

            # 정규화 파라미터
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),

            # Huber loss 파라미터
            "huber_slope": trial.suggest_float("huber_slope", 5.0, 20.0),

            # 기타
            "random_state": self.seed
        }

    def _create_model(self, params: Dict[str, Any]) -> XGBoostModel:
        """XGBoost 모델 생성"""
        huber_delta = params.pop("huber_slope", 10.0)

        model_params = {
            **params,
            "objective": "reg:pseudohubererror",
            "huber_slope": huber_delta,
            "verbosity": 0
        }

        return XGBoostModel(params=model_params, use_huber=True, huber_delta=huber_delta)


class LightGBMOptimizer(OptunaOptimizer):
    """LightGBM 하이퍼파라미터 최적화"""

    @property
    def model_name(self) -> str:
        return "lightgbm"

    def _suggest_params(self, trial: Trial) -> Dict[str, Any]:
        """LightGBM 파라미터 범위 정의"""
        return {
            # Tree 파라미터
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),

            # Sampling 파라미터
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

            # Learning 파라미터
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),

            # 정규화 파라미터
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),

            # Huber loss 파라미터
            "huber_alpha": trial.suggest_float("huber_alpha", 5.0, 20.0),

            # 기타
            "random_state": self.seed,
            "verbose": -1
        }

    def _create_model(self, params: Dict[str, Any]) -> LightGBMModel:
        """LightGBM 모델 생성"""
        huber_alpha = params.pop("huber_alpha", 10.0)

        model_params = {
            **params,
            "objective": "huber",
            "alpha": huber_alpha
        }

        return LightGBMModel(params=model_params, use_huber=True, huber_delta=huber_alpha)


class RidgeOptimizer(OptunaOptimizer):
    """Ridge 하이퍼파라미터 최적화"""

    @property
    def model_name(self) -> str:
        return "ridge"

    def _suggest_params(self, trial: Trial) -> Dict[str, Any]:
        """Ridge 파라미터 범위 정의"""
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True)
        }

    def _create_model(self, params: Dict[str, Any]) -> RidgeModel:
        """Ridge 모델 생성"""
        return RidgeModel(
            alpha=params["alpha"],
            normalize_features=True
        )


class EnsembleOptimizer:
    """
    앙상블 가중치 최적화.

    개별 모델의 검증 성능을 기반으로 최적 앙상블 가중치를 찾습니다.
    """

    def __init__(
        self,
        models: List[BaseModel],
        n_trials: int = 100,
        seed: int = 42
    ):
        """
        Args:
            models: 앙상블할 모델 리스트
            n_trials: 최적화 시도 횟수
            seed: 랜덤 시드
        """
        self.models = models
        self.n_trials = n_trials
        self.seed = seed

        self._study: Optional[optuna.Study] = None
        self._predictions: Optional[np.ndarray] = None
        self._y_true: Optional[np.ndarray] = None

    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[List[float], float]:
        """
        앙상블 가중치 최적화.

        Args:
            X: 검증 피처
            y: 검증 타겟

        Returns:
            (최적 가중치 리스트, 최적 RMSE)
        """
        # 각 모델의 예측값 수집
        self._predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        self._y_true = y.values

        # Optuna Study 생성
        sampler = TPESampler(seed=self.seed)
        self._study = optuna.create_study(direction="minimize", sampler=sampler)

        logger.info(f"Optimizing ensemble weights for {len(self.models)} models...")

        self._study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        # 최적 가중치 정규화
        best_params = self._study.best_params
        raw_weights = [best_params[f"w{i}"] for i in range(len(self.models))]
        total = sum(raw_weights)
        best_weights = [w / total for w in raw_weights]

        logger.info(f"Best ensemble weights: {dict(zip([m.name for m in self.models], best_weights))}")
        logger.info(f"Best ensemble RMSE: {self._study.best_value:.4f}")

        return best_weights, self._study.best_value

    def _objective(self, trial: Trial) -> float:
        """가중치 최적화 목적 함수"""
        # 각 모델의 가중치
        weights = []
        for i in range(len(self.models)):
            w = trial.suggest_float(f"w{i}", 0.0, 1.0)
            weights.append(w)

        # 정규화
        total = sum(weights)
        if total == 0:
            return float("inf")

        weights = [w / total for w in weights]

        # 앙상블 예측
        ensemble_pred = np.zeros(len(self._y_true))
        for i, w in enumerate(weights):
            ensemble_pred += w * self._predictions[:, i]

        # RMSE 계산
        rmse = np.sqrt(np.mean((self._y_true - ensemble_pred) ** 2))

        return rmse


def optimize_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    cv_folds: int = 5,
    seed: int = 42
) -> Dict[str, OptimizationResult]:
    """
    모든 모델의 하이퍼파라미터 최적화.

    Args:
        X_train: 학습 피처
        y_train: 학습 타겟
        X_val: 검증 피처
        y_val: 검증 타겟
        n_trials: 최적화 시도 횟수
        cv_folds: CV fold 수
        seed: 랜덤 시드

    Returns:
        모델명 -> OptimizationResult 딕셔너리
    """
    results = {}

    # XGBoost 최적화
    logger.info("=" * 50)
    logger.info("Optimizing XGBoost...")
    xgb_optimizer = XGBoostOptimizer(n_trials=n_trials, cv_folds=cv_folds, seed=seed)
    results["xgboost"] = xgb_optimizer.optimize(X_train, y_train)

    # LightGBM 최적화
    logger.info("=" * 50)
    logger.info("Optimizing LightGBM...")
    lgb_optimizer = LightGBMOptimizer(n_trials=n_trials, cv_folds=cv_folds, seed=seed)
    results["lightgbm"] = lgb_optimizer.optimize(X_train, y_train)

    # Ridge 최적화
    logger.info("=" * 50)
    logger.info("Optimizing Ridge...")
    ridge_optimizer = RidgeOptimizer(n_trials=n_trials, cv_folds=cv_folds, seed=seed)
    results["ridge"] = ridge_optimizer.optimize(X_train, y_train)

    # 결과 요약
    logger.info("=" * 50)
    logger.info("Optimization Summary:")
    for name, result in results.items():
        logger.info(f"  {name}: RMSE={result.best_score:.4f}")

    return results
