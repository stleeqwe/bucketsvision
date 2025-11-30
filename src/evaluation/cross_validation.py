"""
Cross Validation Module.

시계열 교차검증 및 다양한 CV 전략을 구현합니다.
NBA 시즌 특성을 고려한 검증 전략을 포함합니다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.models.base import BaseModel, ModelMetrics
from src.evaluation.metrics import calculate_metrics, MetricsReport
from src.utils.logger import logger


@dataclass
class CVResult:
    """교차검증 결과"""
    model_name: str
    n_folds: int
    fold_metrics: List[MetricsReport]
    mean_rmse: float
    std_rmse: float
    mean_win_accuracy: float
    std_win_accuracy: float
    fold_sizes: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "n_folds": self.n_folds,
            "mean_rmse": self.mean_rmse,
            "std_rmse": self.std_rmse,
            "mean_win_accuracy": self.mean_win_accuracy,
            "std_win_accuracy": self.std_win_accuracy,
            "fold_sizes": self.fold_sizes
        }

    def summary(self) -> str:
        return (
            f"CV Results ({self.n_folds} folds):\n"
            f"  RMSE: {self.mean_rmse:.4f} (+/- {self.std_rmse:.4f})\n"
            f"  Win Accuracy: {self.mean_win_accuracy:.2%} (+/- {self.std_win_accuracy:.2%})"
        )


class BaseCrossValidator(ABC):
    """
    교차검증 기본 클래스.

    시계열 데이터의 특성을 고려한 검증 전략을 정의합니다.
    """

    @abstractmethod
    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        groups: pd.Series = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        학습/검증 인덱스 분할 생성.

        Args:
            X: 피처 DataFrame
            y: 타겟 Series
            groups: 그룹 정보 (시즌, 날짜 등)

        Yields:
            (train_indices, val_indices) 튜플
        """
        pass

    @abstractmethod
    def get_n_splits(self) -> int:
        """분할 수 반환"""
        pass


class TimeSeriesCV(BaseCrossValidator):
    """
    시계열 교차검증.

    sklearn의 TimeSeriesSplit을 래핑하여 NBA 데이터에 맞게 조정합니다.
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        max_train_size: Optional[int] = None,
        test_size: Optional[int] = None
    ):
        """
        Args:
            n_splits: 분할 수
            gap: 학습/검증 사이 갭 (데이터 누출 방지)
            max_train_size: 최대 학습 데이터 크기
            test_size: 검증 데이터 크기 (None이면 자동)
        """
        self.n_splits = n_splits
        self.gap = gap
        self.max_train_size = max_train_size
        self.test_size = test_size

        self._cv = TimeSeriesSplit(
            n_splits=n_splits,
            gap=gap,
            max_train_size=max_train_size,
            test_size=test_size
        )

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        groups: pd.Series = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """시계열 분할"""
        for train_idx, val_idx in self._cv.split(X):
            yield train_idx, val_idx

    def get_n_splits(self) -> int:
        return self.n_splits


class SeasonBasedCV(BaseCrossValidator):
    """
    시즌 기반 교차검증.

    NBA 시즌을 기준으로 분할하여 시즌 간 일반화 성능을 평가합니다.
    """

    def __init__(
        self,
        season_col: str = "season",
        train_seasons: Optional[List[int]] = None,
        val_seasons: Optional[List[int]] = None,
        expanding: bool = True
    ):
        """
        Args:
            season_col: 시즌 컬럼명
            train_seasons: 학습 시즌 리스트 (None이면 자동)
            val_seasons: 검증 시즌 리스트 (None이면 자동)
            expanding: True면 이전 모든 시즌으로 학습, False면 직전 시즌만
        """
        self.season_col = season_col
        self.train_seasons = train_seasons
        self.val_seasons = val_seasons
        self.expanding = expanding

        self._n_splits = 0

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        groups: pd.Series = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """시즌 기반 분할"""
        if self.season_col not in X.columns:
            raise ValueError(f"Column '{self.season_col}' not found in DataFrame")

        seasons = sorted(X[self.season_col].unique())

        if len(seasons) < 2:
            raise ValueError("Need at least 2 seasons for cross-validation")

        self._n_splits = 0

        for i in range(1, len(seasons)):
            val_season = seasons[i]

            if self.expanding:
                # 이전 모든 시즌으로 학습
                train_seasons = seasons[:i]
            else:
                # 직전 시즌만으로 학습
                train_seasons = [seasons[i - 1]]

            train_mask = X[self.season_col].isin(train_seasons)
            val_mask = X[self.season_col] == val_season

            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]

            if len(train_idx) > 0 and len(val_idx) > 0:
                self._n_splits += 1
                yield train_idx, val_idx

    def get_n_splits(self) -> int:
        return self._n_splits if self._n_splits > 0 else 1


class WalkForwardCV(BaseCrossValidator):
    """
    Walk-Forward 교차검증.

    시간 순서대로 학습 윈도우를 이동하며 검증합니다.
    실제 운영 시나리오를 시뮬레이션합니다.
    """

    def __init__(
        self,
        initial_train_size: int = 500,
        step_size: int = 100,
        test_size: int = 100,
        max_splits: Optional[int] = None,
        date_col: str = "game_date"
    ):
        """
        Args:
            initial_train_size: 초기 학습 데이터 크기
            step_size: 각 단계별 이동 크기
            test_size: 검증 데이터 크기
            max_splits: 최대 분할 수
            date_col: 날짜 컬럼명 (정렬용)
        """
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.test_size = test_size
        self.max_splits = max_splits
        self.date_col = date_col

        self._n_splits = 0

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        groups: pd.Series = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Walk-forward 분할"""
        n_samples = len(X)

        # 날짜 기준 정렬 인덱스
        if self.date_col in X.columns:
            sorted_indices = X[self.date_col].argsort().values
        else:
            sorted_indices = np.arange(n_samples)

        self._n_splits = 0
        train_end = self.initial_train_size

        while train_end + self.test_size <= n_samples:
            if self.max_splits and self._n_splits >= self.max_splits:
                break

            train_idx = sorted_indices[:train_end]
            val_idx = sorted_indices[train_end:train_end + self.test_size]

            self._n_splits += 1
            yield train_idx, val_idx

            train_end += self.step_size

    def get_n_splits(self) -> int:
        return self._n_splits if self._n_splits > 0 else 1


class MonthlyCV(BaseCrossValidator):
    """
    월별 교차검증.

    각 월을 검증 데이터로 사용하고 이전 모든 데이터로 학습합니다.
    NBA 시즌 내 성능 변화를 추적할 수 있습니다.
    """

    def __init__(
        self,
        date_col: str = "game_date",
        min_train_months: int = 2
    ):
        """
        Args:
            date_col: 날짜 컬럼명
            min_train_months: 최소 학습 기간 (월)
        """
        self.date_col = date_col
        self.min_train_months = min_train_months
        self._n_splits = 0

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        groups: pd.Series = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """월별 분할"""
        if self.date_col not in X.columns:
            raise ValueError(f"Column '{self.date_col}' not found in DataFrame")

        # 날짜를 datetime으로 변환
        dates = pd.to_datetime(X[self.date_col])
        X = X.copy()
        X["_year_month"] = dates.dt.to_period("M")

        year_months = sorted(X["_year_month"].unique())

        self._n_splits = 0

        for i in range(self.min_train_months, len(year_months)):
            val_month = year_months[i]
            train_months = year_months[:i]

            train_mask = X["_year_month"].isin(train_months)
            val_mask = X["_year_month"] == val_month

            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]

            if len(train_idx) > 0 and len(val_idx) > 0:
                self._n_splits += 1
                yield train_idx, val_idx

    def get_n_splits(self) -> int:
        return self._n_splits if self._n_splits > 0 else 1


def cross_validate_model(
    model: BaseModel,
    X: pd.DataFrame,
    y: pd.Series,
    cv: BaseCrossValidator,
    fit_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> CVResult:
    """
    모델 교차검증 실행.

    Args:
        model: 학습할 모델
        X: 피처 DataFrame
        y: 타겟 Series
        cv: 교차검증기
        fit_params: 모델 학습 파라미터
        verbose: 상세 출력 여부

    Returns:
        CVResult 객체
    """
    fit_params = fit_params or {}
    fold_metrics = []
    fold_sizes = []

    if verbose:
        logger.info(f"Starting cross-validation for {model.name}...")

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        # 모델 복사 및 학습
        # 대부분의 sklearn 호환 모델은 clone 사용
        import copy
        fold_model = copy.deepcopy(model)

        # eval_set 설정
        fold_fit_params = fit_params.copy()
        if "eval_set" not in fold_fit_params:
            fold_fit_params["eval_set"] = (X_val, y_val)

        fold_model.fit(X_train, y_train, **fold_fit_params)

        # 예측 및 평가
        y_pred = fold_model.predict(X_val)
        metrics = calculate_metrics(y_val.values, y_pred, model.name)

        fold_metrics.append(metrics)
        fold_sizes.append(len(val_idx))

        if verbose:
            logger.info(
                f"  Fold {fold_idx + 1}: "
                f"RMSE={metrics.rmse:.4f}, "
                f"Win Acc={metrics.win_accuracy:.2%}, "
                f"n={len(val_idx)}"
            )

    # 결과 집계
    rmses = [m.rmse for m in fold_metrics]
    win_accs = [m.win_accuracy for m in fold_metrics]

    result = CVResult(
        model_name=model.name,
        n_folds=len(fold_metrics),
        fold_metrics=fold_metrics,
        mean_rmse=np.mean(rmses),
        std_rmse=np.std(rmses),
        mean_win_accuracy=np.mean(win_accs),
        std_win_accuracy=np.std(win_accs),
        fold_sizes=fold_sizes
    )

    if verbose:
        logger.info(f"\n{result.summary()}")

    return result


def compare_cv_results(
    results: Dict[str, CVResult]
) -> pd.DataFrame:
    """
    여러 모델의 CV 결과 비교.

    Args:
        results: 모델명 -> CVResult 딕셔너리

    Returns:
        비교 DataFrame
    """
    comparison = []

    for model_name, cv_result in results.items():
        comparison.append({
            "model": model_name,
            "n_folds": cv_result.n_folds,
            "mean_rmse": cv_result.mean_rmse,
            "std_rmse": cv_result.std_rmse,
            "mean_win_accuracy": cv_result.mean_win_accuracy,
            "std_win_accuracy": cv_result.std_win_accuracy,
            "rmse_ci_lower": cv_result.mean_rmse - 1.96 * cv_result.std_rmse / np.sqrt(cv_result.n_folds),
            "rmse_ci_upper": cv_result.mean_rmse + 1.96 * cv_result.std_rmse / np.sqrt(cv_result.n_folds)
        })

    df = pd.DataFrame(comparison)
    df = df.sort_values("mean_rmse")

    return df
