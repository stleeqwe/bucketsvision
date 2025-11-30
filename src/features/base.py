"""
Base classes for feature engineering.

모든 피처 모듈의 기반이 되는 추상 클래스와 유틸리티를 제공합니다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import date

import pandas as pd
import numpy as np

from src.utils.logger import logger


@dataclass
class FeatureContext:
    """
    피처 계산을 위한 컨텍스트 데이터.

    경기 정보와 함께 필요한 모든 배경 데이터를 담고 있습니다.
    """
    # 경기 기본 정보
    game_id: str
    game_date: str
    home_team_id: int
    away_team_id: int
    season: int

    # D&T 데이터
    team_epm: Optional[pd.DataFrame] = None
    player_epm: Optional[pd.DataFrame] = None
    season_epm: Optional[pd.DataFrame] = None

    # NBA Stats 데이터
    games_history: Optional[pd.DataFrame] = None
    boxscores: Optional[pd.DataFrame] = None
    schedules: Optional[pd.DataFrame] = None

    # 추가 컨텍스트
    arena_coords: Optional[pd.DataFrame] = None

    def get_team_history(
        self,
        team_id: int,
        before_date: str,
        n_games: Optional[int] = None
    ) -> pd.DataFrame:
        """
        특정 팀의 경기 이력 조회.

        Args:
            team_id: 팀 ID
            before_date: 기준 날짜 (이전 경기만)
            n_games: 최근 N경기 (None이면 전체)

        Returns:
            경기 이력 DataFrame
        """
        if self.games_history is None or self.games_history.empty:
            return pd.DataFrame()

        df = self.games_history[
            (self.games_history["team_id"] == team_id) &
            (self.games_history["game_date"] < before_date)
        ].sort_values("game_date", ascending=False)

        if n_games:
            df = df.head(n_games)

        return df

    def get_team_epm_on_date(self, team_id: int, game_date: str) -> Optional[Dict]:
        """특정 날짜의 팀 EPM 데이터 조회"""
        if self.team_epm is None or self.team_epm.empty:
            return None

        row = self.team_epm[
            (self.team_epm["team_id"] == team_id) &
            (self.team_epm["game_dt"] == game_date)
        ]

        if row.empty:
            # 가장 가까운 이전 날짜 데이터 사용
            row = self.team_epm[
                (self.team_epm["team_id"] == team_id) &
                (self.team_epm["game_dt"] <= game_date)
            ].sort_values("game_dt", ascending=False).head(1)

        if row.empty:
            return None

        return row.iloc[0].to_dict()


@dataclass
class FeatureResult:
    """피처 계산 결과"""
    features: Dict[str, float]
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    missing_features: List[str] = field(default_factory=list)

    def get(self, name: str, default: float = 0.0) -> float:
        """피처 값 조회"""
        return self.features.get(name, default)

    def merge(self, other: "FeatureResult") -> "FeatureResult":
        """다른 결과와 병합"""
        merged_features = {**self.features, **other.features}
        merged_warnings = self.warnings + other.warnings
        merged_missing = self.missing_features + other.missing_features

        return FeatureResult(
            features=merged_features,
            is_valid=self.is_valid and other.is_valid,
            warnings=merged_warnings,
            missing_features=merged_missing
        )


class BaseFeature(ABC):
    """
    피처 기본 추상 클래스.

    모든 피처 모듈은 이 클래스를 상속받아 구현합니다.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """피처 모듈 이름"""
        pass

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """생성되는 피처 이름 목록"""
        pass

    @property
    def required_context(self) -> List[str]:
        """필요한 컨텍스트 데이터 목록"""
        return []

    @abstractmethod
    def compute(self, context: FeatureContext) -> FeatureResult:
        """
        피처 계산.

        Args:
            context: 피처 계산 컨텍스트

        Returns:
            FeatureResult 객체
        """
        pass

    def validate_context(self, context: FeatureContext) -> Tuple[bool, List[str]]:
        """
        컨텍스트 유효성 검증.

        Args:
            context: 피처 계산 컨텍스트

        Returns:
            (유효 여부, 에러 메시지 리스트)
        """
        errors = []

        for attr in self.required_context:
            value = getattr(context, attr, None)
            if value is None:
                errors.append(f"Missing required context: {attr}")
            elif isinstance(value, pd.DataFrame) and value.empty:
                errors.append(f"Empty DataFrame for: {attr}")

        return len(errors) == 0, errors

    def compute_safe(self, context: FeatureContext) -> FeatureResult:
        """
        안전한 피처 계산 (예외 처리 포함).

        Args:
            context: 피처 계산 컨텍스트

        Returns:
            FeatureResult 객체
        """
        try:
            # 컨텍스트 검증
            is_valid, errors = self.validate_context(context)
            if not is_valid:
                logger.warning(f"[{self.name}] Context validation failed: {errors}")
                return FeatureResult(
                    features={name: np.nan for name in self.feature_names},
                    is_valid=False,
                    warnings=errors,
                    missing_features=self.feature_names
                )

            # 피처 계산
            result = self.compute(context)

            # 누락 피처 확인
            missing = [n for n in self.feature_names if n not in result.features]
            if missing:
                result.missing_features.extend(missing)
                for name in missing:
                    result.features[name] = np.nan

            return result

        except Exception as e:
            logger.error(f"[{self.name}] Feature computation error: {e}")
            return FeatureResult(
                features={name: np.nan for name in self.feature_names},
                is_valid=False,
                warnings=[str(e)],
                missing_features=self.feature_names
            )


class CompositeFeature(BaseFeature):
    """
    복합 피처 클래스.

    여러 피처 모듈을 조합하여 하나의 피처 세트로 관리합니다.
    """

    def __init__(self, features: List[BaseFeature]):
        """
        Args:
            features: 포함할 피처 모듈 리스트
        """
        self._features = features
        self._feature_names = []
        for f in features:
            self._feature_names.extend(f.feature_names)

    @property
    def name(self) -> str:
        return "composite"

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    @property
    def required_context(self) -> List[str]:
        required = set()
        for f in self._features:
            required.update(f.required_context)
        return list(required)

    def compute(self, context: FeatureContext) -> FeatureResult:
        """모든 하위 피처 계산 및 병합"""
        result = FeatureResult(features={})

        for feature in self._features:
            sub_result = feature.compute_safe(context)
            result = result.merge(sub_result)

        return result


# ===================
# Utility Functions
# ===================

def safe_divide(
    numerator: Union[float, int],
    denominator: Union[float, int],
    default: float = 0.0
) -> float:
    """안전한 나눗셈"""
    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
        return default
    return float(numerator) / float(denominator)


def rolling_stat(
    values: pd.Series,
    window: int,
    stat: str = "mean",
    min_periods: int = 1
) -> pd.Series:
    """
    롤링 통계 계산.

    Args:
        values: 값 Series
        window: 윈도우 크기
        stat: 통계 종류 ("mean", "sum", "std")
        min_periods: 최소 필요 데이터 수

    Returns:
        롤링 통계 Series
    """
    roller = values.rolling(window=window, min_periods=min_periods)

    if stat == "mean":
        return roller.mean()
    elif stat == "sum":
        return roller.sum()
    elif stat == "std":
        return roller.std()
    else:
        raise ValueError(f"Unknown stat: {stat}")


def clip_feature(
    value: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> float:
    """피처 값 클리핑"""
    if pd.isna(value):
        return value

    if min_val is not None:
        value = max(min_val, value)
    if max_val is not None:
        value = min(max_val, value)

    return value


def compute_diff(home_value: float, away_value: float) -> float:
    """홈-어웨이 차분 계산"""
    if pd.isna(home_value) or pd.isna(away_value):
        return np.nan
    return home_value - away_value
