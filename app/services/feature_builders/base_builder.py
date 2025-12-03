"""
피처 빌더 베이스 클래스.

리팩토링 Phase 2: Template Method 패턴 적용.

기존 위치:
- app/services/data_loader.py: build_features_v4(), build_features_v4_3(),
                               build_features_v5_2(), build_features_v5_4()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class FeatureBuildContext:
    """피처 빌드에 필요한 컨텍스트 데이터"""
    game_date: date
    home_team_id: int
    away_team_id: int
    team_logs: pd.DataFrame
    team_epm: Dict[int, Dict]  # {team_id: {team_epm, sos, ...}}
    player_epm: pd.DataFrame
    is_home_b2b: bool = False
    is_away_b2b: bool = False

    def get_team_epm_value(self, team_id: int, key: str = 'team_epm',
                           default: float = 0.0) -> float:
        """팀 EPM 값 안전하게 조회"""
        team_data = self.team_epm.get(team_id, {})
        val = team_data.get(key)
        return val if val is not None else default


@dataclass
class FeatureBuildResult:
    """피처 빌드 결과"""
    features: Dict[str, float]
    feature_names: List[str]
    is_valid: bool = True
    error_message: Optional[str] = None

    @classmethod
    def error(cls, message: str) -> 'FeatureBuildResult':
        """에러 결과 생성"""
        return cls(
            features={},
            feature_names=[],
            is_valid=False,
            error_message=message
        )


class BaseFeatureBuilder(ABC):
    """
    피처 빌더 베이스 클래스.

    Template Method 패턴을 사용하여 피처 빌드 프로세스를 정의합니다.
    서브클래스는 각 단계를 구현하여 버전별 피처를 생성합니다.

    Template Method: build()
    - validate_context() → 컨텍스트 검증
    - build_team_features() → 팀 기반 피처
    - build_player_features() → 선수 기반 피처
    - build_schedule_features() → 스케줄 기반 피처
    - combine_features() → 피처 조합
    """

    # 서브클래스에서 정의
    VERSION: str = "base"
    FEATURE_NAMES: List[str] = []

    def build(self, context: FeatureBuildContext) -> FeatureBuildResult:
        """
        Template Method: 피처 빌드 프로세스.

        Args:
            context: 피처 빌드 컨텍스트

        Returns:
            FeatureBuildResult
        """
        # 1. 컨텍스트 검증
        validation_error = self.validate_context(context)
        if validation_error:
            return FeatureBuildResult.error(validation_error)

        try:
            # 2. 각 카테고리별 피처 빌드
            team_features = self.build_team_features(context)
            player_features = self.build_player_features(context)
            schedule_features = self.build_schedule_features(context)

            # 3. 피처 조합
            all_features = self.combine_features(
                team_features, player_features, schedule_features
            )

            # 4. 피처 순서 정렬 및 검증
            ordered_features = self._order_features(all_features)

            return FeatureBuildResult(
                features=ordered_features,
                feature_names=self.FEATURE_NAMES,
                is_valid=True
            )

        except Exception as e:
            return FeatureBuildResult.error(f"Feature build failed: {str(e)}")

    def validate_context(self, context: FeatureBuildContext) -> Optional[str]:
        """
        컨텍스트 검증 (Hook).

        Args:
            context: 피처 빌드 컨텍스트

        Returns:
            에러 메시지 (None이면 유효)
        """
        if context.home_team_id == context.away_team_id:
            return "Home and away team IDs are the same"

        # team_epm이 없어도 0으로 처리하므로 검증 완화
        return None

    @abstractmethod
    def build_team_features(self, context: FeatureBuildContext) -> Dict[str, float]:
        """
        팀 기반 피처 빌드.

        Args:
            context: 피처 빌드 컨텍스트

        Returns:
            팀 피처 딕셔너리
        """
        pass

    @abstractmethod
    def build_player_features(self, context: FeatureBuildContext) -> Dict[str, float]:
        """
        선수 기반 피처 빌드.

        Args:
            context: 피처 빌드 컨텍스트

        Returns:
            선수 피처 딕셔너리
        """
        pass

    def build_schedule_features(self, context: FeatureBuildContext) -> Dict[str, float]:
        """
        스케줄 기반 피처 빌드 (선택적 Hook).

        기본 구현: SOS diff만 계산.

        Args:
            context: 피처 빌드 컨텍스트

        Returns:
            스케줄 피처 딕셔너리
        """
        home_sos = context.get_team_epm_value(context.home_team_id, 'sos', 0.0)
        away_sos = context.get_team_epm_value(context.away_team_id, 'sos', 0.0)

        return {
            'sos_diff': home_sos - away_sos
        }

    def combine_features(self, team_features: Dict[str, float],
                        player_features: Dict[str, float],
                        schedule_features: Dict[str, float]) -> Dict[str, float]:
        """
        피처 조합 (선택적 Hook).

        기본 구현: 모든 피처를 단순 병합.

        Args:
            team_features: 팀 피처
            player_features: 선수 피처
            schedule_features: 스케줄 피처

        Returns:
            조합된 피처 딕셔너리
        """
        combined = {}
        combined.update(team_features)
        combined.update(player_features)
        combined.update(schedule_features)
        return combined

    def _order_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        피처를 FEATURE_NAMES 순서로 정렬.

        Args:
            features: 정렬되지 않은 피처

        Returns:
            정렬된 피처 딕셔너리
        """
        ordered = {}
        for name in self.FEATURE_NAMES:
            if name in features:
                ordered[name] = features[name]
            else:
                # 누락된 피처는 0으로 채움
                ordered[name] = 0.0
        return ordered

    # 유틸리티 메서드

    @staticmethod
    def safe_diff(home_val, away_val, default: float = 0.0) -> float:
        """안전한 차이 계산"""
        h = home_val if home_val is not None else default
        a = away_val if away_val is not None else default
        return float(h - a)

    @staticmethod
    def get_team_players(player_epm: pd.DataFrame, team_id: int) -> pd.DataFrame:
        """팀 선수 EPM 필터링"""
        if player_epm.empty:
            return pd.DataFrame()

        team_col = 'team_id' if 'team_id' in player_epm.columns else 'TEAM_ID'
        return player_epm[player_epm[team_col] == team_id].copy()

    @staticmethod
    def calculate_top_n_epm(team_players: pd.DataFrame, n: int = 5) -> float:
        """상위 N명 EPM 평균 (MPG 기준 상위 선수)"""
        if team_players.empty:
            return 0.0

        # MPG 기준 상위 N명 선택
        mpg_col = 'mpg' if 'mpg' in team_players.columns else 'MPG'
        if mpg_col not in team_players.columns:
            return 0.0

        # EPM 컬럼 (DNT API는 'tot' 사용)
        epm_col = 'tot' if 'tot' in team_players.columns else ('epm' if 'epm' in team_players.columns else 'EPM')
        if epm_col not in team_players.columns:
            return 0.0

        top_n = team_players.nlargest(n, mpg_col)
        return float(top_n[epm_col].mean()) if len(top_n) > 0 else 0.0

    @staticmethod
    def calculate_bench_strength(team_players: pd.DataFrame,
                                 exclude_top: int = 5) -> float:
        """벤치 선수 EPM 평균 (MPG 상위 N명 제외, 6-10위)"""
        if team_players.empty or len(team_players) < exclude_top + 1:
            return -2.0  # 기본값

        # MPG 기준 정렬
        mpg_col = 'mpg' if 'mpg' in team_players.columns else 'MPG'
        if mpg_col not in team_players.columns:
            return -2.0

        # EPM 컬럼 (DNT API는 'tot' 사용)
        epm_col = 'tot' if 'tot' in team_players.columns else ('epm' if 'epm' in team_players.columns else 'EPM')
        if epm_col not in team_players.columns:
            return -2.0

        sorted_players = team_players.nlargest(10, mpg_col)
        bench = sorted_players.iloc[exclude_top:] if len(sorted_players) > exclude_top else pd.DataFrame()
        return float(bench[epm_col].mean()) if len(bench) > 0 else -2.0
