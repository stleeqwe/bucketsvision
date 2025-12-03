"""
V5.4 피처 빌더.

리팩토링 Phase 2: Template Method 패턴 구현체.

V5.4 모델 피처 (5개):
- team_epm_diff: 팀 EPM 차이 (홈-원정)
- top5_epm_diff: 상위 5인 EPM 차이
- bench_strength_diff: 벤치 선수 EPM 차이
- ft_rate_diff: Free Throw Rate 차이
- sos_diff: Strength of Schedule 차이
"""

from typing import Dict, List

import pandas as pd

from app.services.feature_builders.base_builder import (
    BaseFeatureBuilder,
    FeatureBuildContext,
)


class V54FeatureBuilder(BaseFeatureBuilder):
    """
    V5.4 피처 빌더.

    Logistic Regression 모델용 5개 피처를 생성합니다.
    78.05% 전체 정확도, 87.88% 고신뢰도 정확도.
    """

    VERSION = "5.4"
    FEATURE_NAMES: List[str] = [
        'team_epm_diff',
        'top5_epm_diff',
        'bench_strength_diff',
        'ft_rate_diff',
        'sos_diff',
    ]

    # 피처 계수 (모델 해석용)
    FEATURE_COEFFICIENTS = {
        'team_epm_diff': 0.407,
        'top5_epm_diff': 0.361,
        'bench_strength_diff': 0.259,
        'ft_rate_diff': 0.066,
        'sos_diff': 0.016,
    }

    def build_team_features(self, context: FeatureBuildContext) -> Dict[str, float]:
        """
        팀 기반 피처 빌드.

        - team_epm_diff: 팀 전체 EPM 차이
        - ft_rate_diff: Free Throw Rate 차이

        Args:
            context: 피처 빌드 컨텍스트

        Returns:
            팀 피처 딕셔너리
        """
        # 팀 EPM 차이
        home_epm = context.get_team_epm_value(context.home_team_id, 'team_epm', 0.0)
        away_epm = context.get_team_epm_value(context.away_team_id, 'team_epm', 0.0)
        team_epm_diff = self.safe_diff(home_epm, away_epm)

        # FT Rate 차이 (팀 로그에서 계산)
        ft_rate_diff = self._calculate_ft_rate_diff(context)

        return {
            'team_epm_diff': team_epm_diff,
            'ft_rate_diff': ft_rate_diff,
        }

    def build_player_features(self, context: FeatureBuildContext) -> Dict[str, float]:
        """
        선수 기반 피처 빌드.

        - top5_epm_diff: 상위 5인 EPM 차이
        - bench_strength_diff: 벤치 선수 EPM 차이

        Args:
            context: 피처 빌드 컨텍스트

        Returns:
            선수 피처 딕셔너리
        """
        # 홈팀 선수 EPM
        home_players = self.get_team_players(
            context.player_epm, context.home_team_id
        )
        home_top5 = self.calculate_top_n_epm(home_players, n=5)
        home_bench = self.calculate_bench_strength(home_players, exclude_top=5)

        # 원정팀 선수 EPM
        away_players = self.get_team_players(
            context.player_epm, context.away_team_id
        )
        away_top5 = self.calculate_top_n_epm(away_players, n=5)
        away_bench = self.calculate_bench_strength(away_players, exclude_top=5)

        return {
            'top5_epm_diff': self.safe_diff(home_top5, away_top5),
            'bench_strength_diff': self.safe_diff(home_bench, away_bench),
        }

    def _calculate_ft_rate_diff(self, context: FeatureBuildContext) -> float:
        """
        Free Throw Rate 차이 계산.

        FT Rate = FTA / FGA (팀 로그 기반)

        Args:
            context: 피처 빌드 컨텍스트

        Returns:
            FT Rate 차이
        """
        if context.team_logs.empty:
            return 0.0

        home_ft_rate = self._get_team_ft_rate(
            context.team_logs, context.home_team_id, context.game_date
        )
        away_ft_rate = self._get_team_ft_rate(
            context.team_logs, context.away_team_id, context.game_date
        )

        return self.safe_diff(home_ft_rate, away_ft_rate)

    def _get_team_ft_rate(self, team_logs: pd.DataFrame,
                          team_id: int, before_date) -> float:
        """
        팀의 FT Rate 계산.

        Args:
            team_logs: 팀 로그 데이터
            team_id: 팀 ID
            before_date: 기준 날짜

        Returns:
            FT Rate (FTA/FGA)
        """
        if team_logs.empty:
            return 0.0

        # 팀 로그 필터링
        team_data = team_logs[team_logs['team_id'] == team_id].copy()
        if team_data.empty:
            return 0.0

        # 날짜 필터링
        team_data['game_date'] = pd.to_datetime(team_data['game_date'])
        team_data = team_data[team_data['game_date'] < pd.to_datetime(before_date)]

        if team_data.empty:
            return 0.0

        # 최근 10경기
        recent = team_data.sort_values('game_date', ascending=False).head(10)

        # FT Rate 계산
        fta_col = 'fta' if 'fta' in recent.columns else 'FTA'
        fga_col = 'fga' if 'fga' in recent.columns else 'FGA'

        if fta_col not in recent.columns or fga_col not in recent.columns:
            return 0.0

        total_fta = recent[fta_col].sum()
        total_fga = recent[fga_col].sum()

        if total_fga == 0:
            return 0.0

        return float(total_fta / total_fga)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        피처 중요도 반환.

        Returns:
            피처별 계수 딕셔너리
        """
        return self.FEATURE_COEFFICIENTS.copy()

    def describe_features(self, features: Dict[str, float]) -> str:
        """
        피처 설명 생성 (디버깅/로깅용).

        Args:
            features: 피처 딕셔너리

        Returns:
            설명 문자열
        """
        lines = [f"V{self.VERSION} Features:"]
        for name in self.FEATURE_NAMES:
            value = features.get(name, 0.0)
            coef = self.FEATURE_COEFFICIENTS.get(name, 0.0)
            contribution = value * coef
            lines.append(
                f"  {name}: {value:+.4f} (coef: {coef:.3f}, "
                f"contribution: {contribution:+.4f})"
            )
        return "\n".join(lines)
