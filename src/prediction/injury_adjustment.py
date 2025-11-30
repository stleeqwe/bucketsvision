"""
예측 결과 사후 조정 모듈.

모델 예측 후 신규 부상자 정보를 반영하여 점수와 승률을 조정합니다.

사용 방식:
1. 모델이 기본 예측 생성 (predicted_margin)
2. ESPN에서 현재 부상 정보 조회
3. Out 상태 신규 부상자의 EPM 영향도 계산
4. 예측 점수 조정: adjusted_margin = predicted_margin - home_impact + away_impact
5. 조정된 점수로 승률 재계산
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

from src.data_collection.espn_injury_client import ESPNInjuryClient, ESPNInjury
from src.features.injury_impact import InjuryImpactCalculator, load_player_epm
from src.utils.logger import logger


@dataclass
class InjuryAdjustment:
    """부상 조정 결과"""
    player_name: str
    team_abbr: str
    player_epm: float
    bench_avg_epm: float
    minutes: float
    impact_points: float  # 양수 = 팀에 불리


@dataclass
class AdjustedPrediction:
    """조정된 예측 결과"""
    home_team: str
    away_team: str
    original_margin: float
    adjusted_margin: float
    home_win_prob: float
    adjustment_detail: Dict[str, List[InjuryAdjustment]]


class PredictionAdjuster:
    """
    예측 결과 사후 조정기.

    신규 부상자 정보를 반영하여 예측 결과를 조정합니다.
    """

    # 점수 -> 승률 변환을 위한 표준편차 (경험적 값)
    MARGIN_STD = 12.0

    def __init__(
        self,
        injury_calculator: InjuryImpactCalculator,
        espn_client: Optional[ESPNInjuryClient] = None
    ):
        """
        Args:
            injury_calculator: EPM 기반 영향도 계산기
            espn_client: ESPN 부상 정보 클라이언트
        """
        self.injury_calc = injury_calculator
        self.espn_client = espn_client or ESPNInjuryClient()

    def calculate_impact_points(
        self,
        player_epm: float,
        bench_avg_epm: float,
        minutes: float = 32.0
    ) -> float:
        """
        선수 결장 시 예상 점수 영향 계산.

        공식: (Player EPM - Bench Avg EPM) × (Minutes / 48)

        Args:
            player_epm: 선수 EPM
            bench_avg_epm: 팀 벤치 평균 EPM
            minutes: 예상 출전 시간 (기본: 32분)

        Returns:
            영향 점수 (양수 = 팀에 불리)
        """
        impact = (player_epm - bench_avg_epm) * (minutes / 48)
        return round(impact, 2)

    def margin_to_win_probability(self, margin: float) -> float:
        """
        예측 점수차를 승률로 변환.

        정규분포 CDF 사용: P(홈팀 승리) = P(actual_margin > 0)

        Args:
            margin: 예측 홈팀 점수차

        Returns:
            홈팀 승리 확률 (0-1)
        """
        from scipy.stats import norm
        prob = norm.cdf(margin / self.MARGIN_STD)
        return round(prob, 4)

    def get_team_out_players(
        self,
        team_abbr: str,
        exclude_long_term: bool = True
    ) -> List[ESPNInjury]:
        """
        팀의 Out 상태 선수 조회.

        Args:
            team_abbr: 팀 약어
            exclude_long_term: 장기 부상자 제외 여부

        Returns:
            Out 상태 선수 리스트
        """
        out_players = self.espn_client.get_out_players(team_abbr)

        if exclude_long_term:
            # 장기 부상 키워드 필터링
            long_term_keywords = [
                "out for season", "out indefinitely", "surgery",
                "acl", "achilles", "fracture"
            ]

            filtered = []
            for player in out_players:
                detail_lower = (player.detail or "").lower()
                is_long_term = any(kw in detail_lower for kw in long_term_keywords)

                if not is_long_term:
                    filtered.append(player)

            return filtered

        return out_players

    def calculate_game_adjustment(
        self,
        home_team: str,
        away_team: str
    ) -> Tuple[float, float, Dict[str, List[InjuryAdjustment]]]:
        """
        경기별 부상 조정값 계산.

        Args:
            home_team: 홈팀 약어
            away_team: 원정팀 약어

        Returns:
            (home_adjustment, away_adjustment, detail)
            adjustment = 팀에 불리한 점수 (양수)
        """
        detail = {"home": [], "away": []}

        # 홈팀 Out 선수
        home_out = self.get_team_out_players(home_team)
        home_adjustment = 0.0

        for player in home_out:
            impact_info = self._calculate_player_adjustment(player, home_team)
            if impact_info:
                home_adjustment += impact_info.impact_points
                detail["home"].append(impact_info)

        # 원정팀 Out 선수
        away_out = self.get_team_out_players(away_team)
        away_adjustment = 0.0

        for player in away_out:
            impact_info = self._calculate_player_adjustment(player, away_team)
            if impact_info:
                away_adjustment += impact_info.impact_points
                detail["away"].append(impact_info)

        return home_adjustment, away_adjustment, detail

    def _calculate_player_adjustment(
        self,
        injury: ESPNInjury,
        team_abbr: str
    ) -> Optional[InjuryAdjustment]:
        """
        개별 선수 부상 영향 계산.

        Args:
            injury: ESPN 부상 정보
            team_abbr: 팀 약어

        Returns:
            InjuryAdjustment 또는 None (영향 없는 경우)
        """
        # 선수 EPM 데이터 조회
        player = self.injury_calc.find_player(injury.player_name, team_abbr)

        if player is None:
            logger.debug(f"Player not found in EPM data: {injury.player_name}")
            return None

        player_epm = player['tot']
        mpg = player['mpg']

        # 벤치 선수는 영향 미미
        if mpg < self.injury_calc.STARTER_MPG_THRESHOLD:
            return None

        bench_avg = self.injury_calc.bench_avg_epm.get(team_abbr, -2.0)
        impact = self.calculate_impact_points(player_epm, bench_avg, mpg)

        # 영향이 거의 없는 경우 제외
        if abs(impact) < 0.5:
            return None

        return InjuryAdjustment(
            player_name=injury.player_name,
            team_abbr=team_abbr,
            player_epm=round(player_epm, 2),
            bench_avg_epm=round(bench_avg, 2),
            minutes=round(mpg, 1),
            impact_points=impact
        )

    def adjust_prediction(
        self,
        home_team: str,
        away_team: str,
        predicted_margin: float
    ) -> AdjustedPrediction:
        """
        예측 결과 조정.

        Args:
            home_team: 홈팀 약어
            away_team: 원정팀 약어
            predicted_margin: 모델 예측 홈팀 점수차

        Returns:
            조정된 예측 결과
        """
        home_adj, away_adj, detail = self.calculate_game_adjustment(
            home_team, away_team
        )

        # 조정된 마진 계산
        # home_adj가 양수면 홈팀에 불리 -> 마진 감소
        # away_adj가 양수면 원정팀에 불리 -> 마진 증가
        adjusted_margin = predicted_margin - home_adj + away_adj

        # 승률 계산
        win_prob = self.margin_to_win_probability(adjusted_margin)

        logger.info(
            f"{home_team} vs {away_team}: "
            f"Original={predicted_margin:.1f}, "
            f"HomeAdj={home_adj:.1f}, AwayAdj={away_adj:.1f}, "
            f"Adjusted={adjusted_margin:.1f}, "
            f"WinProb={win_prob:.1%}"
        )

        return AdjustedPrediction(
            home_team=home_team,
            away_team=away_team,
            original_margin=predicted_margin,
            adjusted_margin=round(adjusted_margin, 2),
            home_win_prob=win_prob,
            adjustment_detail=detail
        )

    def refresh_injury_data(self) -> None:
        """부상 데이터 새로고침"""
        self.espn_client.clear_cache()
        self.espn_client.fetch_all_injuries(force_refresh=True)


def create_prediction_adjuster(
    data_dir: Path,
    season: int = 2026
) -> PredictionAdjuster:
    """
    PredictionAdjuster 팩토리 함수.

    Args:
        data_dir: 데이터 디렉토리
        season: 시즌 (Player EPM 데이터용)

    Returns:
        PredictionAdjuster 인스턴스
    """
    player_epm = load_player_epm(data_dir, season)
    injury_calc = InjuryImpactCalculator(player_epm)

    return PredictionAdjuster(injury_calc)


# 사용 예시
if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path("data")

    # 조정기 생성
    adjuster = create_prediction_adjuster(data_dir, season=2026)

    # 예측 조정 예시
    result = adjuster.adjust_prediction(
        home_team="LAL",
        away_team="BOS",
        predicted_margin=2.5  # 모델 예측: 홈팀 +2.5
    )

    print(f"\n=== 예측 조정 결과 ===")
    print(f"경기: {result.home_team} vs {result.away_team}")
    print(f"원래 예측: {result.original_margin:+.1f}")
    print(f"조정 후: {result.adjusted_margin:+.1f}")
    print(f"홈팀 승률: {result.home_win_prob:.1%}")

    if result.adjustment_detail["home"]:
        print(f"\n홈팀 결장:")
        for adj in result.adjustment_detail["home"]:
            print(f"  - {adj.player_name}: EPM={adj.player_epm:.1f}, Impact={adj.impact_points:+.1f}")

    if result.adjustment_detail["away"]:
        print(f"\n원정팀 결장:")
        for adj in result.adjustment_detail["away"]:
            print(f"  - {adj.player_name}: EPM={adj.player_epm:.1f}, Impact={adj.impact_points:+.1f}")
