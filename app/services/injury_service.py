"""
부상 정보 처리 서비스.

리팩토링 Phase 2.2: data_loader.py에서 부상 관련 로직 분리.

책임:
- ESPN/NBA PDF 부상 정보 조회
- 부상 영향력 계산기 관리
- 부상 요약 정보 생성
"""

import math
from datetime import date
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

from src.data_collection.espn_injury_client import ESPNInjuryClient, ESPNInjury
from src.data_collection.nba_injury_client import NBAInjuryClient, InjuryStatus
from src.features.injury_impact import InjuryImpactCalculator
from src.features.advanced_injury_impact import (
    AdvancedInjuryImpactCalculator,
    PlayerImpactResult,
    create_advanced_injury_calculator,
)
from src.utils.logger import logger

if TYPE_CHECKING:
    from app.services.data_loader import DataLoader


class InjuryService:
    """
    부상 정보 처리 서비스.

    ESPN과 NBA 공식 Injury Report에서 부상 정보를 조회하고,
    부상 영향력을 계산합니다.
    """

    # EPM 기반 영향력 계산 상수
    STARTER_MPG_THRESHOLD = 12.0

    def __init__(
        self,
        espn_client: Optional[ESPNInjuryClient] = None,
        nba_injury_client: Optional[NBAInjuryClient] = None,
    ):
        """
        Args:
            espn_client: ESPN 부상 정보 클라이언트
            nba_injury_client: NBA 공식 Injury Report 클라이언트
        """
        self.espn_client = espn_client or ESPNInjuryClient()
        self.nba_injury_client = nba_injury_client or NBAInjuryClient()

        # 캐시
        self._nba_injury_cache: Dict[str, Dict] = {}
        self._injury_calc: Optional[InjuryImpactCalculator] = None
        self._advanced_injury_calc: Optional[AdvancedInjuryImpactCalculator] = None

    def clear_cache(self) -> None:
        """
        부상 정보 캐시 초기화.

        ESPN과 NBA 부상 클라이언트의 캐시를 모두 초기화합니다.
        """
        self.espn_client.clear_cache()
        self.nba_injury_client.clear_cache()
        self._nba_injury_cache = {}
        self._injury_calc = None
        self._advanced_injury_calc = None
        logger.info("Injury caches cleared (ESPN + NBA PDF)")

    def get_injuries(self, team_abbr: str) -> List[ESPNInjury]:
        """
        팀 부상자 조회 (Out 상태).

        Args:
            team_abbr: 팀 약어 (예: "LAL")

        Returns:
            Out 상태 선수 리스트
        """
        return self.espn_client.get_out_players(team_abbr)

    def get_gtd_players(self, team_abbr: str) -> List[ESPNInjury]:
        """
        팀 GTD(Game Time Decision) 선수 조회.

        Args:
            team_abbr: 팀 약어

        Returns:
            GTD 상태 선수 리스트
        """
        return self.espn_client.get_gtd_players(team_abbr)

    def get_gtd_players_with_status(
        self,
        team_abbr: str,
        target_date: Optional[date] = None
    ) -> List[Dict]:
        """
        GTD 선수 조회 + NBA PDF로 세부 상태 확인.

        ESPN의 Day-To-Day를 NBA 공식 Injury Report로 세분화:
        - Probable: 출전 가능성 75%
        - Questionable: 출전 가능성 50%
        - Doubtful: 출전 가능성 25%

        Args:
            team_abbr: 팀 약어
            target_date: 기준 날짜

        Returns:
            [{"player_name", "espn_status", "nba_status", "play_probability", "miss_probability"}, ...]
        """
        if target_date is None:
            target_date = date.today()

        espn_gtd = self.espn_client.get_gtd_players(team_abbr)

        # NBA PDF 캐시 확인
        cache_key = target_date.isoformat()
        if cache_key not in self._nba_injury_cache:
            try:
                self._nba_injury_cache[cache_key] = self.nba_injury_client.fetch_injuries(target_date)
            except Exception as e:
                logger.warning(f"Failed to fetch NBA injury report: {e}")
                self._nba_injury_cache[cache_key] = {}

        nba_injuries = self._nba_injury_cache.get(cache_key, {})
        nba_team_injuries = nba_injuries.get(team_abbr, [])

        results = []
        for espn_inj in espn_gtd:
            player_name = espn_inj.player_name
            result = {
                "player_name": player_name,
                "espn_status": espn_inj.status,
                "nba_status": None,
                "detail": espn_inj.detail,
                # 기본값: ESPN GTD는 50% 출전 확률
                "play_probability": 0.5,
                "miss_probability": 0.5,
            }

            # NBA PDF에서 매칭 찾기
            espn_name_lower = player_name.lower().strip()
            for nba_inj in nba_team_injuries:
                nba_name_lower = nba_inj.player_name.lower().strip()
                # 정확한 매칭 또는 부분 매칭
                if espn_name_lower == nba_name_lower or \
                   espn_name_lower in nba_name_lower or \
                   nba_name_lower in espn_name_lower:
                    result["nba_status"] = nba_inj.status.value
                    result["play_probability"] = nba_inj.status.play_probability
                    result["miss_probability"] = 1.0 - nba_inj.status.play_probability
                    break

            results.append(result)

        return results

    def get_injury_calculator(
        self,
        player_epm: pd.DataFrame
    ) -> Optional[InjuryImpactCalculator]:
        """
        부상 영향 계산기 반환.

        Args:
            player_epm: 선수 EPM DataFrame

        Returns:
            InjuryImpactCalculator 또는 None
        """
        if self._injury_calc is not None:
            return self._injury_calc

        try:
            if player_epm.empty:
                logger.warning("Player EPM data is empty for injury calculator")
                return None
            self._injury_calc = InjuryImpactCalculator(player_epm)
            return self._injury_calc
        except Exception as e:
            logger.error(f"Error loading injury calculator: {e}")
            return None

    def get_player_impact(
        self,
        player_name: str,
        team_abbr: str,
        player_epm: pd.DataFrame
    ) -> Optional[Dict]:
        """
        개별 선수의 영향도 계산.

        Args:
            player_name: 선수 이름
            team_abbr: 팀 약어
            player_epm: 선수 EPM DataFrame

        Returns:
            선수 정보 딕셔너리 (없으면 None)
        """
        calc = self.get_injury_calculator(player_epm)
        if calc is None:
            return None

        player = calc.find_player(player_name, team_abbr)
        if player is None:
            return None

        mpg = player["mpg"]
        player_epm_val = player["tot"]

        # NaN 값 체크
        if math.isnan(mpg) or math.isnan(player_epm_val):
            return None

        # EPM 양수인 선수만 반영
        if player_epm_val <= 0:
            return None

        if mpg < self.STARTER_MPG_THRESHOLD:
            return None

        bench_avg = calc.bench_avg_epm.get(team_abbr, -2.0)
        impact = (player_epm_val - bench_avg) * (mpg / 48)

        if abs(impact) < 0.5:
            return None

        return {
            "name": player_name,
            "epm": round(player_epm_val, 1),
            "mpg": round(mpg, 0),
            "impact": round(impact, 1),
        }

    def calculate_injury_impact(
        self,
        team_abbr: str,
        injuries: List[ESPNInjury],
        player_epm: pd.DataFrame
    ) -> Tuple[float, List[Dict]]:
        """
        팀 부상 영향 계산.

        Args:
            team_abbr: 팀 약어
            injuries: 부상자 리스트
            player_epm: 선수 EPM DataFrame

        Returns:
            (총 영향도, 선수별 상세)
        """
        calc = self.get_injury_calculator(player_epm)
        if calc is None:
            return 0.0, []

        total_impact = 0.0
        details = []

        for injury in injuries:
            player = calc.find_player(injury.player_name, team_abbr)
            if player is None:
                continue

            mpg = player["mpg"]
            player_epm_val = player["tot"]

            # NaN 값 체크
            if math.isnan(mpg) or math.isnan(player_epm_val):
                continue

            # EPM 양수인 선수만 반영 (음수 선수는 빠져도 영향 없음)
            if player_epm_val <= 0:
                continue

            if mpg < self.STARTER_MPG_THRESHOLD:
                continue

            bench_avg = calc.bench_avg_epm.get(team_abbr, -2.0)
            impact = (player_epm_val - bench_avg) * (mpg / 48)

            if abs(impact) < 0.5:
                continue

            total_impact += impact
            details.append({
                "name": injury.player_name,
                "epm": round(player_epm_val, 1),
                "mpg": round(mpg, 0),
                "impact": round(impact, 1),
                "detail": injury.detail
            })

        return round(total_impact, 1), details

    def get_advanced_injury_calculator(
        self,
        player_epm_df: pd.DataFrame,
        team_logs: pd.DataFrame,
        player_logs: pd.DataFrame,
        team_epm: Dict[int, Dict]
    ) -> Optional[AdvancedInjuryImpactCalculator]:
        """
        고급 부상 영향력 계산기 반환.

        Args:
            player_epm_df: 선수 EPM DataFrame
            team_logs: 팀 경기 로그
            player_logs: 선수 경기 로그
            team_epm: 팀 EPM 딕셔너리

        Returns:
            AdvancedInjuryImpactCalculator 또는 None
        """
        if self._advanced_injury_calc is not None:
            return self._advanced_injury_calc

        try:
            if team_logs.empty or player_logs.empty:
                logger.warning("경기 로그 데이터 부족으로 고급 부상 계산기 생성 실패")
                return None

            if player_epm_df.empty:
                logger.warning("선수 EPM 데이터 부족으로 고급 부상 계산기 생성 실패")
                return None

            self._advanced_injury_calc = create_advanced_injury_calculator(
                player_epm_df=player_epm_df,
                team_game_logs=team_logs,
                player_game_logs=player_logs,
                team_epm=team_epm,
            )
            return self._advanced_injury_calc

        except Exception as e:
            logger.error(f"Error creating advanced injury calculator: {e}")
            return None

    def calculate_advanced_injury_impact(
        self,
        home_abbr: str,
        away_abbr: str,
        calc: AdvancedInjuryImpactCalculator,
    ) -> Tuple[float, float, List[PlayerImpactResult]]:
        """
        경기별 고급 부상 영향력 계산.

        Args:
            home_abbr: 홈팀 약어
            away_abbr: 원정팀 약어
            calc: 고급 부상 계산기

        Returns:
            (home_impact, away_impact, player_details)
            양수 = 해당 팀에 불리
        """
        # Out 선수 조회
        home_out = self.get_injuries(home_abbr)
        away_out = self.get_injuries(away_abbr)

        # GTD 선수 조회
        home_gtd = self.get_gtd_players(home_abbr)
        away_gtd = self.get_gtd_players(away_abbr)

        home_out_names = [inj.player_name for inj in home_out]
        away_out_names = [inj.player_name for inj in away_out]
        home_gtd_names = [inj.player_name for inj in home_gtd]
        away_gtd_names = [inj.player_name for inj in away_gtd]

        return calc.get_game_injury_impact(
            home_team=home_abbr,
            away_team=away_abbr,
            home_out_players=home_out_names,
            away_out_players=away_out_names,
            home_gtd_players=home_gtd_names,
            away_gtd_players=away_gtd_names,
        )

    def get_injury_summary(
        self,
        team_abbr: str,
        target_date: date,
        calc: Optional[AdvancedInjuryImpactCalculator]
    ) -> Dict:
        """
        팀 부상자 요약 정보 반환.

        Args:
            team_abbr: 팀 약어
            target_date: 기준 날짜
            calc: 고급 부상 계산기 (None일 수 있음)

        Returns:
            {
                "out_players": [...],
                "gtd_players": [...],
                "total_prob_shift": float  # % 단위
            }
        """
        out_injuries = self.get_injuries(team_abbr)
        gtd_with_status = self.get_gtd_players_with_status(team_abbr, target_date)

        out_details = []
        gtd_details = []
        total_impact = 0.0

        # Out 선수 처리
        for inj in out_injuries:
            if calc:
                result = calc.calculate_player_impact(inj.player_name, team_abbr)
                if result.is_valid:
                    total_impact += result.prob_shift
                    out_details.append({
                        "name": inj.player_name,
                        "status": "Out",
                        "detail": inj.detail,
                        "epm": round(result.epm, 2),
                        "mpg": round(result.mpg, 1),
                        "prob_shift": round(result.prob_shift * 100, 1),
                        "played_games": result.played_games,
                        "missed_games": result.missed_games,
                        "schedule_diff": round(result.schedule_diff, 3),
                    })
                else:
                    out_details.append({
                        "name": inj.player_name,
                        "status": "Out",
                        "detail": inj.detail,
                        "prob_shift": 0.0,
                        "skip_reason": result.skip_reason,
                    })
            else:
                out_details.append({
                    "name": inj.player_name,
                    "status": "Out",
                    "detail": inj.detail,
                    "prob_shift": 0.0,
                })

        # GTD 선수 처리
        for gtd_info in gtd_with_status:
            player_name = gtd_info["player_name"]
            miss_prob = gtd_info["miss_probability"]
            nba_status = gtd_info.get("nba_status") or "GTD"

            if calc:
                result = calc.calculate_player_impact(player_name, team_abbr)
                if result.is_valid:
                    gtd_impact = result.prob_shift * miss_prob
                    total_impact += gtd_impact
                    gtd_details.append({
                        "name": player_name,
                        "status": nba_status,
                        "detail": gtd_info.get("detail"),
                        "epm": round(result.epm, 2),
                        "mpg": round(result.mpg, 1),
                        "prob_shift": round(result.prob_shift * 100, 1),
                        "applied_shift": round(gtd_impact * 100, 1),
                        "miss_probability": round(miss_prob * 100, 0),
                        "played_games": result.played_games,
                        "missed_games": result.missed_games,
                        "schedule_diff": round(result.schedule_diff, 3),
                    })
                else:
                    gtd_details.append({
                        "name": player_name,
                        "status": nba_status,
                        "detail": gtd_info.get("detail"),
                        "prob_shift": 0.0,
                        "miss_probability": round(miss_prob * 100, 0),
                        "skip_reason": result.skip_reason,
                    })
            else:
                gtd_details.append({
                    "name": player_name,
                    "status": nba_status,
                    "detail": gtd_info.get("detail"),
                    "prob_shift": 0.0,
                    "miss_probability": round(miss_prob * 100, 0),
                })

        return {
            "out_players": out_details,
            "gtd_players": gtd_details,
            "total_prob_shift": round(total_impact * 100, 1),
        }
