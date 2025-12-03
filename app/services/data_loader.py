"""
데이터 로더 모듈.

오늘 경기 스케줄과 팀 EPM 데이터를 로드합니다.

리팩토링 Phase 2: 모듈 분리 및 서비스 통합.
- StatCalculator: 통계 계산 담당
- InjuryService: 부상 정보 처리 담당
- GameScheduleService: 경기 스케줄 조회 담당
"""

import math
import sys
import warnings
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

# 캐시 TTL 설정
TEAM_EPM_CACHE_TTL_MINUTES = 60  # Team EPM: 1시간

import numpy as np
import pandas as pd

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.dnt_api import DNTApiClient
from src.data_collection.nba_stats_client import NBAStatsClient
from src.data_collection.espn_injury_client import ESPNInjuryClient, ESPNInjury
from src.data_collection.nba_injury_client import NBAInjuryClient, InjuryStatus
from src.data_collection.odds_api_client import OddsAPIClient, GameOdds
from src.features.injury_impact import InjuryImpactCalculator
from src.features.advanced_injury_impact import (
    AdvancedInjuryImpactCalculator,
    PlayerImpactResult,
    create_advanced_injury_calculator
)
from src.utils.logger import logger
from src.utils.memory import optimize_dataframe
from config.constants import TEAM_INFO, ABBR_TO_ID

# 리팩토링 Phase 2: Feature Builder
from app.services.feature_builders import V54FeatureBuilder
from app.services.feature_builders.base_builder import FeatureBuildContext

# 리팩토링 Phase 2: 새로운 서비스들
from app.services.calculators import StatCalculator, PlayerStatCalculator
from app.services.injury_service import InjuryService
from app.services.game_schedule_service import GameScheduleService


class DataLoader:
    """
    데이터 로더.

    Phase 2 리팩토링으로 다음 서비스들에 위임:
    - GameScheduleService: 경기 스케줄 조회
    - InjuryService: 부상 정보 처리
    - StatCalculator: 통계 계산
    """

    def __init__(self, data_dir: Path):
        """
        Args:
            data_dir: 데이터 디렉토리
        """
        self.data_dir = data_dir

        # API 클라이언트
        self.nba_client = NBAStatsClient()
        self.espn_client = ESPNInjuryClient()
        self.nba_injury_client = NBAInjuryClient()
        self.dnt_client = DNTApiClient()
        self.odds_client = OddsAPIClient()

        # Phase 2: 새로운 서비스들 (위임 패턴)
        self._game_schedule_service = GameScheduleService(self.nba_client)
        self._injury_service = InjuryService(self.espn_client, self.nba_injury_client)

        # 캐시
        self._team_epm_date_cache: Dict[str, Dict[int, Dict]] = {}
        self._team_epm_timestamp: Optional[datetime] = None
        self._nba_injury_cache: Dict[str, Dict] = {}
        self._injury_calc: Optional[InjuryImpactCalculator] = None
        self._advanced_injury_calc: Optional[AdvancedInjuryImpactCalculator] = None
        self._team_game_logs_cache: Optional[pd.DataFrame] = None
        self._player_game_logs_cache: Optional[pd.DataFrame] = None
        self._team_stats_cache: Dict[int, Dict] = {}
        self._player_epm_cache: Dict[int, pd.DataFrame] = {}
        self._odds_cache: Optional[Dict[Tuple[str, str], GameOdds]] = None

    def clear_injury_caches(self) -> None:
        """
        부상 정보 캐시 초기화.

        ESPN과 NBA 부상 클라이언트의 캐시를 모두 초기화합니다.
        프론트엔드 새로고침 버튼에서 호출됩니다.
        """
        # Phase 2: InjuryService에 위임
        self._injury_service.clear_cache()
        self._nba_injury_cache = {}
        self._injury_calc = None
        self._advanced_injury_calc = None

    def clear_all_api_caches(self) -> None:
        """
        모든 API 캐시 초기화.

        Team EPM, 부상 정보, 배당 등 모든 API 캐시를 초기화합니다.
        """
        # Team EPM 캐시
        self._team_epm_date_cache = {}
        self._team_epm_timestamp = None

        # 부상 캐시
        self.clear_injury_caches()

        # 배당 캐시
        self._odds_cache = None

        logger.info("All API caches cleared (Team EPM + Injury + Odds)")

    def get_team_info(self, team_id: int) -> Dict:
        """팀 정보 조회"""
        return TEAM_INFO.get(team_id, {"abbr": "UNK", "name": "Unknown", "color": "#666666"})

    def get_team_id(self, abbr: str) -> int:
        """팀 약어로 ID 조회"""
        return ABBR_TO_ID.get(abbr, 0)

    def get_game_odds(self, home_abbr: str, away_abbr: str) -> Optional[Dict]:
        """
        경기별 배당 정보 조회.

        Args:
            home_abbr: 홈팀 약어 (예: "HOU")
            away_abbr: 원정팀 약어 (예: "UTA")

        Returns:
            배당 정보 딕셔너리 또는 None
        """
        # 캐시가 없으면 로드
        if self._odds_cache is None:
            try:
                self._odds_cache = self.odds_client.get_all_games_odds()
                logger.info(f"Loaded odds for {len(self._odds_cache)} games")
            except Exception as e:
                logger.warning(f"Failed to load odds: {e}")
                self._odds_cache = {}

        # 캐시에서 조회
        odds = self._odds_cache.get((home_abbr, away_abbr))
        if odds:
            return {
                "spread_home": odds.spread_home,
                "spread_away": odds.spread_away,
                "spread_home_odds": odds.spread_home_odds,
                "spread_away_odds": odds.spread_away_odds,
                "moneyline_home": odds.moneyline_home,
                "moneyline_away": odds.moneyline_away,
                "total_line": odds.total_line,
                "bookmaker": odds.bookmaker,
            }
        return None

    def clear_odds_cache(self) -> None:
        """배당 캐시 초기화 (새로고침 시)"""
        self._odds_cache = None

    def _is_team_epm_cache_valid(self) -> bool:
        """Team EPM 캐시 유효성 검사 (TTL 기반)"""
        if not self._team_epm_date_cache or self._team_epm_timestamp is None:
            return False

        elapsed = datetime.now() - self._team_epm_timestamp
        return elapsed < timedelta(minutes=TEAM_EPM_CACHE_TTL_MINUTES)

    def load_team_epm(self, target_date: Optional[date] = None) -> Dict[int, Dict]:
        """
        팀 EPM 데이터 로드 (DNT API에서).

        Args:
            target_date: 기준 날짜 (없으면 최신)

        Returns:
            team_id -> EPM 데이터 딕셔너리
        """
        # 날짜별 캐시 키 생성
        cache_key = target_date.strftime("%Y-%m-%d") if target_date else "latest"

        # TTL 기반 캐시 검사
        if cache_key in self._team_epm_date_cache and self._is_team_epm_cache_valid():
            return self._team_epm_date_cache[cache_key]

        # TTL 만료 시 캐시 클리어
        if not self._is_team_epm_cache_valid():
            self._team_epm_date_cache = {}

        try:
            # DNT API에서 팀 EPM 로드
            date_str = target_date.strftime("%Y-%m-%d") if target_date else None
            team_epm_list = self.dnt_client.get_team_epm(date=date_str)

            epm_data = {}
            for team_data in team_epm_list:
                team_id = int(team_data.get("team_id", 0))
                if team_id == 0:
                    continue

                epm_data[team_id] = {
                    "team_epm": team_data.get("team_epm", 0),
                    "team_oepm": team_data.get("team_oepm", 0),
                    "team_depm": team_data.get("team_depm", 0),
                    "team_epm_game_optimized": team_data.get("team_epm_game_optimized", 0),
                    "team_oepm_game_optimized": team_data.get("team_oepm_game_optimized", 0),
                    "team_depm_game_optimized": team_data.get("team_depm_game_optimized", 0),
                    "sos": team_data.get("sos", 0),
                    "sos_o": team_data.get("sos_o", 0),
                    "sos_d": team_data.get("sos_d", 0),
                    "team_epm_rk": team_data.get("team_epm_rk", 15),
                    "team_oepm_rk": team_data.get("team_oepm_rk", 15),
                    "team_depm_rk": team_data.get("team_depm_rk", 15),
                    "team_epm_z": team_data.get("team_epm_z", 0),
                    "team_oepm_z": team_data.get("team_oepm_z", 0),
                    "team_depm_z": team_data.get("team_depm_z", 0),
                    "team_alias": team_data.get("team_alias", ""),  # 팀 약어 추가
                }

            # 날짜별 캐시에 저장 (TTL 타임스탬프 업데이트)
            self._team_epm_date_cache[cache_key] = epm_data
            self._team_epm_timestamp = datetime.now()
            logger.info(f"Loaded EPM for {len(epm_data)} teams from DNT API (date={cache_key}, TTL={TEAM_EPM_CACHE_TTL_MINUTES}min)")
            return epm_data

        except Exception as e:
            logger.error(f"Error loading team EPM from API: {e}")
            return {}

    def build_features(
        self,
        home_team_id: int,
        away_team_id: int,
        team_epm: Dict[int, Dict]
    ) -> Dict[str, float]:
        """
        경기 피처 생성.

        Args:
            home_team_id: 홈팀 ID
            away_team_id: 원정팀 ID
            team_epm: 팀 EPM 데이터

        Returns:
            피처 딕셔너리
        """
        home = team_epm.get(home_team_id, {})
        away = team_epm.get(away_team_id, {})

        def safe_diff(h_val, a_val, default=0):
            """None 값을 안전하게 처리"""
            h = h_val if h_val is not None else default
            a = a_val if a_val is not None else default
            return h - a

        return {
            "team_epm_diff": safe_diff(home.get("team_epm"), away.get("team_epm"), 0),
            "team_oepm_diff": safe_diff(home.get("team_oepm"), away.get("team_oepm"), 0),
            "team_depm_diff": safe_diff(home.get("team_depm"), away.get("team_depm"), 0),
            "team_epm_go_diff": safe_diff(home.get("team_epm_game_optimized"), away.get("team_epm_game_optimized"), 0),
            "team_oepm_go_diff": safe_diff(home.get("team_oepm_game_optimized"), away.get("team_oepm_game_optimized"), 0),
            "team_depm_go_diff": safe_diff(home.get("team_depm_game_optimized"), away.get("team_depm_game_optimized"), 0),
            "sos_diff": safe_diff(home.get("sos"), away.get("sos"), 0),
            "sos_o_diff": safe_diff(home.get("sos_o"), away.get("sos_o"), 0),
            "sos_d_diff": safe_diff(home.get("sos_d"), away.get("sos_d"), 0),
            "team_epm_rk_diff": safe_diff(home.get("team_epm_rk"), away.get("team_epm_rk"), 15),
            "team_oepm_rk_diff": safe_diff(home.get("team_oepm_rk"), away.get("team_oepm_rk"), 15),
            "team_depm_rk_diff": safe_diff(home.get("team_depm_rk"), away.get("team_depm_rk"), 15),
            "team_epm_z_diff": safe_diff(home.get("team_epm_z"), away.get("team_epm_z"), 0),
            "team_oepm_z_diff": safe_diff(home.get("team_oepm_z"), away.get("team_oepm_z"), 0),
            "team_depm_z_diff": safe_diff(home.get("team_depm_z"), away.get("team_depm_z"), 0),
        }

    def get_injuries(self, team_abbr: str) -> List[ESPNInjury]:
        """팀 부상자 조회 (Out 상태)"""
        return self.espn_client.get_out_players(team_abbr)

    def get_gtd_players(self, team_abbr: str) -> List[ESPNInjury]:
        """팀 GTD 선수 조회"""
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

    def get_injury_calculator(self) -> Optional[InjuryImpactCalculator]:
        """부상 영향 계산기 반환 (DNT API 실시간 데이터 사용)"""
        if self._injury_calc is None:
            try:
                # DNT API에서 실시간 선수 EPM 로드
                player_epm = self.load_player_epm(season=2026)
                if player_epm.empty:
                    logger.warning("Player EPM data is empty for injury calculator")
                    return None
                self._injury_calc = InjuryImpactCalculator(player_epm)
            except Exception as e:
                logger.error(f"Error loading injury calculator: {e}")

        return self._injury_calc

    def get_player_impact(
        self,
        player_name: str,
        team_abbr: str
    ) -> Optional[Dict]:
        """
        개별 선수의 영향도 계산.

        Args:
            player_name: 선수 이름
            team_abbr: 팀 약어

        Returns:
            선수 정보 딕셔너리 (없으면 None)
        """
        calc = self.get_injury_calculator()
        if calc is None:
            return None

        player = calc.find_player(player_name, team_abbr)
        if player is None:
            return None

        mpg = player["mpg"]
        player_epm = player["tot"]

        # NaN 값 체크
        if math.isnan(mpg) or math.isnan(player_epm):
            return None

        # EPM 양수인 선수만 반영
        if player_epm <= 0:
            return None

        if mpg < calc.STARTER_MPG_THRESHOLD:
            return None

        bench_avg = calc.bench_avg_epm.get(team_abbr, -2.0)
        impact = (player_epm - bench_avg) * (mpg / 48)

        if abs(impact) < 0.5:
            return None

        return {
            "name": player_name,
            "epm": round(player_epm, 1),
            "mpg": round(mpg, 0),
            "impact": round(impact, 1),
        }

    def calculate_injury_impact(
        self,
        team_abbr: str,
        injuries: List[ESPNInjury]
    ) -> Tuple[float, List[Dict]]:
        """
        팀 부상 영향 계산.

        Args:
            team_abbr: 팀 약어
            injuries: 부상자 리스트

        Returns:
            (총 영향도, 선수별 상세)
        """
        calc = self.get_injury_calculator()
        if calc is None:
            return 0.0, []

        total_impact = 0.0
        details = []

        for injury in injuries:
            player = calc.find_player(injury.player_name, team_abbr)
            if player is None:
                continue

            mpg = player["mpg"]
            player_epm = player["tot"]

            # NaN 값 체크
            if math.isnan(mpg) or math.isnan(player_epm):
                continue

            # EPM 양수인 선수만 반영 (음수 선수는 빠져도 영향 없음)
            if player_epm <= 0:
                continue

            if mpg < calc.STARTER_MPG_THRESHOLD:
                continue

            bench_avg = calc.bench_avg_epm.get(team_abbr, -2.0)
            impact = (player_epm - bench_avg) * (mpg / 48)

            if abs(impact) < 0.5:
                continue

            total_impact += impact
            details.append({
                "name": injury.player_name,
                "epm": round(player_epm, 1),
                "mpg": round(mpg, 0),
                "impact": round(impact, 1),
                "detail": injury.detail
            })

        return round(total_impact, 1), details

    def get_games(self, game_date: date) -> List[Dict]:
        """
        경기 스케줄 및 결과 조회.

        scoreboardV2는 과거 경기 결과를 반환하지 않으므로,
        LeagueGameFinder를 사용하여 경기 결과를 조회합니다.

        Args:
            game_date: 경기 날짜

        Returns:
            경기 리스트 [{game_id, game_time, home_team_id, away_team_id, home_score, away_score, game_status, home_b2b, away_b2b}, ...]
        """
        date_str = game_date.strftime("%Y-%m-%d")

        try:
            # 오늘 또는 내일 경기인지 확인 (라이브 가능성)
            from datetime import datetime
            import pytz
            et = pytz.timezone('America/New_York')
            et_today = datetime.now(et).date()
            et_tomorrow = et_today + timedelta(days=1)

            # 오늘/내일 경기는 캐시 사용 안 함 (라이브 상태 실시간 반영)
            is_live_date = game_date >= et_today and game_date <= et_tomorrow
            use_cache = not is_live_date

            logger.info(f"Fetching scoreboard for {date_str}, use_cache={use_cache} (et_today={et_today})")

            # 1. scoreboardV2로 스케줄, 시간, 실시간 점수 가져오기
            scoreboard = self.nba_client.get_scoreboard(date_str, use_cache=use_cache)
            schedule_df = scoreboard.get("games", pd.DataFrame())
            line_score_df = scoreboard.get("line_score", pd.DataFrame())

            if schedule_df.empty:
                logger.info(f"No games found for {date_str}")
                return []

            # 디버그: schedule_df 컬럼과 GAME_STATUS_ID 분포 확인
            logger.info(f"Schedule columns: {list(schedule_df.columns)}")
            if "GAME_STATUS_ID" in schedule_df.columns:
                status_counts = schedule_df["GAME_STATUS_ID"].value_counts().to_dict()
                logger.info(f"GAME_STATUS_ID distribution: {status_counts}")
            else:
                logger.warning("GAME_STATUS_ID column not found in schedule_df!")

            # 2. line_score에서 실시간 점수 추출 (라이브/종료 경기 모두)
            # {game_id: {team_id: pts}}
            live_scores: Dict[str, Dict[int, int]] = {}
            if not line_score_df.empty:
                for _, row in line_score_df.iterrows():
                    game_id = str(row.get("GAME_ID", ""))
                    team_id = int(row.get("TEAM_ID", 0))
                    pts = row.get("PTS")

                    if game_id and team_id:
                        if game_id not in live_scores:
                            live_scores[game_id] = {}
                        # PTS가 None이 아니고 숫자인 경우만
                        if pts is not None and pd.notna(pts):
                            live_scores[game_id][team_id] = int(pts)

            logger.debug(f"Live scores from line_score: {live_scores}")

            # 3. LeagueGameFinder로 경기 결과 및 B2B 체크용 데이터 가져오기
            from src.utils.helpers import get_season_from_date
            season = get_season_from_date(game_date)
            results_df = self.nba_client.get_schedule(season=season)

            # B2B 체크를 위한 팀별 경기 날짜 수집
            team_game_dates: Dict[int, set] = {}
            if not results_df.empty:
                results_df['game_date'] = pd.to_datetime(results_df['game_date'])
                for _, row in results_df.iterrows():
                    team_id = row.get('team_id')
                    gdate = row['game_date'].date()
                    if team_id not in team_game_dates:
                        team_game_dates[team_id] = set()
                    team_game_dates[team_id].add(gdate)

            yesterday = game_date - timedelta(days=1)

            # 4. LeagueGameFinder에서 해당 날짜 경기 결과 추출 (과거 경기용)
            # {game_id: {home: {pts, result}, away: {pts, result}}}
            game_results: Dict[str, Dict] = {}
            if not results_df.empty:
                date_results = results_df[results_df['game_date'].dt.strftime('%Y-%m-%d') == date_str]
                for _, row in date_results.iterrows():
                    game_id = str(row['game_id'])
                    matchup = row.get('matchup', '')
                    is_home = ' vs. ' in matchup  # 홈팀은 "vs." 포함

                    if game_id not in game_results:
                        game_results[game_id] = {'home': None, 'away': None}

                    team_data = {
                        'team_id': row.get('team_id'),
                        'pts': row.get('pts'),
                        'result': row.get('result')
                    }
                    if is_home:
                        game_results[game_id]['home'] = team_data
                    else:
                        game_results[game_id]['away'] = team_data

            logger.info(f"Game results from leaguegamefinder: {len(game_results)} games")

            # 4. 스케줄과 점수 병합
            games = []
            for _, row in schedule_df.iterrows():
                game_id = str(row.get("GAME_ID", ""))

                # 경기 시간/상태 텍스트 파싱
                game_time = ""
                if "GAME_STATUS_TEXT" in row:
                    game_time = row["GAME_STATUS_TEXT"]
                elif "GAME_DATE_EST" in row:
                    game_time = row["GAME_DATE_EST"]

                home_team_id = int(row.get("HOME_TEAM_ID", 0))
                away_team_id = int(row.get("VISITOR_TEAM_ID", 0))

                # 경기 상태 판단
                # GAME_STATUS_ID: 1=예정, 2=진행중, 3=종료
                # LIVE_PERIOD: 현재 쿼터 (0=시작 전, 1-4=정규, 5+=연장)
                # GAME_STATUS_TEXT: "Final", "7:00 pm ET", "Q1 5:30", "Halftime" 등
                raw_status = int(row.get("GAME_STATUS_ID", 1))
                live_period = int(row.get("LIVE_PERIOD", 0))
                status_text = str(row.get("GAME_STATUS_TEXT", ""))

                # 경기 상태 결정 로직:
                # 1. GAME_STATUS_ID가 3이면 종료
                # 2. LIVE_PERIOD > 0 이면 라이브 (진행중)
                # 3. status_text가 "Final"이면 종료
                # 4. 그 외에는 예정
                if raw_status == 3 or "Final" in status_text:
                    game_status = 3  # 종료
                elif live_period > 0 or raw_status == 2:
                    game_status = 2  # 진행중 (라이브)
                else:
                    game_status = 1  # 예정

                # 점수 가져오기: line_score 우선, 없으면 leaguegamefinder에서 가져오기
                game_scores = live_scores.get(game_id, {})
                home_score = game_scores.get(home_team_id)
                away_score = game_scores.get(away_team_id)

                # line_score에 점수가 없으면 leaguegamefinder에서 가져오기 (과거 경기)
                game_result = game_results.get(game_id, {})
                if home_score is None and game_result.get('home'):
                    home_score = game_result['home'].get('pts')
                    if home_score is not None:
                        home_score = int(home_score)
                if away_score is None and game_result.get('away'):
                    away_score = game_result['away'].get('pts')
                    if away_score is not None:
                        away_score = int(away_score)

                # 점수가 있고 경기가 종료된 경우 처리
                # leaguegamefinder의 result 필드: 'W'/'L' = 종료, None = 진행중
                home_result = game_result.get('home', {})
                away_result = game_result.get('away', {})
                home_final = home_result.get('result') if home_result else None
                away_final = away_result.get('result') if away_result else None
                is_game_finished = home_final is not None and away_final is not None

                # 라이브 경기 감지: leaguegamefinder에 점수는 있지만 result가 None
                is_live_from_gamefinder = (
                    home_result and away_result and
                    home_result.get('pts') is not None and away_result.get('pts') is not None and
                    home_final is None and away_final is None
                )

                if home_score is not None and away_score is not None:
                    # 라이브 경기 체크를 먼저! (result=None이면 아직 진행중)
                    if is_live_from_gamefinder:
                        game_status = 2  # 라이브 경기 (점수 있고 result=None)
                    elif is_game_finished:
                        game_status = 3  # result='W'/'L'로 종료 확인됨
                    elif game_date < et_today:
                        game_status = 3  # 과거 날짜 (leaguegamefinder에 없는 경우)

                # 로깅 (디버그용)
                logger.debug(f"Game {game_id}: raw_status={raw_status}, live_period={live_period}, "
                           f"status_text={status_text}, final_status={game_status}, scores={home_score}-{away_score}")
                if game_status == 2:
                    if is_live_from_gamefinder:
                        logger.info(f"🔴 Live game {game_id} (from gamefinder): home={home_score}, away={away_score}")
                    else:
                        logger.info(f"🔴 Live game {game_id}: period={live_period}, home={home_score}, away={away_score}")
                elif game_status == 3 and home_score is not None:
                    logger.info(f"✅ Finished game {game_id}: home={home_score}, away={away_score}")

                # B2B 체크 (전날 경기 여부)
                home_b2b = yesterday in team_game_dates.get(home_team_id, set())
                away_b2b = yesterday in team_game_dates.get(away_team_id, set())

                games.append({
                    "game_id": game_id,
                    "game_time": game_time,
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "game_status": game_status,
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_b2b": home_b2b,
                    "away_b2b": away_b2b,
                })

            # game_id 기준으로 정렬 (경기 순서 보장)
            # game_id 형식: 00224MMDDNNNN (MMDD=날짜, NNNN=경기번호순)
            games.sort(key=lambda g: g.get("game_id", ""))

            logger.info(f"Found {len(games)} games for {date_str}, sorted by game_id")
            return games

        except Exception as e:
            logger.error(f"Error fetching games for {date_str}: {e}")
            return []

    def get_today_games(self, game_date: date) -> List[Dict]:
        """get_games의 별칭 (하위 호환성)"""
        return self.get_games(game_date)

    # =========================================================================
    # V4.2 피처 빌딩 메서드
    # =========================================================================

    def load_team_game_logs(self, target_date: date) -> pd.DataFrame:
        """
        팀 게임 로그 로드 (V4 피처 계산용).

        Args:
            target_date: 기준 날짜

        Returns:
            팀 게임 로그 DataFrame
        """
        if self._team_game_logs_cache is not None:
            return self._team_game_logs_cache

        try:
            from src.utils.helpers import get_season_from_date
            season = get_season_from_date(target_date)
            logs = self.nba_client.get_team_game_logs(season=season)

            if not logs.empty:
                # 컬럼명 정규화
                column_mapping = {
                    'TEAM_ID': 'team_id',
                    'GAME_ID': 'game_id',
                    'GAME_DATE': 'game_date',
                    'MATCHUP': 'matchup',
                    'WL': 'result',
                    'PTS': 'pts',
                    'FGM': 'fg',
                    'FGA': 'fga',
                    'FG3M': 'fg3',
                    'FG3A': 'fg3a',
                    'FTM': 'ft',
                    'FTA': 'fta',
                    'OREB': 'orb',
                    'DREB': 'drb',
                    'REB': 'reb',
                    'PLUS_MINUS': 'margin',
                }
                rename_dict = {k: v for k, v in column_mapping.items() if k in logs.columns}
                logs = logs.rename(columns=rename_dict)

                # 날짜 변환
                logs['game_date'] = pd.to_datetime(logs['game_date'])

                # 홈/원정 구분
                logs['is_home'] = logs['matchup'].str.contains(' vs. ')

                # 상대팀 점수 계산 (margin = pts - opp_pts)
                logs['opp_pts'] = logs['pts'] - logs['margin']

                # 메모리 최적화
                logs = optimize_dataframe(logs, verbose=True)

                self._team_game_logs_cache = logs
                logger.info(f"Loaded {len(logs)} team game logs for season {season}")

            return logs if not logs.empty else pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading team game logs: {e}")
            return pd.DataFrame()

    def _compute_team_stats(
        self,
        team_id: int,
        logs: pd.DataFrame,
        target_date: date,
        window: int = 10
    ) -> Dict[str, float]:
        """
        팀별 V4 피처용 통계 계산.

        Args:
            team_id: 팀 ID
            logs: 팀 게임 로그
            target_date: 기준 날짜
            window: 롤링 윈도우

        Returns:
            팀 통계 딕셔너리
        """
        # 해당 팀의 과거 경기만 필터링
        team_logs = logs[
            (logs['team_id'] == team_id) &
            (logs['game_date'] < pd.Timestamp(target_date))
        ].sort_values('game_date', ascending=False)

        if len(team_logs) == 0:
            return self._default_team_stats()

        recent = team_logs.head(window)
        last5 = team_logs.head(5)

        # Four Factors
        efg_pct = self._calc_efg(recent)
        ft_rate = self._calc_ft_rate(recent)

        # 모멘텀
        last5_win_pct = (last5['result'] == 'W').mean() if len(last5) > 0 else 0.5
        streak = self._calc_streak(team_logs)
        margin_ewma = self._calc_ewma_margin(team_logs, span=5, window=window)

        # 리바운드
        orb_avg = recent['orb'].mean() if 'orb' in recent.columns and len(recent) > 0 else 10.0

        # 원정 승률
        away_games = team_logs[team_logs['is_home'] == False]
        away_win_pct = (away_games['result'] == 'W').mean() if len(away_games) > 0 else 0.45

        return {
            'efg_pct': efg_pct,
            'ft_rate': ft_rate,
            'last5_win_pct': last5_win_pct,
            'streak': streak,
            'margin_ewma': margin_ewma,
            'orb_avg': orb_avg,
            'away_win_pct': away_win_pct,
        }

    def _default_team_stats(self) -> Dict[str, float]:
        """기본값 반환"""
        return {
            'efg_pct': 0.50,
            'ft_rate': 0.20,
            'last5_win_pct': 0.5,
            'streak': 0,
            'margin_ewma': 0.0,
            'orb_avg': 10.0,
            'away_win_pct': 0.45,
        }

    def _calc_efg(self, games: pd.DataFrame) -> float:
        """eFG% 계산: (FG + 0.5 * 3P) / FGA"""
        if len(games) == 0:
            return 0.50
        fg = games['fg'].sum() if 'fg' in games.columns else 0
        fg3 = games['fg3'].sum() if 'fg3' in games.columns else 0
        fga = games['fga'].sum() if 'fga' in games.columns else 0
        if fga == 0:
            return 0.50
        return (fg + 0.5 * fg3) / fga

    def _calc_ft_rate(self, games: pd.DataFrame) -> float:
        """FT Rate 계산: FTM / FGA"""
        if len(games) == 0:
            return 0.20
        ft = games['ft'].sum() if 'ft' in games.columns else 0
        fga = games['fga'].sum() if 'fga' in games.columns else 0
        if fga == 0:
            return 0.20
        return ft / fga

    def _calc_streak(self, games: pd.DataFrame) -> int:
        """연승/연패 계산 (양수=연승, 음수=연패)"""
        if len(games) == 0:
            return 0
        streak = 0
        first_result = games.iloc[0]['result']
        for _, row in games.iterrows():
            if row['result'] == first_result:
                streak += 1 if first_result == 'W' else -1
            else:
                break
        return min(max(streak, -10), 10)

    def _calc_ewma_margin(
        self,
        games: pd.DataFrame,
        span: int = 5,
        window: int = 10
    ) -> float:
        """EWMA 마진 계산"""
        if len(games) < 3:
            return 0.0
        margins = games.head(window)['margin']
        if len(margins) == 0:
            return 0.0
        return margins.ewm(span=span, adjust=False).mean().iloc[0]

    def build_v4_features(
        self,
        home_team_id: int,
        away_team_id: int,
        team_epm: Dict[int, Dict],
        target_date: date
    ) -> Dict[str, float]:
        """
        V4.2 피처 생성 (11개).

        Args:
            home_team_id: 홈팀 ID
            away_team_id: 원정팀 ID
            team_epm: 팀 EPM 데이터 (DNT API)
            target_date: 기준 날짜

        Returns:
            V4 피처 딕셔너리 (11개)
        """
        # 팀 게임 로그 로드
        logs = self.load_team_game_logs(target_date)

        # 팀별 통계 계산
        if home_team_id not in self._team_stats_cache:
            self._team_stats_cache[home_team_id] = self._compute_team_stats(
                home_team_id, logs, target_date
            )
        if away_team_id not in self._team_stats_cache:
            self._team_stats_cache[away_team_id] = self._compute_team_stats(
                away_team_id, logs, target_date
            )

        home_stats = self._team_stats_cache[home_team_id]
        away_stats = self._team_stats_cache[away_team_id]

        # EPM 데이터
        home_epm = team_epm.get(home_team_id, {})
        away_epm = team_epm.get(away_team_id, {})

        def safe_diff(h_val, a_val, default=0):
            h = h_val if h_val is not None else default
            a = a_val if a_val is not None else default
            return h - a

        # margin_ewma_diff 클리핑 (이상치 방지: ±30점 제한)
        # 단일 경기 대패(-41점 등)로 인한 과도한 영향 방지
        raw_margin_ewma_diff = home_stats['margin_ewma'] - away_stats['margin_ewma']
        clipped_margin_ewma_diff = max(-30.0, min(30.0, raw_margin_ewma_diff))

        # V4.2 11개 피처
        return {
            # EPM 핵심 (4개)
            'team_epm_diff': safe_diff(home_epm.get('team_epm'), away_epm.get('team_epm'), 0),
            'team_oepm_diff': safe_diff(home_epm.get('team_oepm'), away_epm.get('team_oepm'), 0),
            'team_depm_diff': safe_diff(home_epm.get('team_depm'), away_epm.get('team_depm'), 0),
            'sos_diff': safe_diff(home_epm.get('sos'), away_epm.get('sos'), 0),
            # 모멘텀 (3개)
            'last5_win_pct_diff': home_stats['last5_win_pct'] - away_stats['last5_win_pct'],
            'streak_diff': home_stats['streak'] - away_stats['streak'],
            'margin_ewma_diff': clipped_margin_ewma_diff,
            # Four Factors (2개)
            'efg_pct_diff': home_stats['efg_pct'] - away_stats['efg_pct'],
            'ft_rate_diff': home_stats['ft_rate'] - away_stats['ft_rate'],
            # 컨텍스트 (1개)
            'away_road_strength': away_stats['away_win_pct'] - 0.5,
            # 리바운드 (1개)
            'orb_diff': home_stats['orb_avg'] - away_stats['orb_avg'],
        }

    # =========================================================================
    # V4.3 선수 EPM 피처 빌딩 메서드
    # =========================================================================

    def load_player_epm(self, season: int) -> pd.DataFrame:
        """
        시즌별 선수 EPM 데이터 로드 (DNT API 실시간 호출).

        Args:
            season: 시즌 연도 (예: 2026)

        Returns:
            선수 EPM DataFrame (player_id, player_name, team_id, team_alias, tot, mpg 등)
        """
        if season in self._player_epm_cache:
            return self._player_epm_cache[season]

        try:
            # DNT API 실시간 호출 - 시즌 전체 EPM 엔드포인트 사용
            player_epm_raw = self.dnt_client.get_season_epm(season=season)

            if not player_epm_raw:
                logger.warning(f"No player EPM data from DNT API for season {season}")
                return pd.DataFrame()

            df = pd.DataFrame(player_epm_raw)
            df = optimize_dataframe(df)  # 메모리 최적화
            self._player_epm_cache[season] = df
            logger.info(f"Loaded player EPM for season {season}: {len(df)} players")
            return df
        except Exception as e:
            logger.error(f"Error loading player EPM for season {season}: {e}")
            return pd.DataFrame()

    def _get_team_players(self, team_id: int, season: int) -> pd.DataFrame:
        """팀의 선수 EPM 데이터 조회"""
        player_epm = self.load_player_epm(season)
        if player_epm.empty:
            return pd.DataFrame()
        return player_epm[player_epm['team_id'] == team_id]

    def _calc_rotation_epm(self, team_id: int, season: int, min_mpg: float = 12.0) -> float:
        """
        로테이션 선수(MPG >= min_mpg)의 가중 평균 EPM.

        공식: Σ(EPM_i × MPG_i) / Σ(MPG_i)
        """
        players = self._get_team_players(team_id, season)
        if len(players) == 0:
            return 0.0

        rotation = players[players['mpg'] >= min_mpg]
        if len(rotation) == 0 or rotation['mpg'].sum() == 0:
            return 0.0

        weighted_epm = (rotation['tot'] * rotation['mpg']).sum() / rotation['mpg'].sum()
        return weighted_epm

    def _calc_bench_strength(self, team_id: int, season: int) -> float:
        """
        벤치 선수(6-10번째 MPG)의 평균 EPM.
        """
        players = self._get_team_players(team_id, season)
        if len(players) < 6:
            return -2.0

        sorted_players = players.nlargest(10, 'mpg')
        bench = sorted_players.iloc[5:10] if len(sorted_players) >= 10 else sorted_players.iloc[5:]

        if len(bench) == 0:
            return -2.0

        return bench['tot'].mean()

    def build_v4_3_features(
        self,
        home_team_id: int,
        away_team_id: int,
        team_epm: Dict[int, Dict],
        target_date: date
    ) -> Dict[str, float]:
        """
        V4.3 피처 생성 (13개 = V4.2 11개 + 선수 EPM 2개).

        Args:
            home_team_id: 홈팀 ID
            away_team_id: 원정팀 ID
            team_epm: 팀 EPM 데이터 (DNT API)
            target_date: 기준 날짜

        Returns:
            V4.3 피처 딕셔너리 (13개)
        """
        # V4.2 기본 피처 (11개)
        features = self.build_v4_features(home_team_id, away_team_id, team_epm, target_date)

        # V4.3 선수 EPM 피처 추가 (2개)
        from src.utils.helpers import get_season_from_date
        season = get_season_from_date(target_date)

        h_rotation = self._calc_rotation_epm(home_team_id, season)
        a_rotation = self._calc_rotation_epm(away_team_id, season)
        features['player_rotation_epm_diff'] = h_rotation - a_rotation

        h_bench = self._calc_bench_strength(home_team_id, season)
        a_bench = self._calc_bench_strength(away_team_id, season)
        features['bench_strength_diff'] = h_bench - a_bench

        return features

    # =========================================================================
    # V5.2 피처 빌딩 메서드
    # =========================================================================

    def _calc_top5_epm(self, team_id: int, season: int) -> float:
        """
        상위 5명 선수(MPG 기준)의 평균 EPM.
        """
        players = self._get_team_players(team_id, season)
        if len(players) < 5:
            return 0.0
        return players.nlargest(5, 'mpg')['tot'].mean()

    def _calc_orb_pct(self, games: pd.DataFrame) -> float:
        """
        공격 리바운드 비율: ORB / (ORB + DRB)
        """
        if len(games) == 0:
            return 0.25
        orb = games['orb'].sum() if 'orb' in games.columns else 0
        drb = games['drb'].sum() if 'drb' in games.columns else 0
        total = orb + drb
        if total == 0:
            return 0.25
        return orb / total

    def _calc_last3_win_pct(self, games: pd.DataFrame) -> float:
        """최근 3경기 승률"""
        if len(games) == 0:
            return 0.5
        last3 = games.head(3)
        return (last3['result'] == 'W').mean()

    def _calc_rest_days(
        self,
        team_id: int,
        target_date: date,
        logs: pd.DataFrame
    ) -> int:
        """
        전 경기로부터의 휴식일 수 계산.

        Args:
            team_id: 팀 ID
            target_date: 기준 날짜
            logs: 팀 게임 로그

        Returns:
            휴식일 수 (0=B2B, 1=1일 휴식, ..., 최대 7)
        """
        team_logs = logs[logs['team_id'] == team_id].copy()
        if len(team_logs) == 0:
            return 7  # 데이터 없으면 충분히 쉰 것으로

        team_logs['game_date'] = pd.to_datetime(team_logs['game_date'])
        target_dt = pd.to_datetime(target_date)

        # target_date 이전 경기만
        prev_games = team_logs[team_logs['game_date'] < target_dt]
        if len(prev_games) == 0:
            return 7  # 시즌 첫 경기

        last_game_date = prev_games['game_date'].max()
        rest_days = (target_dt - last_game_date).days - 1  # 경기일 제외

        # 클리핑 (0~7)
        return min(max(rest_days, 0), 7)

    def build_v5_2_features(
        self,
        home_team_id: int,
        away_team_id: int,
        team_epm: Dict[int, Dict],
        target_date: date,
        home_b2b: bool = False,
        away_b2b: bool = False
    ) -> Dict[str, float]:
        """
        V5.2 피처 생성 (11개).

        Args:
            home_team_id: 홈팀 ID
            away_team_id: 원정팀 ID
            team_epm: 팀 EPM 데이터 (DNT API)
            target_date: 기준 날짜
            home_b2b: 홈팀 B2B 여부
            away_b2b: 어웨이팀 B2B 여부

        Returns:
            V5.2 피처 딕셔너리 (11개)
        """
        from src.utils.helpers import get_season_from_date
        season = get_season_from_date(target_date)

        # 팀 게임 로그 로드
        logs = self.load_team_game_logs(target_date)

        # EPM 데이터
        home_epm_data = team_epm.get(home_team_id, {})
        away_epm_data = team_epm.get(away_team_id, {})

        def safe_get(d, key, default=0):
            val = d.get(key)
            return val if val is not None else default

        # 팀별 최근 경기 통계
        home_logs = logs[logs['team_id'] == home_team_id].copy() if not logs.empty else pd.DataFrame()
        away_logs = logs[logs['team_id'] == away_team_id].copy() if not logs.empty else pd.DataFrame()

        if not home_logs.empty:
            home_logs['game_date'] = pd.to_datetime(home_logs['game_date'])
            home_logs = home_logs[home_logs['game_date'] < pd.to_datetime(target_date)]
            home_logs = home_logs.sort_values('game_date', ascending=False)

        if not away_logs.empty:
            away_logs['game_date'] = pd.to_datetime(away_logs['game_date'])
            away_logs = away_logs[away_logs['game_date'] < pd.to_datetime(target_date)]
            away_logs = away_logs.sort_values('game_date', ascending=False)

        # 최근 10경기
        home_recent = home_logs.head(10) if not home_logs.empty else pd.DataFrame()
        away_recent = away_logs.head(10) if not away_logs.empty else pd.DataFrame()

        # 휴식일 계산
        home_rest = self._calc_rest_days(home_team_id, target_date, logs)
        away_rest = self._calc_rest_days(away_team_id, target_date, logs)

        # V5.2 11개 피처
        features = {
            # EPM 핵심 (4개)
            'team_epm_diff': safe_get(home_epm_data, 'team_epm') - safe_get(away_epm_data, 'team_epm'),
            'player_rotation_epm_diff': self._calc_rotation_epm(home_team_id, season) - self._calc_rotation_epm(away_team_id, season),
            'bench_strength_diff': self._calc_bench_strength(home_team_id, season) - self._calc_bench_strength(away_team_id, season),
            'top5_epm_diff': self._calc_top5_epm(home_team_id, season) - self._calc_top5_epm(away_team_id, season),

            # Four Factors (3개)
            'ft_rate_diff': self._calc_ft_rate(home_recent) - self._calc_ft_rate(away_recent),
            'orb_pct_diff': self._calc_orb_pct(home_recent) - self._calc_orb_pct(away_recent),
            'efg_pct_diff': self._calc_efg(home_recent) - self._calc_efg(away_recent),

            # 모멘텀 (2개)
            'last3_win_pct_diff': self._calc_last3_win_pct(home_logs) - self._calc_last3_win_pct(away_logs),
            'last5_win_pct_diff': (
                ((home_logs.head(5)['result'] == 'W').mean() if len(home_logs) > 0 else 0.5) -
                ((away_logs.head(5)['result'] == 'W').mean() if len(away_logs) > 0 else 0.5)
            ),

            # 피로도 (2개) - V5.2 신규
            'b2b_diff': (1 if home_b2b else 0) - (1 if away_b2b else 0),
            'rest_days_diff': home_rest - away_rest,
        }

        return features

    def build_v5_4_features(
        self,
        home_team_id: int,
        away_team_id: int,
        team_epm: Dict[int, Dict],
        target_date: date
    ) -> Dict[str, float]:
        """
        V5.4 피처 생성 (5개).

        리팩토링 Phase 2: V54FeatureBuilder로 위임.

        Args:
            home_team_id: 홈팀 ID
            away_team_id: 원정팀 ID
            team_epm: 팀 EPM 데이터 (DNT API)
            target_date: 기준 날짜

        Returns:
            V5.4 피처 딕셔너리 (5개)
        """
        from src.utils.helpers import get_season_from_date
        season = get_season_from_date(target_date)

        # 팀 게임 로그 로드
        logs = self.load_team_game_logs(target_date)

        # 선수 EPM 데이터 로드
        player_epm = self.load_player_epm(season)

        # 피처 빌드 컨텍스트 생성
        context = FeatureBuildContext(
            game_date=target_date,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            team_logs=logs,
            team_epm=team_epm,
            player_epm=player_epm,
        )

        # V54FeatureBuilder로 피처 생성
        builder = V54FeatureBuilder()
        result = builder.build(context)

        if result.is_valid:
            return result.features
        else:
            # 에러 시 폴백: 기존 로직 사용
            logger.warning(f"V54FeatureBuilder failed: {result.error_message}, using fallback")
            return self._build_v5_4_features_fallback(
                home_team_id, away_team_id, team_epm, target_date, season, logs
            )

    def _build_v5_4_features_fallback(
        self,
        home_team_id: int,
        away_team_id: int,
        team_epm: Dict[int, Dict],
        target_date: date,
        season: int,
        logs: pd.DataFrame
    ) -> Dict[str, float]:
        """V5.4 피처 폴백 (기존 로직)"""
        home_epm_data = team_epm.get(home_team_id, {})
        away_epm_data = team_epm.get(away_team_id, {})

        def safe_get(d, key, default=0):
            val = d.get(key)
            return val if val is not None else default

        # 팀별 최근 경기 통계
        home_logs = logs[logs['team_id'] == home_team_id].copy() if not logs.empty else pd.DataFrame()
        away_logs = logs[logs['team_id'] == away_team_id].copy() if not logs.empty else pd.DataFrame()

        if not home_logs.empty:
            home_logs['game_date'] = pd.to_datetime(home_logs['game_date'])
            home_logs = home_logs[home_logs['game_date'] < pd.to_datetime(target_date)]
            home_logs = home_logs.sort_values('game_date', ascending=False)

        if not away_logs.empty:
            away_logs['game_date'] = pd.to_datetime(away_logs['game_date'])
            away_logs = away_logs[away_logs['game_date'] < pd.to_datetime(target_date)]
            away_logs = away_logs.sort_values('game_date', ascending=False)

        # 최근 10경기
        home_recent = home_logs.head(10) if not home_logs.empty else pd.DataFrame()
        away_recent = away_logs.head(10) if not away_logs.empty else pd.DataFrame()

        # V5.4 5개 피처
        features = {
            'team_epm_diff': safe_get(home_epm_data, 'team_epm') - safe_get(away_epm_data, 'team_epm'),
            'sos_diff': safe_get(home_epm_data, 'sos') - safe_get(away_epm_data, 'sos'),
            'bench_strength_diff': self._calc_bench_strength(home_team_id, season) - self._calc_bench_strength(away_team_id, season),
            'top5_epm_diff': self._calc_top5_epm(home_team_id, season) - self._calc_top5_epm(away_team_id, season),
            'ft_rate_diff': self._calc_ft_rate(home_recent) - self._calc_ft_rate(away_recent),
        }

        return features

    def clear_cache(self) -> None:
        """캐시 초기화"""
        self._team_epm_date_cache = {}
        self._team_game_logs_cache = None
        self._player_game_logs_cache = None
        self._team_stats_cache = {}
        self._player_epm_cache = {}
        self._advanced_injury_calc = None
        self.espn_client.clear_cache()

    # =========================================================================
    # V4.4 고급 부상 영향력 계산
    # =========================================================================

    def load_player_game_logs(self, target_date: date) -> pd.DataFrame:
        """
        선수 경기 로그 로드.

        Args:
            target_date: 기준 날짜

        Returns:
            선수 경기 로그 DataFrame
        """
        if self._player_game_logs_cache is not None:
            return self._player_game_logs_cache

        try:
            from src.utils.helpers import get_season_from_date
            season = get_season_from_date(target_date)
            logs = self.nba_client.get_player_game_logs(season=season)

            if not logs.empty:
                logger.info(f"Loaded {len(logs)} player game logs for season {season}")
                self._player_game_logs_cache = logs

            return logs if not logs.empty else pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading player game logs: {e}")
            return pd.DataFrame()

    def get_advanced_injury_calculator(
        self,
        target_date: date,
        team_epm: Dict[int, Dict]
    ) -> Optional[AdvancedInjuryImpactCalculator]:
        """
        고급 부상 영향력 계산기 반환.

        Args:
            target_date: 기준 날짜
            team_epm: 팀 EPM 딕셔너리

        Returns:
            AdvancedInjuryImpactCalculator 또는 None
        """
        if self._advanced_injury_calc is not None:
            return self._advanced_injury_calc

        try:
            from src.utils.helpers import get_season_from_date
            season = get_season_from_date(target_date)

            # 팀 및 선수 경기 로그 로드
            team_logs = self.load_team_game_logs(target_date)
            player_logs = self.load_player_game_logs(target_date)

            if team_logs.empty or player_logs.empty:
                logger.warning("경기 로그 데이터 부족으로 고급 부상 계산기 생성 실패")
                return None

            # 선수 EPM 데이터 로드 (DNT API 실시간)
            player_epm_df = self.load_player_epm(season)
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
        target_date: date,
        team_epm: Dict[int, Dict],
        gtd_included: Optional[Dict[str, bool]] = None
    ) -> Tuple[float, float, List[PlayerImpactResult]]:
        """
        경기별 고급 부상 영향력 계산.

        Args:
            home_abbr: 홈팀 약어
            away_abbr: 원정팀 약어
            target_date: 기준 날짜
            team_epm: 팀 EPM 딕셔너리
            gtd_included: GTD 선수 포함 여부 딕셔너리

        Returns:
            (home_impact, away_impact, player_details)
            양수 = 해당 팀에 불리
        """
        calc = self.get_advanced_injury_calculator(target_date, team_epm)
        if calc is None:
            return 0.0, 0.0, []

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
        team_epm: Dict[int, Dict]
    ) -> Dict:
        """
        팀 부상자 요약 정보 반환.

        Args:
            team_abbr: 팀 약어
            target_date: 기준 날짜
            team_epm: 팀 EPM 딕셔너리

        Returns:
            {
                "out_players": [...],
                "gtd_players": [...],
                "total_impact": float
            }
        """
        calc = self.get_advanced_injury_calculator(target_date, team_epm)

        out_injuries = self.get_injuries(team_abbr)
        # NBA PDF로 세분화된 GTD 상태 조회
        gtd_with_status = self.get_gtd_players_with_status(team_abbr, target_date)

        out_details = []
        gtd_details = []
        total_impact = 0.0

        for inj in out_injuries:
            if calc:
                result = calc.calculate_player_impact(inj.player_name, team_abbr)
                if result.is_valid:
                    # prob_shift는 이미 0 이상 값만 반환 (음수는 0 처리됨)
                    total_impact += result.prob_shift
                    out_details.append({
                        "name": inj.player_name,
                        "status": "Out",
                        "detail": inj.detail,
                        "epm": round(result.epm, 2),
                        "mpg": round(result.mpg, 1),
                        "prob_shift": round(result.prob_shift * 100, 1),  # %로 변환
                        "played_games": result.played_games,
                        "missed_games": result.missed_games,
                        "schedule_diff": round(result.schedule_diff, 3),  # 성과 차이
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

        for gtd_info in gtd_with_status:
            player_name = gtd_info["player_name"]
            # NBA PDF에서 세분화된 상태 (Probable 25%, Questionable 50%, Doubtful 75%)
            miss_prob = gtd_info["miss_probability"]
            nba_status = gtd_info.get("nba_status") or "GTD"

            if calc:
                result = calc.calculate_player_impact(player_name, team_abbr)
                if result.is_valid:
                    # 세분화된 상태별 가중치 적용
                    # Out: 100%, Probable: 25%, Questionable: 50%, Doubtful: 75%
                    gtd_impact = result.prob_shift * miss_prob
                    total_impact += gtd_impact
                    gtd_details.append({
                        "name": player_name,
                        "status": nba_status,
                        "detail": gtd_info.get("detail"),
                        "epm": round(result.epm, 2),
                        "mpg": round(result.mpg, 1),
                        "prob_shift": round(result.prob_shift * 100, 1),  # 원래 영향력 (%)
                        "applied_shift": round(gtd_impact * 100, 1),  # 실제 반영된 영향력
                        "miss_probability": round(miss_prob * 100, 0),  # 결장 확률 (%)
                        "played_games": result.played_games,
                        "missed_games": result.missed_games,
                        "schedule_diff": round(result.schedule_diff, 3),  # 성과 차이
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
            "total_prob_shift": round(total_impact * 100, 1),  # %로 변환 (승률 감소)
        }
