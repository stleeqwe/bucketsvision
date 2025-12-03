"""
경기 스케줄 서비스.

리팩토링 Phase 2.4: data_loader.py의 get_games() 메서드 분해.

책임:
- 경기 스케줄 조회
- 라이브 점수 추출
- 경기 상태 판단
- B2B 체크
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz

from src.data_collection.nba_stats_client import NBAStatsClient
from src.utils.logger import logger
from src.utils.helpers import get_season_from_date


class GameScheduleService:
    """
    경기 스케줄 및 상태 조회 서비스.

    NBA Stats API에서 경기 스케줄, 점수, 상태를 조회하고
    B2B 여부를 판단합니다.
    """

    # 경기 상태 상수
    GAME_STATUS_SCHEDULED = 1
    GAME_STATUS_LIVE = 2
    GAME_STATUS_FINAL = 3

    def __init__(self, nba_client: Optional[NBAStatsClient] = None):
        """
        Args:
            nba_client: NBA Stats API 클라이언트
        """
        self.nba_client = nba_client or NBAStatsClient()
        self._et = pytz.timezone('America/New_York')

    def get_games(self, game_date: date) -> List[Dict]:
        """
        경기 스케줄 및 결과 조회.

        Args:
            game_date: 경기 날짜

        Returns:
            경기 리스트 [{game_id, game_time, home_team_id, away_team_id,
                        home_score, away_score, game_status, home_b2b, away_b2b}, ...]
        """
        date_str = game_date.strftime("%Y-%m-%d")

        try:
            # 1. 캐시 사용 여부 판단
            use_cache = self._should_use_cache(game_date)
            logger.info(f"Fetching scoreboard for {date_str}, use_cache={use_cache}")

            # 2. 스코어보드 데이터 조회
            schedule_df, line_score_df = self._fetch_scoreboard(date_str, use_cache)
            if schedule_df.empty:
                logger.info(f"No games found for {date_str}")
                return []

            # 3. 라이브 점수 추출
            live_scores = self._extract_live_scores(line_score_df)

            # 4. LeagueGameFinder 데이터 조회 (결과 + B2B용)
            season = get_season_from_date(game_date)
            results_df = self.nba_client.get_schedule(season=season)

            # 5. B2B 체크용 팀별 경기 날짜
            team_game_dates = self._build_team_game_dates(results_df)

            # 6. 해당 날짜 경기 결과 추출
            game_results = self._extract_game_results(results_df, date_str)

            # 7. 경기 정보 병합
            et_today = datetime.now(self._et).date()
            yesterday = game_date - timedelta(days=1)

            games = []
            for _, row in schedule_df.iterrows():
                game = self._process_game_row(
                    row=row,
                    game_date=game_date,
                    live_scores=live_scores,
                    game_results=game_results,
                    team_game_dates=team_game_dates,
                    et_today=et_today,
                    yesterday=yesterday,
                )
                games.append(game)

            # game_id 기준 정렬
            games.sort(key=lambda g: g.get("game_id", ""))
            logger.info(f"Found {len(games)} games for {date_str}")
            return games

        except Exception as e:
            logger.error(f"Error fetching games for {date_str}: {e}")
            return []

    def _should_use_cache(self, game_date: date) -> bool:
        """
        캐시 사용 여부 판단.

        오늘/내일 경기는 라이브 상태 실시간 반영을 위해 캐시 사용 안 함.

        Args:
            game_date: 경기 날짜

        Returns:
            캐시 사용 여부
        """
        et_today = datetime.now(self._et).date()
        et_tomorrow = et_today + timedelta(days=1)
        is_live_date = et_today <= game_date <= et_tomorrow
        return not is_live_date

    def _fetch_scoreboard(
        self,
        date_str: str,
        use_cache: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        스코어보드 데이터 조회.

        Args:
            date_str: 날짜 문자열 (YYYY-MM-DD)
            use_cache: 캐시 사용 여부

        Returns:
            (schedule_df, line_score_df)
        """
        scoreboard = self.nba_client.get_scoreboard(date_str, use_cache=use_cache)
        schedule_df = scoreboard.get("games", pd.DataFrame())
        line_score_df = scoreboard.get("line_score", pd.DataFrame())

        # 디버그 로깅
        if not schedule_df.empty and "GAME_STATUS_ID" in schedule_df.columns:
            status_counts = schedule_df["GAME_STATUS_ID"].value_counts().to_dict()
            logger.debug(f"GAME_STATUS_ID distribution: {status_counts}")

        return schedule_df, line_score_df

    def _extract_live_scores(
        self,
        line_score_df: pd.DataFrame
    ) -> Dict[str, Dict[int, int]]:
        """
        라이브 점수 추출.

        Args:
            line_score_df: 라인 스코어 DataFrame

        Returns:
            {game_id: {team_id: pts}}
        """
        live_scores: Dict[str, Dict[int, int]] = {}

        if line_score_df.empty:
            return live_scores

        for _, row in line_score_df.iterrows():
            game_id = str(row.get("GAME_ID", ""))
            team_id = int(row.get("TEAM_ID", 0))
            pts = row.get("PTS")

            if game_id and team_id:
                if game_id not in live_scores:
                    live_scores[game_id] = {}
                if pts is not None and pd.notna(pts):
                    live_scores[game_id][team_id] = int(pts)

        return live_scores

    def _build_team_game_dates(
        self,
        results_df: pd.DataFrame
    ) -> Dict[int, set]:
        """
        B2B 체크용 팀별 경기 날짜 수집.

        Args:
            results_df: LeagueGameFinder 결과 DataFrame

        Returns:
            {team_id: {date, ...}}
        """
        team_game_dates: Dict[int, set] = {}

        if results_df.empty:
            return team_game_dates

        results_df['game_date'] = pd.to_datetime(results_df['game_date'])

        for _, row in results_df.iterrows():
            team_id = row.get('team_id')
            gdate = row['game_date'].date()
            if team_id not in team_game_dates:
                team_game_dates[team_id] = set()
            team_game_dates[team_id].add(gdate)

        return team_game_dates

    def _extract_game_results(
        self,
        results_df: pd.DataFrame,
        date_str: str
    ) -> Dict[str, Dict]:
        """
        해당 날짜 경기 결과 추출.

        Args:
            results_df: LeagueGameFinder 결과 DataFrame
            date_str: 날짜 문자열

        Returns:
            {game_id: {home: {...}, away: {...}}}
        """
        game_results: Dict[str, Dict] = {}

        if results_df.empty:
            return game_results

        results_df['game_date'] = pd.to_datetime(results_df['game_date'])
        date_results = results_df[results_df['game_date'].dt.strftime('%Y-%m-%d') == date_str]

        for _, row in date_results.iterrows():
            game_id = str(row['game_id'])
            matchup = row.get('matchup', '')
            is_home = ' vs. ' in matchup

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

        logger.debug(f"Game results from leaguegamefinder: {len(game_results)} games")
        return game_results

    def _determine_game_status(
        self,
        row: pd.Series,
        game_results: Dict[str, Dict],
        game_date: date,
        et_today: date,
        live_scores: Dict[str, Dict[int, int]],
    ) -> Tuple[int, Optional[int], Optional[int]]:
        """
        경기 상태 및 점수 판단.

        Args:
            row: 스케줄 DataFrame의 행
            game_results: LeagueGameFinder 결과
            game_date: 경기 날짜
            et_today: 오늘 날짜 (ET)
            live_scores: 라이브 점수 데이터

        Returns:
            (game_status, home_score, away_score)
        """
        game_id = str(row.get("GAME_ID", ""))
        home_team_id = int(row.get("HOME_TEAM_ID", 0))
        away_team_id = int(row.get("VISITOR_TEAM_ID", 0))

        # 기본 상태 판단
        raw_status = int(row.get("GAME_STATUS_ID", 1))
        live_period = int(row.get("LIVE_PERIOD", 0))
        status_text = str(row.get("GAME_STATUS_TEXT", ""))

        if raw_status == 3 or "Final" in status_text:
            game_status = self.GAME_STATUS_FINAL
        elif live_period > 0 or raw_status == 2:
            game_status = self.GAME_STATUS_LIVE
        else:
            game_status = self.GAME_STATUS_SCHEDULED

        # 점수 추출 (라이브 스코어 우선)
        game_scores = live_scores.get(game_id, {})
        home_score = game_scores.get(home_team_id)
        away_score = game_scores.get(away_team_id)

        # LeagueGameFinder 백업
        game_result = game_results.get(game_id, {})
        if home_score is None and game_result.get('home'):
            pts = game_result['home'].get('pts')
            if pts is not None:
                home_score = int(pts)
        if away_score is None and game_result.get('away'):
            pts = game_result['away'].get('pts')
            if pts is not None:
                away_score = int(pts)

        # 상태 보정 (점수 기반)
        if home_score is not None and away_score is not None:
            home_result = game_result.get('home', {})
            away_result = game_result.get('away', {})
            home_final = home_result.get('result') if home_result else None
            away_final = away_result.get('result') if away_result else None

            is_live_from_gamefinder = (
                home_result and away_result and
                home_result.get('pts') is not None and
                away_result.get('pts') is not None and
                home_final is None and away_final is None
            )

            if is_live_from_gamefinder:
                game_status = self.GAME_STATUS_LIVE
            elif home_final is not None and away_final is not None:
                game_status = self.GAME_STATUS_FINAL
            elif game_date < et_today:
                game_status = self.GAME_STATUS_FINAL

        return game_status, home_score, away_score

    def _process_game_row(
        self,
        row: pd.Series,
        game_date: date,
        live_scores: Dict[str, Dict[int, int]],
        game_results: Dict[str, Dict],
        team_game_dates: Dict[int, set],
        et_today: date,
        yesterday: date,
    ) -> Dict:
        """
        단일 경기 행 처리.

        Args:
            row: 스케줄 DataFrame의 행
            game_date: 경기 날짜
            live_scores: 라이브 점수
            game_results: LeagueGameFinder 결과
            team_game_dates: 팀별 경기 날짜
            et_today: 오늘 날짜 (ET)
            yesterday: 어제 날짜

        Returns:
            경기 정보 딕셔너리
        """
        game_id = str(row.get("GAME_ID", ""))
        home_team_id = int(row.get("HOME_TEAM_ID", 0))
        away_team_id = int(row.get("VISITOR_TEAM_ID", 0))

        # 경기 시간
        game_time = ""
        if "GAME_STATUS_TEXT" in row:
            game_time = row["GAME_STATUS_TEXT"]
        elif "GAME_DATE_EST" in row:
            game_time = row["GAME_DATE_EST"]

        # 경기 상태 및 점수
        game_status, home_score, away_score = self._determine_game_status(
            row=row,
            game_results=game_results,
            game_date=game_date,
            et_today=et_today,
            live_scores=live_scores,
        )

        # B2B 체크
        home_b2b = yesterday in team_game_dates.get(home_team_id, set())
        away_b2b = yesterday in team_game_dates.get(away_team_id, set())

        # 로깅
        if game_status == self.GAME_STATUS_LIVE:
            logger.info(f"🔴 Live game {game_id}: home={home_score}, away={away_score}")
        elif game_status == self.GAME_STATUS_FINAL and home_score is not None:
            logger.debug(f"✅ Finished game {game_id}: home={home_score}, away={away_score}")

        return {
            "game_id": game_id,
            "game_time": game_time,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "game_status": game_status,
            "home_score": home_score,
            "away_score": away_score,
            "home_b2b": home_b2b,
            "away_b2b": away_b2b,
        }
