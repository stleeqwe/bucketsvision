"""
통계 계산기 모듈.

리팩토링 Phase 2.1: data_loader.py에서 통계 계산 로직 분리.
"""

from datetime import date
from typing import Dict, Optional

import pandas as pd


class StatCalculator:
    """
    팀/선수 통계 계산 담당.

    책임:
    - Four Factors 계산 (eFG%, FT Rate, ORB%)
    - 모멘텀 지표 계산 (연승, 승률, EWMA 마진)
    - 휴식일 계산
    """

    # 기본값 상수
    DEFAULT_EFG_PCT = 0.50
    DEFAULT_FT_RATE = 0.20
    DEFAULT_ORB_PCT = 0.25
    DEFAULT_WIN_PCT = 0.50
    DEFAULT_REST_DAYS = 7
    MAX_STREAK = 10

    @staticmethod
    def calc_efg(games: pd.DataFrame) -> float:
        """
        eFG% 계산: (FG + 0.5 * 3P) / FGA

        Args:
            games: 경기 로그 DataFrame (fg, fg3, fga 컬럼 필요)

        Returns:
            eFG% (0.0 ~ 1.0)
        """
        if len(games) == 0:
            return StatCalculator.DEFAULT_EFG_PCT

        fg = games['fg'].sum() if 'fg' in games.columns else 0
        fg3 = games['fg3'].sum() if 'fg3' in games.columns else 0
        fga = games['fga'].sum() if 'fga' in games.columns else 0

        if fga == 0:
            return StatCalculator.DEFAULT_EFG_PCT

        return (fg + 0.5 * fg3) / fga

    @staticmethod
    def calc_ft_rate(games: pd.DataFrame) -> float:
        """
        FT Rate 계산: FTM / FGA

        Args:
            games: 경기 로그 DataFrame (ft, fga 컬럼 필요)

        Returns:
            FT Rate (0.0 ~ 1.0)
        """
        if len(games) == 0:
            return StatCalculator.DEFAULT_FT_RATE

        ft = games['ft'].sum() if 'ft' in games.columns else 0
        fga = games['fga'].sum() if 'fga' in games.columns else 0

        if fga == 0:
            return StatCalculator.DEFAULT_FT_RATE

        return ft / fga

    @staticmethod
    def calc_orb_pct(games: pd.DataFrame) -> float:
        """
        공격 리바운드 비율: ORB / (ORB + DRB)

        Args:
            games: 경기 로그 DataFrame (orb, drb 컬럼 필요)

        Returns:
            ORB% (0.0 ~ 1.0)
        """
        if len(games) == 0:
            return StatCalculator.DEFAULT_ORB_PCT

        orb = games['orb'].sum() if 'orb' in games.columns else 0
        drb = games['drb'].sum() if 'drb' in games.columns else 0
        total = orb + drb

        if total == 0:
            return StatCalculator.DEFAULT_ORB_PCT

        return orb / total

    @staticmethod
    def calc_streak(games: pd.DataFrame) -> int:
        """
        연승/연패 계산 (양수=연승, 음수=연패).

        Args:
            games: 경기 로그 DataFrame (result 컬럼 필요, 최신순 정렬)

        Returns:
            연승/연패 (-10 ~ +10)
        """
        if len(games) == 0:
            return 0

        streak = 0
        first_result = games.iloc[0]['result']

        for _, row in games.iterrows():
            if row['result'] == first_result:
                streak += 1 if first_result == 'W' else -1
            else:
                break

        return min(max(streak, -StatCalculator.MAX_STREAK), StatCalculator.MAX_STREAK)

    @staticmethod
    def calc_ewma_margin(games: pd.DataFrame, span: int = 5, window: int = 10) -> float:
        """
        EWMA 마진 계산.

        Args:
            games: 경기 로그 DataFrame (margin 컬럼 필요)
            span: EWMA span 파라미터
            window: 사용할 경기 수

        Returns:
            EWMA 마진
        """
        if len(games) < 3:
            return 0.0

        margins = games.head(window)['margin']
        if len(margins) == 0:
            return 0.0

        return margins.ewm(span=span, adjust=False).mean().iloc[0]

    @staticmethod
    def calc_last3_win_pct(games: pd.DataFrame) -> float:
        """
        최근 3경기 승률.

        Args:
            games: 경기 로그 DataFrame (result 컬럼 필요, 최신순 정렬)

        Returns:
            승률 (0.0 ~ 1.0)
        """
        if len(games) == 0:
            return StatCalculator.DEFAULT_WIN_PCT

        last3 = games.head(3)
        return (last3['result'] == 'W').mean()

    @staticmethod
    def calc_last5_win_pct(games: pd.DataFrame) -> float:
        """
        최근 5경기 승률.

        Args:
            games: 경기 로그 DataFrame (result 컬럼 필요, 최신순 정렬)

        Returns:
            승률 (0.0 ~ 1.0)
        """
        if len(games) == 0:
            return StatCalculator.DEFAULT_WIN_PCT

        last5 = games.head(5)
        return (last5['result'] == 'W').mean()

    @staticmethod
    def calc_rest_days(
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
            return StatCalculator.DEFAULT_REST_DAYS

        team_logs['game_date'] = pd.to_datetime(team_logs['game_date'])
        target_dt = pd.to_datetime(target_date)

        # target_date 이전 경기만
        prev_games = team_logs[team_logs['game_date'] < target_dt]
        if len(prev_games) == 0:
            return StatCalculator.DEFAULT_REST_DAYS  # 시즌 첫 경기

        last_game_date = prev_games['game_date'].max()
        rest_days = (target_dt - last_game_date).days - 1  # 경기일 제외

        # 클리핑 (0~7)
        return min(max(rest_days, 0), StatCalculator.DEFAULT_REST_DAYS)

    @staticmethod
    def calc_away_win_pct(games: pd.DataFrame) -> float:
        """
        원정 경기 승률 계산.

        Args:
            games: 경기 로그 DataFrame (is_away, result 컬럼 필요)

        Returns:
            원정 승률 (0.0 ~ 1.0)
        """
        if games.empty or 'is_away' not in games.columns:
            return 0.45  # 기본 원정 승률

        away_games = games[games['is_away'] == True]
        if len(away_games) == 0:
            return 0.45

        return (away_games['result'] == 'W').mean()

    @staticmethod
    def default_team_stats() -> Dict[str, float]:
        """팀 통계 기본값 반환"""
        return {
            'efg_pct': StatCalculator.DEFAULT_EFG_PCT,
            'ft_rate': StatCalculator.DEFAULT_FT_RATE,
            'last5_win_pct': StatCalculator.DEFAULT_WIN_PCT,
            'streak': 0,
            'margin_ewma': 0.0,
            'orb_avg': 10.0,
            'away_win_pct': 0.45,
        }


class PlayerStatCalculator:
    """
    선수 EPM 기반 통계 계산 담당.

    책임:
    - 로테이션 EPM 계산
    - 벤치 스트렝스 계산
    - 상위 5인 EPM 계산
    """

    # 기본값 상수
    DEFAULT_ROTATION_EPM = 0.0
    DEFAULT_BENCH_EPM = -2.0
    DEFAULT_TOP5_EPM = 0.0
    MIN_ROTATION_MPG = 12.0

    @staticmethod
    def get_team_players(player_epm: pd.DataFrame, team_id: int) -> pd.DataFrame:
        """
        팀의 선수 EPM 데이터 조회.

        Args:
            player_epm: 전체 선수 EPM DataFrame
            team_id: 팀 ID

        Returns:
            해당 팀 선수들의 DataFrame
        """
        if player_epm.empty:
            return pd.DataFrame()
        return player_epm[player_epm['team_id'] == team_id]

    @staticmethod
    def calc_rotation_epm(
        player_epm: pd.DataFrame,
        team_id: int,
        min_mpg: float = 12.0
    ) -> float:
        """
        로테이션 선수(MPG >= min_mpg)의 가중 평균 EPM.

        공식: Σ(EPM_i × MPG_i) / Σ(MPG_i)

        Args:
            player_epm: 전체 선수 EPM DataFrame
            team_id: 팀 ID
            min_mpg: 최소 MPG 기준

        Returns:
            가중 평균 EPM
        """
        players = PlayerStatCalculator.get_team_players(player_epm, team_id)
        if len(players) == 0:
            return PlayerStatCalculator.DEFAULT_ROTATION_EPM

        rotation = players[players['mpg'] >= min_mpg]
        if len(rotation) == 0 or rotation['mpg'].sum() == 0:
            return PlayerStatCalculator.DEFAULT_ROTATION_EPM

        weighted_epm = (rotation['tot'] * rotation['mpg']).sum() / rotation['mpg'].sum()
        return weighted_epm

    @staticmethod
    def calc_bench_strength(player_epm: pd.DataFrame, team_id: int) -> float:
        """
        벤치 선수(6-10번째 MPG)의 평균 EPM.

        Args:
            player_epm: 전체 선수 EPM DataFrame
            team_id: 팀 ID

        Returns:
            벤치 평균 EPM
        """
        players = PlayerStatCalculator.get_team_players(player_epm, team_id)
        if len(players) < 6:
            return PlayerStatCalculator.DEFAULT_BENCH_EPM

        sorted_players = players.nlargest(10, 'mpg')
        bench = sorted_players.iloc[5:10] if len(sorted_players) >= 10 else sorted_players.iloc[5:]

        if len(bench) == 0:
            return PlayerStatCalculator.DEFAULT_BENCH_EPM

        return bench['tot'].mean()

    @staticmethod
    def calc_top5_epm(player_epm: pd.DataFrame, team_id: int) -> float:
        """
        상위 5명 선수(MPG 기준)의 평균 EPM.

        Args:
            player_epm: 전체 선수 EPM DataFrame
            team_id: 팀 ID

        Returns:
            상위 5인 평균 EPM
        """
        players = PlayerStatCalculator.get_team_players(player_epm, team_id)
        if len(players) < 5:
            return PlayerStatCalculator.DEFAULT_TOP5_EPM

        return players.nlargest(5, 'mpg')['tot'].mean()
