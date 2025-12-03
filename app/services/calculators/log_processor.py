"""
로그 처리 유틸리티.

리팩토링 Phase 1: 중복 코드 통합.
"""

from datetime import date
from typing import Optional

import pandas as pd


class LogProcessor:
    """팀/선수 로그 처리 유틸리티"""

    @staticmethod
    def safe_diff(h_val, a_val, default: float = 0.0) -> float:
        """
        안전한 차이 계산 (None 처리).

        Args:
            h_val: 홈팀 값
            a_val: 원정팀 값
            default: 기본값

        Returns:
            h_val - a_val
        """
        h = h_val if h_val is not None else default
        a = a_val if a_val is not None else default
        return float(h - a)

    @staticmethod
    def filter_team_logs(logs: pd.DataFrame, team_id: int,
                        before_date: date) -> pd.DataFrame:
        """
        팀 로그 필터링 (날짜 이전, 정렬).

        Args:
            logs: 전체 팀 로그
            team_id: 팀 ID
            before_date: 기준 날짜 (이전만 포함)

        Returns:
            필터링된 로그 (최신순 정렬)
        """
        if logs.empty:
            return pd.DataFrame()

        team_logs = logs[logs['team_id'] == team_id].copy()
        if team_logs.empty:
            return pd.DataFrame()

        team_logs['game_date'] = pd.to_datetime(team_logs['game_date'])
        team_logs = team_logs[team_logs['game_date'] < pd.to_datetime(before_date)]

        return team_logs.sort_values('game_date', ascending=False)

    @staticmethod
    def get_recent_games(logs: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """
        최근 경기 추출.

        Args:
            logs: 팀 로그 (정렬된 상태)
            window: 추출할 경기 수

        Returns:
            최근 N경기
        """
        if logs.empty:
            return pd.DataFrame()
        return logs.head(window)

    @staticmethod
    def safe_get(d: dict, key: str, default: float = 0.0) -> float:
        """
        딕셔너리에서 안전하게 값 추출.

        Args:
            d: 딕셔너리
            key: 키
            default: 기본값

        Returns:
            값 또는 기본값
        """
        val = d.get(key)
        return val if val is not None else default
