"""
General helper functions and utilities.

프로젝트 전반에서 사용되는 유틸리티 함수들.
"""

import math
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

import pandas as pd
import numpy as np


# ===================
# Date Utilities
# ===================

def parse_date(date_str: str) -> date:
    """
    날짜 문자열 파싱.

    Args:
        date_str: YYYY-MM-DD 형식 문자열

    Returns:
        date 객체
    """
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def format_date(d: Union[date, datetime]) -> str:
    """
    날짜를 YYYY-MM-DD 형식으로 포맷.

    Args:
        d: date 또는 datetime 객체

    Returns:
        형식화된 문자열
    """
    return d.strftime("%Y-%m-%d")


def get_season_from_date(game_date: Union[str, date]) -> int:
    """
    경기 날짜로부터 시즌 연도 추출.

    NBA 시즌은 10월에 시작하므로, 10월 이후는 다음 해 시즌.
    예: 2024-11-15 -> 2025 시즌 (24-25)

    Args:
        game_date: 경기 날짜

    Returns:
        시즌 연도 (종료 연도 기준)
    """
    if isinstance(game_date, str):
        game_date = parse_date(game_date)

    if game_date.month >= 10:
        return game_date.year + 1
    return game_date.year


def get_season_date_range(season: int) -> Tuple[date, date]:
    """
    시즌의 대략적인 날짜 범위 반환.

    Args:
        season: 시즌 연도 (예: 2025 for 24-25 시즌)

    Returns:
        (시즌 시작일, 시즌 종료일) 튜플
    """
    start_date = date(season - 1, 10, 15)  # 대략 10월 중순 시작
    end_date = date(season, 6, 30)  # 플레이오프 포함 6월 말까지
    return start_date, end_date


def days_between(date1: Union[str, date], date2: Union[str, date]) -> int:
    """
    두 날짜 사이의 일수 계산.

    Args:
        date1: 첫 번째 날짜
        date2: 두 번째 날짜

    Returns:
        일수 (date2 - date1)
    """
    if isinstance(date1, str):
        date1 = parse_date(date1)
    if isinstance(date2, str):
        date2 = parse_date(date2)
    return (date2 - date1).days


# ===================
# Distance Calculations
# ===================

def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Haversine 공식을 사용한 두 좌표 간 거리 계산.

    Args:
        lat1, lon1: 첫 번째 좌표 (위도, 경도)
        lat2, lon2: 두 번째 좌표 (위도, 경도)

    Returns:
        거리 (km)
    """
    R = 6371  # 지구 반지름 (km)

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# ===================
# Data Processing
# ===================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    안전한 나눗셈 (0으로 나누기 방지).

    Args:
        numerator: 분자
        denominator: 분모
        default: 0으로 나눌 때 반환값

    Returns:
        나눗셈 결과 또는 기본값
    """
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def clip_value(value: float, min_val: float, max_val: float) -> float:
    """
    값을 지정된 범위로 클리핑.

    Args:
        value: 입력값
        min_val: 최소값
        max_val: 최대값

    Returns:
        클리핑된 값
    """
    return max(min_val, min(max_val, value))


def rolling_mean(
    values: List[float],
    window: int,
    min_periods: int = 1
) -> List[float]:
    """
    롤링 평균 계산.

    Args:
        values: 값 리스트
        window: 윈도우 크기
        min_periods: 최소 필요 데이터 수

    Returns:
        롤링 평균 리스트
    """
    result = []
    for i in range(len(values)):
        start_idx = max(0, i - window + 1)
        window_values = values[start_idx:i + 1]

        if len(window_values) >= min_periods:
            result.append(sum(window_values) / len(window_values))
        else:
            result.append(np.nan)

    return result


def calculate_win_percentage(wins: int, games: int) -> float:
    """
    승률 계산.

    Args:
        wins: 승리 수
        games: 총 경기 수

    Returns:
        승률 (0-1)
    """
    return safe_divide(wins, games, 0.5)


# ===================
# Four Factors Calculations
# ===================

def calculate_efg(fg: int, fg3: int, fga: int) -> float:
    """
    Effective Field Goal Percentage 계산.

    eFG% = (FG + 0.5 * 3P) / FGA

    Args:
        fg: Field Goals Made
        fg3: 3-Point Field Goals Made
        fga: Field Goal Attempts

    Returns:
        eFG%
    """
    return safe_divide(fg + 0.5 * fg3, fga, 0.0)


def calculate_tov_pct(tov: int, fga: int, fta: int) -> float:
    """
    Turnover Percentage 계산.

    TOV% = TOV / (FGA + 0.44 * FTA + TOV)

    Args:
        tov: Turnovers
        fga: Field Goal Attempts
        fta: Free Throw Attempts

    Returns:
        TOV%
    """
    possessions = fga + 0.44 * fta + tov
    return safe_divide(tov, possessions, 0.0)


def calculate_orb_pct(orb: int, opp_drb: int) -> float:
    """
    Offensive Rebound Percentage 계산.

    ORB% = ORB / (ORB + Opp_DRB)

    Args:
        orb: Offensive Rebounds
        opp_drb: Opponent Defensive Rebounds

    Returns:
        ORB%
    """
    return safe_divide(orb, orb + opp_drb, 0.0)


def calculate_ft_rate(ft: int, fga: int) -> float:
    """
    Free Throw Rate 계산.

    FT Rate = FT / FGA

    Args:
        ft: Free Throws Made
        fga: Field Goal Attempts

    Returns:
        FT Rate
    """
    return safe_divide(ft, fga, 0.0)


# ===================
# NBA Season Utilities
# ===================

def get_nba_api_season_string(season: int) -> str:
    """
    NBA Stats API용 시즌 문자열 생성.

    Args:
        season: 시즌 연도 (예: 2025)

    Returns:
        API용 시즌 문자열 (예: "2024-25")
    """
    return f"{season - 1}-{str(season)[2:]}"


# 하위 호환성을 위한 별칭
get_nba_season_string = get_nba_api_season_string


# ===================
# File Utilities
# ===================

def ensure_dir(path: Path) -> Path:
    """
    디렉토리 존재 확인 및 생성.

    Args:
        path: 디렉토리 경로

    Returns:
        생성된 디렉토리 경로
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, path: Path, format: str = "parquet") -> None:
    """
    DataFrame 저장.

    Args:
        df: 저장할 DataFrame
        path: 파일 경로
        format: 저장 형식 ("parquet", "csv", "json")
    """
    ensure_dir(path.parent)

    if format == "parquet":
        df.to_parquet(path, index=False)
    elif format == "csv":
        df.to_csv(path, index=False)
    elif format == "json":
        df.to_json(path, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_dataframe(path: Path, format: str = "parquet") -> pd.DataFrame:
    """
    DataFrame 로드.

    Args:
        path: 파일 경로
        format: 파일 형식

    Returns:
        로드된 DataFrame
    """
    if format == "parquet":
        return pd.read_parquet(path)
    elif format == "csv":
        return pd.read_csv(path)
    elif format == "json":
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported format: {format}")


# ===================
# Statistics Utilities
# ===================

def calculate_zscore(value: float, mean: float, std: float) -> float:
    """
    Z-score 계산.

    Args:
        value: 원본 값
        mean: 평균
        std: 표준편차

    Returns:
        Z-score
    """
    if std == 0:
        return 0.0
    return (value - mean) / std


def normalize_to_range(
    value: float,
    min_val: float,
    max_val: float,
    new_min: float = 0.0,
    new_max: float = 1.0
) -> float:
    """
    값을 새 범위로 정규화.

    Args:
        value: 원본 값
        min_val: 원본 최소값
        max_val: 원본 최대값
        new_min: 새 최소값
        new_max: 새 최대값

    Returns:
        정규화된 값
    """
    if max_val == min_val:
        return (new_min + new_max) / 2

    normalized = (value - min_val) / (max_val - min_val)
    return new_min + normalized * (new_max - new_min)
