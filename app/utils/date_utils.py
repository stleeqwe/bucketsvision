"""
날짜/시간 유틸리티.

리팩토링 Phase 4: main.py에서 추출.
"""

from datetime import date, datetime, timedelta
from typing import Dict

import pytz


# 타임존 상수
ET = pytz.timezone('America/New_York')
KST = pytz.timezone('Asia/Seoul')

# 요일 한글
WEEKDAYS_KR = ['월', '화', '수', '목', '금', '토', '일']


def get_et_today() -> date:
    """
    미국 동부 시간 기준 오늘 날짜 반환.

    NBA 경기 스케줄은 ET 기준으로 관리됩니다.

    Returns:
        ET 기준 오늘 날짜
    """
    return datetime.now(ET).date()


def get_kst_date(et_date: date) -> date:
    """
    ET 날짜를 KST 날짜로 변환.

    NBA 경기는 미국 동부 저녁 = 한국 다음날 오전.
    예: ET 11/26 경기 → KST 11/27 오전 경기

    Args:
        et_date: ET 기준 날짜

    Returns:
        KST 기준 날짜 (ET + 1일)
    """
    return et_date + timedelta(days=1)


def format_date_kst(game_date: date) -> str:
    """
    경기 날짜를 한국 시간 기준으로 포맷팅.

    Args:
        game_date: ET 기준 경기 날짜

    Returns:
        포맷팅된 문자열 (예: "2025년 12월 04일")
    """
    kst_date = get_kst_date(game_date)
    return kst_date.strftime('%Y년 %m월 %d일')


def get_weekday_kr(d: date) -> str:
    """
    날짜의 한글 요일 반환.

    Args:
        d: 날짜

    Returns:
        한글 요일 (예: "월", "화", ...)
    """
    return WEEKDAYS_KR[d.weekday()]


def get_cache_date_key() -> str:
    """
    ET 오전 5시 기준 캐시 날짜 키 생성.

    - ET 05:00 이전: 전날 날짜 반환
    - ET 05:00 이후: 당일 날짜 반환

    NBA 경기가 새벽에 끝나고 DNT가 업데이트한 후
    첫 호출 시 새 데이터를 가져오기 위함.

    Returns:
        캐시 키 (YYYY-MM-DD 형식)
    """
    now_et = datetime.now(ET)

    # ET 오전 5시 기준
    if now_et.hour < 5:
        cache_date = now_et.date() - timedelta(days=1)
    else:
        cache_date = now_et.date()

    return cache_date.strftime("%Y-%m-%d")


def get_cache_info() -> Dict[str, str]:
    """
    캐시 정보 반환.

    Returns:
        캐시 정보 딕셔너리:
        - cache_date: 캐시 기준 날짜
        - current_time_et: 현재 ET 시간
        - next_refresh_et: 다음 갱신 시간
    """
    now_et = datetime.now(ET)
    now_kst = datetime.now(KST)
    cache_date = get_cache_date_key()

    # 다음 갱신 시간 계산
    if now_et.hour < 5:
        next_refresh = now_et.replace(hour=5, minute=0, second=0, microsecond=0)
    else:
        next_refresh = (now_et + timedelta(days=1)).replace(
            hour=5, minute=0, second=0, microsecond=0
        )

    return {
        "cache_date": cache_date,
        "current_time_et": now_et.strftime("%Y-%m-%d %H:%M ET"),
        "current_time_kst": now_kst.strftime("%H:%M KST"),
        "next_refresh_et": next_refresh.strftime("%Y-%m-%d %H:%M ET"),
    }


def get_current_time_kst() -> str:
    """현재 KST 시간 문자열 반환"""
    return datetime.now(KST).strftime("%H:%M:%S")


def get_season_date_range(season_start: date, et_today: date, max_future_days: int = 7):
    """
    시즌 날짜 범위 계산.

    Args:
        season_start: 시즌 시작일
        et_today: 오늘 날짜 (ET)
        max_future_days: 미래 최대 일수

    Returns:
        (min_date, max_date) 튜플
    """
    min_date = max(season_start, et_today - timedelta(days=60))
    max_date = et_today + timedelta(days=max_future_days)
    return min_date, max_date
