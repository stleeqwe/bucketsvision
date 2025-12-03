"""
날짜 선택 사이드바 컴포넌트.

리팩토링 Phase 4: main.py에서 추출.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Tuple

import streamlit as st

from app.utils.date_utils import get_kst_date, get_weekday_kr, format_date_kst


@dataclass
class DateSelection:
    """날짜 선택 결과"""
    mode: str           # daily, weekly, monthly, season
    start_date: date
    end_date: date
    selected_date: date  # 현재 선택된 날짜 (네비게이션용)
    header_text: str     # 페이지 헤더 텍스트


def render_date_picker(
    et_today: date,
    season_start: date = date(2025, 10, 22),
    max_past_days: int = 60,
    max_future_days: int = 7,
) -> DateSelection:
    """
    날짜 선택 UI 렌더링.

    Args:
        et_today: 오늘 날짜 (ET)
        season_start: 시즌 시작일
        max_past_days: 과거 최대 일수
        max_future_days: 미래 최대 일수

    Returns:
        DateSelection 결과
    """
    # 날짜 범위 설정
    min_date = max(season_start, et_today - timedelta(days=max_past_days))
    max_date = et_today + timedelta(days=max_future_days)

    # 세션 스테이트 초기화
    if "selected_date" not in st.session_state:
        st.session_state.selected_date = et_today
    if "date_mode" not in st.session_state:
        st.session_state.date_mode = "daily"

    selected_date = st.session_state.selected_date
    date_mode = st.session_state.date_mode

    # 범위 초과 시 보정
    if selected_date < min_date:
        selected_date = min_date
        st.session_state.selected_date = min_date
    elif selected_date > max_date:
        selected_date = max_date
        st.session_state.selected_date = max_date

    # 조회 범위 선택 버튼
    st.markdown("**조회 범위**")
    mode_cols = st.columns(4)
    mode_options = [
        ("daily", "일별"),
        ("weekly", "주간"),
        ("monthly", "월간"),
        ("season", "시즌"),
    ]

    for i, (mode_key, mode_label) in enumerate(mode_options):
        with mode_cols[i]:
            is_selected = date_mode == mode_key
            if st.button(
                mode_label,
                key=f"mode_{mode_key}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                st.session_state.date_mode = mode_key
                st.rerun()

    st.markdown("")

    # 모드별 날짜 선택 UI
    if date_mode == "daily":
        _render_daily_picker(selected_date, min_date, max_date)
    elif date_mode == "weekly":
        _render_weekly_picker(selected_date, min_date, max_date)
    elif date_mode == "monthly":
        _render_monthly_picker(selected_date, min_date, max_date)
    else:  # season
        _render_season_info(season_start)

    # 날짜 범위 계산
    start_date, end_date, header_text = _calculate_date_range(
        date_mode, selected_date, season_start, et_today
    )

    return DateSelection(
        mode=date_mode,
        start_date=start_date,
        end_date=end_date,
        selected_date=selected_date,
        header_text=header_text,
    )


def _render_daily_picker(selected_date: date, min_date: date, max_date: date):
    """일별 날짜 선택 UI"""
    st.markdown("**경기 날짜 선택**")
    col_prev, col_date, col_next = st.columns([1, 2, 1])

    with col_prev:
        if st.button("◀", disabled=(selected_date <= min_date), use_container_width=True):
            st.session_state.selected_date = selected_date - timedelta(days=1)
            st.rerun()

    with col_date:
        kst_date = get_kst_date(selected_date)
        weekday_kr = get_weekday_kr(kst_date)
        date_str = kst_date.strftime(f'%m/%d ({weekday_kr})')
        st.markdown(
            f"<div style='text-align: center; font-size: 1.1rem; padding: 6px 0;'>{date_str}</div>",
            unsafe_allow_html=True
        )

    with col_next:
        if st.button("▶", disabled=(selected_date >= max_date), use_container_width=True):
            st.session_state.selected_date = selected_date + timedelta(days=1)
            st.rerun()


def _render_weekly_picker(selected_date: date, min_date: date, max_date: date):
    """주간 날짜 선택 UI"""
    week_start = selected_date - timedelta(days=selected_date.weekday())
    week_end = min(week_start + timedelta(days=6), max_date)

    st.markdown("**주간 선택**")
    col_prev, col_date, col_next = st.columns([1, 2, 1])

    with col_prev:
        prev_week = week_start - timedelta(days=7)
        if st.button("◀", disabled=(prev_week < min_date), use_container_width=True, key="week_prev"):
            st.session_state.selected_date = prev_week
            st.rerun()

    with col_date:
        kst_start = get_kst_date(week_start)
        kst_end = get_kst_date(week_end)
        st.markdown(
            f"<div style='text-align: center; font-size: 0.95rem; padding: 6px 0;'>"
            f"{kst_start.strftime('%m/%d')} ~ {kst_end.strftime('%m/%d')}</div>",
            unsafe_allow_html=True
        )

    with col_next:
        next_week = week_start + timedelta(days=7)
        if st.button("▶", disabled=(next_week > max_date), use_container_width=True, key="week_next"):
            st.session_state.selected_date = next_week
            st.rerun()


def _render_monthly_picker(selected_date: date, min_date: date, max_date: date):
    """월간 날짜 선택 UI"""
    month_start = selected_date.replace(day=1)
    next_month = (month_start + timedelta(days=32)).replace(day=1)

    st.markdown("**월간 선택**")
    col_prev, col_date, col_next = st.columns([1, 2, 1])

    with col_prev:
        prev_month = (month_start - timedelta(days=1)).replace(day=1)
        if st.button("◀", disabled=(prev_month < min_date), use_container_width=True, key="month_prev"):
            st.session_state.selected_date = prev_month
            st.rerun()

    with col_date:
        st.markdown(
            f"<div style='text-align: center; font-size: 1.1rem; padding: 6px 0;'>"
            f"{month_start.strftime('%Y년 %m월')}</div>",
            unsafe_allow_html=True
        )

    with col_next:
        if st.button("▶", disabled=(next_month > max_date), use_container_width=True, key="month_next"):
            st.session_state.selected_date = next_month
            st.rerun()


def _render_season_info(season_start: date):
    """시즌 정보 표시"""
    st.markdown(
        f"<div style='text-align: center; color: #9ca3af; font-size: 0.9rem; padding: 10px 0;'>"
        f"2025-26 시즌 전체<br>"
        f"<span style='font-size: 0.75rem;'>{season_start.strftime('%Y.%m.%d')} ~ 현재</span>"
        f"</div>",
        unsafe_allow_html=True
    )


def _calculate_date_range(
    date_mode: str,
    selected_date: date,
    season_start: date,
    et_today: date
) -> Tuple[date, date, str]:
    """
    날짜 범위 및 헤더 텍스트 계산.

    Returns:
        (start_date, end_date, header_text) 튜플
    """
    if date_mode == "daily":
        start_date = selected_date
        end_date = selected_date
        header_text = f"📅 {format_date_kst(selected_date)} 경기 예측"

    elif date_mode == "weekly":
        start_date = selected_date - timedelta(days=selected_date.weekday())
        end_date = min(start_date + timedelta(days=6), et_today)
        kst_start = get_kst_date(start_date)
        kst_end = get_kst_date(end_date)
        header_text = f"📅 주간 예측 ({kst_start.strftime('%m/%d')} ~ {kst_end.strftime('%m/%d')})"

    elif date_mode == "monthly":
        start_date = selected_date.replace(day=1)
        next_month = (start_date + timedelta(days=32)).replace(day=1)
        end_date = min(next_month - timedelta(days=1), et_today)
        header_text = f"📅 {start_date.strftime('%Y년 %m월')} 예측"

    else:  # season
        start_date = season_start
        end_date = et_today
        header_text = "📅 2025-26 시즌 전체 예측"

    return start_date, end_date, header_text
