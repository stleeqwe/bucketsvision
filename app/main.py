"""
🏀 BucketsVision - NBA 승부 예측 서비스

Streamlit 메인 엔트리포인트

V5.2 모델 사용:
- 알고리즘: XGBoost
- 피처: 11개 (EPM 4개 + Four Factors 3개 + 모멘텀 2개 + 피로도 2개)
- B2B, 휴식일: 모델 피처로 통합 (학습에 반영)
- 부상 영향: 후행 지표로 예측 후 조정
"""

import sys
import json
from pathlib import Path
from datetime import date, datetime, timedelta

import pytz
import streamlit as st
from scipy.stats import norm

# V4.4 B2B 보정 상수 (비대칭 적용)
B2B_AWAY_ONLY = 1.5   # 원정팀만 B2B: 홈팀 +1.5점
B2B_HOME_ONLY = -1.0  # 홈팀만 B2B: 홈팀 -1.0점
B2B_BOTH = 0.5        # 둘 다 B2B: 홈팀 +0.5점

# V4.4 부상 보정 상수
MAX_INJURY_SHIFT = 0.10  # 최대 부상 보정 한도 (±10%p)


def apply_injury_correction(
    base_prob: float,
    home_prob_shift: float,
    away_prob_shift: float
) -> float:
    """
    부상 영향력 보정 적용 (V2).

    Args:
        base_prob: 기본 예측 확률 (홈팀 승리)
        home_prob_shift: 홈팀 부상으로 인한 승률 감소 (% 단위, 양수)
        away_prob_shift: 원정팀 부상으로 인한 승률 감소 (% 단위, 양수)

    Returns:
        부상 보정된 확률

    공식:
        - 홈팀 부상 → 홈팀 승률 감소 → base_prob 감소
        - 원정팀 부상 → 원정팀 승률 감소 → base_prob 증가
        - 최종 보정 = (away_shift - home_shift) / 100
    """
    # % 단위를 소수로 변환 (3.0% → 0.03)
    home_shift = max(home_prob_shift, 0) / 100.0
    away_shift = max(away_prob_shift, 0) / 100.0

    # 부상 영향 차이 (양수 = 원정팀이 더 불리 = 홈팀 유리)
    net_shift = away_shift - home_shift

    if net_shift == 0:
        return base_prob

    # 최대 한도 적용
    net_shift = max(min(net_shift, MAX_INJURY_SHIFT), -MAX_INJURY_SHIFT)

    adjusted_prob = min(max(base_prob + net_shift, 0.01), 0.99)

    return adjusted_prob


def apply_b2b_correction(base_prob: float, home_b2b: bool, away_b2b: bool) -> float:
    """
    B2B 보정 적용 (비대칭).

    Args:
        base_prob: V4.3 기본 예측 확률
        home_b2b: 홈팀 B2B 여부
        away_b2b: 원정팀 B2B 여부

    Returns:
        B2B 보정된 확률
    """
    # 비대칭 B2B 마진 계산
    if away_b2b and home_b2b:
        # 둘 다 B2B: 홈팀 +0.5점 (원정 B2B가 더 힘듦)
        b2b_margin = B2B_BOTH
    elif away_b2b:
        # 원정팀만 B2B: 홈팀 +1.5점
        b2b_margin = B2B_AWAY_ONLY
    elif home_b2b:
        # 홈팀만 B2B: 홈팀 -1.0점
        b2b_margin = B2B_HOME_ONLY
    else:
        # 둘 다 아님
        return base_prob

    # 마진 보정을 확률로 변환
    prob_shift = norm.cdf(b2b_margin / 12.0) - 0.5

    # 확률 범위 제한 (0.01 ~ 0.99)
    adjusted_prob = min(max(base_prob + prob_shift, 0.01), 0.99)
    return adjusted_prob


def get_et_today() -> date:
    """미국 동부 시간 기준 오늘 날짜 반환 (NBA 경기 스케줄 조회용)"""
    et = pytz.timezone('America/New_York')
    return datetime.now(et).date()


def get_kst_now() -> datetime:
    """한국 시간 현재 datetime 반환"""
    kst = pytz.timezone('Asia/Seoul')
    return datetime.now(kst)


def format_date_kst(game_date: date) -> str:
    """경기 날짜를 한국 시간 기준으로 표시 (다음날 오전)"""
    # NBA 경기는 미국 동부 저녁 = 한국 다음날 오전
    # 예: 11/26 ET 경기 → 한국 11/27 오전 경기
    kst_date = game_date + timedelta(days=1)
    return kst_date.strftime('%Y년 %m월 %d일')

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.predictor_v5 import V5PredictionService
from app.services.data_loader import DataLoader, TEAM_INFO
from app.components.game_card_v2 import (
    inject_card_styles,
    render_game_card,
    render_day_summary,
    render_no_games
)
from app.theme import COLORS
from app.components.team_roster import get_team_options, render_team_roster_page
import pandas as pd

# 페이지 설정
st.set_page_config(
    page_title="BucketsVision",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 다크 테마 스타일 (COLORS 사용)
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {COLORS['bg_primary']};
    }}
    .main-header {{
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #ff6b35, #f7931e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 30px;
    }}
    .sub-header {{
        text-align: center;
        color: {COLORS['text_secondary']};
        margin-bottom: 40px;
    }}
    .metric-card {{
        background: {COLORS['bg_secondary']};
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }}
    /* 경기 구분선 흰색 */
    hr {{
        border-color: white !important;
        border-top: 1px solid white !important;
        background-color: white !important;
    }}
    [data-testid="stMarkdownContainer"] hr {{
        border-color: white !important;
        border-top: 1px solid white !important;
        background-color: white !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# 게임 카드 CSS 주입
inject_card_styles()


@st.cache_resource
def get_prediction_service():
    """V5.2 예측 서비스 로드 (캐시)"""
    model_dir = project_root / "bucketsvision_v4" / "models"
    return V5PredictionService(model_dir)


def get_data_loader():
    """데이터 로더 (캐시 제거 - 매번 새로 생성)"""
    data_dir = project_root / "data"
    return DataLoader(data_dir)


def load_paper_betting_data():
    """Paper Betting 데이터 로드"""
    bets_file = project_root / "data" / "paper_betting" / "bets.json"
    if bets_file.exists():
        with open(bets_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def render_paper_betting_page():
    """Paper Betting 대시보드 렌더링"""
    st.subheader("💰 Paper Betting Dashboard")

    data = load_paper_betting_data()

    if not data:
        st.warning("Paper Betting 데이터가 없습니다. 스크립트를 먼저 실행해주세요.")
        st.code("python scripts/paper_betting.py", language="bash")
        return

    summary = data.get("summary", {})
    bets = data.get("bets", [])
    metadata = data.get("metadata", {})

    # 요약 통계
    st.markdown("### 📊 Overall Performance")

    total_bets = summary.get("total_bets", 0)
    wins = summary.get("wins", 0)
    losses = summary.get("losses", 0)
    pending = summary.get("pending", 0)
    total_profit = summary.get("total_profit", 0)
    roi = summary.get("roi", 0)

    settled = wins + losses
    win_rate = (wins / settled * 100) if settled > 0 else 0

    # 메트릭 카드
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("총 베팅", f"{total_bets}건")
    with col2:
        st.metric("승률", f"{win_rate:.1f}%" if settled > 0 else "-")
    with col3:
        profit_color = "normal" if total_profit >= 0 else "inverse"
        st.metric("총 수익", f"${total_profit:+,.0f}", delta_color=profit_color)
    with col4:
        st.metric("ROI", f"{roi:+.1f}%")

    # 상세 통계
    st.markdown(f"""
    <div style="
        background: #1a1a2e;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    ">
        <div style="display: flex; justify-content: space-around; text-align: center;">
            <div>
                <div style="color: #22c55e; font-size: 2rem; font-weight: bold;">{wins}</div>
                <div style="color: #888;">승리</div>
            </div>
            <div>
                <div style="color: #ef4444; font-size: 2rem; font-weight: bold;">{losses}</div>
                <div style="color: #888;">패배</div>
            </div>
            <div>
                <div style="color: #f59e0b; font-size: 2rem; font-weight: bold;">{pending}</div>
                <div style="color: #888;">대기중</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 설정 정보
    edge_threshold = metadata.get("edge_threshold", 0.05)
    unit_size = metadata.get("unit_size", 100)
    st.caption(f"⚙️ Edge 기준: ≥{edge_threshold*100:.0f}% | Unit: ${unit_size}")

    st.markdown("---")

    # 베팅 기록
    st.markdown("### 📋 Betting History")

    if not bets:
        st.info("아직 베팅 기록이 없습니다.")
        return

    # 날짜별 그룹핑
    from collections import defaultdict
    daily_bets = defaultdict(list)
    for bet in bets:
        daily_bets[bet['date']].append(bet)

    # 최신순 정렬
    for bet_date in sorted(daily_bets.keys(), reverse=True):
        day_bets = daily_bets[bet_date]

        # 날짜별 소계
        day_profit = sum(b.get('profit', 0) or 0 for b in day_bets if b['status'] == 'settled')
        day_wins = sum(1 for b in day_bets if b.get('result') == 'win')
        day_losses = sum(1 for b in day_bets if b.get('result') == 'loss')
        day_pending = sum(1 for b in day_bets if b['status'] == 'pending')

        # 날짜 헤더
        profit_emoji = "🟢" if day_profit > 0 else ("🔴" if day_profit < 0 else "⚪")
        pending_str = f" | ⏳ {day_pending} pending" if day_pending > 0 else ""

        if day_wins + day_losses > 0:
            st.markdown(f"#### {bet_date} — {day_wins}W-{day_losses}L {profit_emoji} ${day_profit:+,.0f}{pending_str}")
        else:
            st.markdown(f"#### {bet_date}{pending_str}")

        # 개별 베팅
        for bet in day_bets:
            status = bet['status']
            bet_team = bet['bet_team']
            bet_odds = bet['bet_odds']
            edge = bet['bet_edge'] * 100
            home_team = bet['home_team']
            away_team = bet['away_team']

            if status == 'settled':
                result = bet.get('result')
                profit = bet.get('profit', 0)
                home_score = bet.get('home_score', '?')
                away_score = bet.get('away_score', '?')

                if result == 'win':
                    emoji = "✅"
                    profit_str = f"**+${profit:.0f}**"
                    color = "#22c55e"
                else:
                    emoji = "❌"
                    profit_str = f"**-${abs(profit):.0f}**"
                    color = "#ef4444"

                st.markdown(f"""
                <div style="
                    background: #1e293b;
                    border-left: 4px solid {color};
                    padding: 12px 16px;
                    margin: 8px 0;
                    border-radius: 0 8px 8px 0;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            {emoji} <strong>{bet_team}</strong> @{bet_odds:.2f}
                            <span style="color: #64748b; font-size: 0.85rem;">
                                | Edge {edge:.1f}% | {away_team} @ {home_team}
                            </span>
                        </div>
                        <div>
                            <span style="color: #94a3b8;">[{away_score}-{home_score}]</span>
                            <span style="color: {color}; font-weight: bold; margin-left: 10px;">
                                {'+' if profit > 0 else ''}{profit:.0f}
                            </span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Pending
                potential = bet.get('potential_profit', 0)
                st.markdown(f"""
                <div style="
                    background: #1e293b;
                    border-left: 4px solid #f59e0b;
                    padding: 12px 16px;
                    margin: 8px 0;
                    border-radius: 0 8px 8px 0;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            ⏳ <strong>{bet_team}</strong> @{bet_odds:.2f}
                            <span style="color: #64748b; font-size: 0.85rem;">
                                | Edge {edge:.1f}% | {away_team} @ {home_team}
                            </span>
                        </div>
                        <div style="color: #94a3b8;">
                            (potential: +${potential:.0f})
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")




def main():
    """메인 함수"""
    # 헤더
    st.markdown('<div class="main-header">🏀 BucketsVision</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI 기반 NBA 승부 예측 | V4.4 Logistic + Player EPM + B2B</div>', unsafe_allow_html=True)

    # 사이드바
    with st.sidebar:
        st.header("메뉴")

        # 페이지 모드 선택
        if "page_mode" not in st.session_state:
            st.session_state.page_mode = "predictions"

        def format_page_mode(x):
            if x == "predictions":
                return "🏀 경기 예측"
            elif x == "team_roster":
                return "👥 팀 로스터"
            else:
                return "💰 Paper Betting"

        page_mode = st.radio(
            "페이지 선택",
            options=["predictions", "paper_betting", "team_roster"],
            format_func=format_page_mode,
            key="page_mode_radio",
            horizontal=False,
            label_visibility="collapsed"
        )
        st.session_state.page_mode = page_mode

        st.markdown("---")

        # 팀 로스터 모드
        if page_mode == "team_roster":
            st.subheader("팀 선택")
            team_options = get_team_options()
            team_names = [name for name, _ in team_options]
            team_ids = {name: tid for name, tid in team_options}

            selected_team_name = st.selectbox(
                "팀을 선택하세요",
                options=team_names,
                key="team_select",
                label_visibility="collapsed"
            )

            if selected_team_name:
                team_id = team_ids[selected_team_name]
                team_info = TEAM_INFO.get(team_id, {})
                team_color = team_info.get("color", "#666666")

                st.markdown(f"""
                <div style="
                    background-color: {team_color}33;
                    border-left: 4px solid {team_color};
                    padding: 10px 15px;
                    border-radius: 4px;
                    margin: 10px 0;
                ">
                    <strong style="color: white;">{selected_team_name}</strong>
                </div>
                """, unsafe_allow_html=True)

        # 예측 모드 설정
        else:
            st.subheader("설정")

        # 날짜 범위 설정 (예측 모드에서만)
        if page_mode == "predictions":
            et_today = get_et_today()
            season_start = date(2025, 10, 22)  # 2025-26 시즌 시작일
            min_date = max(season_start, et_today - timedelta(days=60))  # 시즌 시작 또는 60일 전
            max_date = et_today + timedelta(days=7)   # 미래 7일

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

            # 일별 모드일 때만 날짜 네비게이션 표시
            if date_mode == "daily":
                st.markdown("**경기 날짜 선택**")
                col_prev, col_date, col_next = st.columns([1, 2, 1])

                with col_prev:
                    if st.button("◀", disabled=(selected_date <= min_date), use_container_width=True):
                        st.session_state.selected_date = selected_date - timedelta(days=1)
                        st.rerun()

                with col_date:
                    # 한국 시간 기준 날짜로 표시 (ET + 1일)
                    kst_date = selected_date + timedelta(days=1)
                    weekdays = ['월', '화', '수', '목', '금', '토', '일']
                    weekday_kr = weekdays[kst_date.weekday()]
                    date_str = kst_date.strftime(f'%m/%d ({weekday_kr})')
                    st.markdown(
                        f"<div style='text-align: center; font-size: 1.1rem; padding: 6px 0;'>{date_str}</div>",
                        unsafe_allow_html=True
                    )

                with col_next:
                    if st.button("▶", disabled=(selected_date >= max_date), use_container_width=True):
                        st.session_state.selected_date = selected_date + timedelta(days=1)
                        st.rerun()

            elif date_mode == "weekly":
                # 주간 선택 (주 단위 이동)
                week_start = selected_date - timedelta(days=selected_date.weekday())  # 월요일
                week_end = min(week_start + timedelta(days=6), max_date)

                st.markdown("**주간 선택**")
                col_prev, col_date, col_next = st.columns([1, 2, 1])

                with col_prev:
                    prev_week = week_start - timedelta(days=7)
                    if st.button("◀", disabled=(prev_week < min_date), use_container_width=True, key="week_prev"):
                        st.session_state.selected_date = prev_week
                        st.rerun()

                with col_date:
                    kst_start = week_start + timedelta(days=1)
                    kst_end = week_end + timedelta(days=1)
                    st.markdown(
                        f"<div style='text-align: center; font-size: 0.95rem; padding: 6px 0;'>{kst_start.strftime('%m/%d')} ~ {kst_end.strftime('%m/%d')}</div>",
                        unsafe_allow_html=True
                    )

                with col_next:
                    next_week = week_start + timedelta(days=7)
                    if st.button("▶", disabled=(next_week > max_date), use_container_width=True, key="week_next"):
                        st.session_state.selected_date = next_week
                        st.rerun()

            elif date_mode == "monthly":
                # 월간 선택
                month_start = selected_date.replace(day=1)
                next_month = (month_start + timedelta(days=32)).replace(day=1)
                month_end = min(next_month - timedelta(days=1), max_date)

                st.markdown("**월간 선택**")
                col_prev, col_date, col_next = st.columns([1, 2, 1])

                with col_prev:
                    prev_month = (month_start - timedelta(days=1)).replace(day=1)
                    if st.button("◀", disabled=(prev_month < min_date), use_container_width=True, key="month_prev"):
                        st.session_state.selected_date = prev_month
                        st.rerun()

                with col_date:
                    st.markdown(
                        f"<div style='text-align: center; font-size: 1.1rem; padding: 6px 0;'>{month_start.strftime('%Y년 %m월')}</div>",
                        unsafe_allow_html=True
                    )

                with col_next:
                    if st.button("▶", disabled=(next_month > max_date), use_container_width=True, key="month_next"):
                        st.session_state.selected_date = next_month
                        st.rerun()

            else:  # season
                st.markdown(
                    f"<div style='text-align: center; color: #9ca3af; font-size: 0.9rem; padding: 10px 0;'>"
                    f"2025-26 시즌 전체<br>"
                    f"<span style='font-size: 0.75rem;'>{season_start.strftime('%Y.%m.%d')} ~ 현재</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            st.markdown("---")

            # 모델 정보
            st.subheader("모델 정보")
            predictor = get_prediction_service()
            model_info = predictor.get_model_info()

            st.metric("모델", "V4.4")
            st.metric("피처 수", 13)
            st.metric("검증 정확도", "76.4%")

            st.markdown("---")

            # 새로고침 버튼
            if st.button("🔄 데이터 새로고침"):
                st.cache_resource.clear()
                st.rerun()

    # 페이지 모드에 따른 콘텐츠 렌더링
    if page_mode == "paper_betting":
        # Paper Betting 페이지
        render_paper_betting_page()

        # 푸터
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #666; font-size: 0.8rem;">
            ⚠️ Paper Betting은 가상 베팅입니다. 실제 베팅에 사용하지 마세요.<br>
            배당 출처: Pinnacle (The Odds API)
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    if page_mode == "team_roster":
        # 팀 로스터 페이지
        team_options = get_team_options()
        team_ids = {name: tid for name, tid in team_options}

        if "team_select" in st.session_state and st.session_state.team_select:
            selected_team_name = st.session_state.team_select
            team_id = team_ids[selected_team_name]
            team_info = TEAM_INFO.get(team_id, {})
            team_color = team_info.get("color", "#666666")

            render_team_roster_page(team_id, selected_team_name, team_color)
        else:
            st.info("왼쪽 사이드바에서 팀을 선택해주세요.")

        # 푸터
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #666; font-size: 0.8rem;">
            데이터 출처: NBA Stats API | 2025-26 시즌
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # 예측 페이지 - 날짜 범위 계산
    if date_mode == "daily":
        start_date = selected_date
        end_date = selected_date
        header_text = f"📅 {format_date_kst(selected_date)} 경기 예측"
    elif date_mode == "weekly":
        start_date = selected_date - timedelta(days=selected_date.weekday())
        end_date = min(start_date + timedelta(days=6), et_today)
        kst_start = start_date + timedelta(days=1)
        kst_end = end_date + timedelta(days=1)
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

    st.subheader(header_text)

    # 서비스 로드
    predictor = get_prediction_service()
    loader = get_data_loader()

    # 팀 EPM 데이터 로드 (최신 날짜 기준)
    with st.spinner("팀 데이터 로딩 중..."):
        team_epm = loader.load_team_epm(et_today)

    if not team_epm:
        st.warning("팀 EPM 데이터를 불러올 수 없습니다.")
        return

    # 날짜 범위의 모든 경기 가져오기
    all_games_by_date = {}
    total_games = 0

    with st.spinner("경기 일정 로딩 중..."):
        current_date = start_date
        while current_date <= end_date:
            games = loader.get_games(current_date)
            if games:
                all_games_by_date[current_date] = games
                total_games += len(games)
            current_date += timedelta(days=1)

    if total_games == 0:
        render_no_games()
        return

    # 전체 통계 (다중 날짜 모드)
    if date_mode != "daily":
        total_finished = sum(
            sum(1 for g in games if g.get("game_status") == 3)
            for games in all_games_by_date.values()
        )
        st.caption(f"총 {total_games}경기 | 종료 {total_finished}경기")

    # 예측 적중 추적 (전체)
    grand_total_finished = 0
    grand_total_correct = 0
    grand_total_error = 0.0

    # 날짜별로 경기 렌더링
    sorted_dates = sorted(all_games_by_date.keys(), reverse=True)  # 최신순

    for game_date in sorted_dates:
        games = all_games_by_date[game_date]

        # 다중 날짜 모드: 날짜 헤더 표시
        if date_mode != "daily":
            kst_game_date = game_date + timedelta(days=1)
            weekdays = ['월', '화', '수', '목', '금', '토', '일']
            weekday_kr = weekdays[kst_game_date.weekday()]
            st.markdown(
                f"### {kst_game_date.strftime('%m월 %d일')} ({weekday_kr}) - {len(games)}경기"
            )

        # 일별 모드: 상태 요약
        if date_mode == "daily":
            live_count = sum(1 for g in games if g.get("game_status") == 2)
            scheduled_count = sum(1 for g in games if g.get("game_status") == 1)
            finished_count = sum(1 for g in games if g.get("game_status") == 3)

            status_parts = []
            if live_count > 0:
                status_parts.append(f"🔴 진행 {live_count}")
            if scheduled_count > 0:
                status_parts.append(f"⏰ 예정 {scheduled_count}")
            if finished_count > 0:
                status_parts.append(f"✅ 종료 {finished_count}")
            if status_parts:
                st.caption(" | ".join(status_parts))

        # 일별 적중 추적
        day_finished = 0
        day_correct = 0
        day_error = 0.0

        # 경기 예측 및 렌더링
        for game in games:
            game_status = game.get("game_status", 1)

            home_id = game["home_team_id"]
            away_id = game["away_team_id"]

            home_info = TEAM_INFO.get(home_id, {})
            away_info = TEAM_INFO.get(away_id, {})

            home_abbr = home_info.get("abbr", "UNK")
            away_abbr = away_info.get("abbr", "UNK")

            # B2B 정보
            home_b2b = game.get("home_b2b", False)
            away_b2b = game.get("away_b2b", False)

            # V5.2 피처 생성 (11개 = EPM 4개 + Four Factors 3개 + 모멘텀 2개 + 피로도 2개)
            # B2B와 휴식일은 모델 피처로 통합
            features = loader.build_v5_2_features(
                home_id, away_id, team_epm, game_date,
                home_b2b=home_b2b, away_b2b=away_b2b
            )

            # V5.2 기본 예측 (XGBoost, B2B/휴식일 포함)
            base_prob = predictor.predict_proba(features)

            # 경기 상태 및 점수
            game_status = game.get("game_status", 1)
            home_score = game.get("home_score")
            away_score = game.get("away_score")

            # V5.2: 부상 영향력 계산 (예정된 경기만, 후행 지표)
            home_injury_summary = None
            away_injury_summary = None
            home_prob_shift = 0.0
            away_prob_shift = 0.0

            if game_status == 1:  # 예정된 경기만 부상 분석
                try:
                    home_injury_summary = loader.get_injury_summary(home_abbr, game_date, team_epm)
                    away_injury_summary = loader.get_injury_summary(away_abbr, game_date, team_epm)
                    home_prob_shift = home_injury_summary.get("total_prob_shift", 0.0)
                    away_prob_shift = away_injury_summary.get("total_prob_shift", 0.0)
                except Exception:
                    pass  # 부상 분석 실패 시 무시

            # V5.2: 부상 보정 적용 (후행 지표)
            home_win_prob = predictor.apply_injury_adjustment(
                base_prob,
                home_prob_shift,
                away_prob_shift
            )

            # 마진 근사값 (확률 -> 마진 역변환, UI 표시용)
            # 가비지 타임 압축: 75% 이상(또는 25% 이하)에서 0.85배 적용
            raw_margin = norm.ppf(home_win_prob) * 12.0
            if abs(home_win_prob - 0.5) > 0.25:  # 75% 이상 또는 25% 이하
                predicted_margin = raw_margin * 0.85
            else:
                predicted_margin = raw_margin

            # 종료된 경기 적중률 및 오차 계산
            if game_status == 3 and home_score is not None and away_score is not None:
                day_finished += 1
                grand_total_finished += 1
                predicted_home_win = home_win_prob >= 0.5
                actual_home_win = home_score > away_score
                actual_margin = home_score - away_score

                # 적중 여부
                if predicted_home_win == actual_home_win:
                    day_correct += 1
                    grand_total_correct += 1

                # 오차 누적 (MAE용)
                error = abs(predicted_margin - actual_margin)
                day_error += error
                grand_total_error += error

            # 라이브 경기(진행 중)는 적중 여부 숨김
            is_live_game = game_status == 2

            # 배당 정보 조회 (예정된 경기만)
            odds_info = None
            if game_status == 1:  # 예정된 경기만 배당 표시
                odds_info = loader.get_game_odds(home_abbr, away_abbr)

            # 게임 카드 렌더링 (V2)
            game_id = game.get("game_id", f"{home_abbr}_{away_abbr}")
            render_game_card(
                home_team=home_abbr,
                away_team=away_abbr,
                home_name=home_info.get("name", "Unknown"),
                away_name=away_info.get("name", "Unknown"),
                home_color=home_info.get("color", COLORS["home"]),
                away_color=away_info.get("color", COLORS["away"]),
                game_time=game["game_time"],
                predicted_margin=round(predicted_margin, 1),
                home_win_prob=home_win_prob,
                game_status=game_status,
                home_score=home_score,
                away_score=away_score,
                home_b2b=home_b2b,
                away_b2b=away_b2b,
                hide_result=is_live_game,
                odds_info=odds_info,
                game_id=game_id,
                enable_custom_input=(game_status == 1),
                home_injury_summary=home_injury_summary,
                away_injury_summary=away_injury_summary,
            )

        # 일별 요약 (다중 날짜 모드에서도 각 날짜별로)
        if day_finished > 0 and date_mode != "daily":
            accuracy = day_correct / day_finished * 100
            mae = day_error / day_finished
            st.caption(f"📊 {day_finished}경기 중 {day_correct}경기 적중 ({accuracy:.1f}%) | MAE: {mae:.1f}pt")
            st.markdown("---")

    # 전체 적중률 요약
    if date_mode == "daily":
        # 일별 모드 (V2)
        if grand_total_finished > 0:
            mae = grand_total_error / grand_total_finished
            render_day_summary(grand_total_finished, grand_total_correct, mae)
    else:
        # 다중 날짜 모드: 전체 통계 (COLORS 적용)
        if grand_total_finished > 0:
            accuracy = grand_total_correct / grand_total_finished * 100
            mae = grand_total_error / grand_total_finished
            acc_color = COLORS['success'] if accuracy >= 50 else COLORS['error']
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 100%);
                    border: 1px solid #2d4a6f;
                    border-radius: 12px;
                    padding: 24px;
                    margin: 20px 0;
                    text-align: center;
                ">
                    <div style="font-size: 1rem; color: {COLORS['text_secondary']}; margin-bottom: 12px;">
                        📊 전체 예측 성과
                    </div>
                    <div style="display: flex; justify-content: center; gap: 40px;">
                        <div>
                            <div style="font-size: 0.8rem; color: {COLORS['text_muted']};">적중률</div>
                            <div style="font-size: 2.2rem; font-weight: 800; color: {acc_color};">
                                {accuracy:.1f}%
                            </div>
                        </div>
                        <div>
                            <div style="font-size: 0.8rem; color: {COLORS['text_muted']};">평균 오차</div>
                            <div style="font-size: 2.2rem; font-weight: 800; color: {COLORS['text_secondary']};">
                                {mae:.1f}pt
                            </div>
                        </div>
                    </div>
                    <div style="font-size: 0.9rem; color: {COLORS['text_muted']}; margin-top: 16px;">
                        {grand_total_finished}경기 중 {grand_total_correct}경기 적중
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # 푸터
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
        ⚠️ 본 예측은 참고용이며, 베팅 등의 목적으로 사용하지 마세요.<br>
        V4.4 Logistic + Player EPM + B2B | 정확도: 76.39% | 학습 데이터: 3,642경기 (22-25 시즌)
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
