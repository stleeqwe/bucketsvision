"""
🏀 BucketsVision - NBA 승부 예측 서비스

Streamlit 메인 엔트리포인트
"""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta

import pytz
import streamlit as st
from scipy.stats import norm

# V4.4 B2B 보정 상수
B2B_WEIGHT = 3.0  # B2B 마진 보정 가중치 (3점)

def apply_b2b_correction(base_prob: float, home_b2b: bool, away_b2b: bool) -> float:
    """
    B2B 보정 적용.

    Args:
        base_prob: V4.3 기본 예측 확률
        home_b2b: 홈팀 B2B 여부
        away_b2b: 원정팀 B2B 여부

    Returns:
        B2B 보정된 확률
    """
    # b2b_simple: 원정팀 B2B면 +1 (홈팀 유리), 홈팀 B2B면 -1 (홈팀 불리)
    b2b_simple = (1 if away_b2b else 0) - (1 if home_b2b else 0)

    if b2b_simple == 0:
        return base_prob

    # 마진 보정을 확률로 변환
    b2b_margin = b2b_simple * B2B_WEIGHT
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

from app.services.predictor_v4 import V4PredictionService
from app.services.data_loader import DataLoader, TEAM_INFO
from app.components.game_card_v2 import render_game_card, render_no_games, render_day_summary, inject_card_styles
from app.components.team_roster import get_team_options, render_team_roster_page
import pandas as pd

# 페이지 설정
st.set_page_config(
    page_title="BucketsVision",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 다크 테마 스타일
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #ff6b35, #f7931e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 30px;
    }
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 40px;
    }
    .metric-card {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def get_prediction_service():
    """V4.3 예측 서비스 로드 (캐시)"""
    model_dir = project_root / "bucketsvision_v4" / "models"
    return V4PredictionService(model_dir, version="4.3")


def get_data_loader():
    """데이터 로더 (캐시 제거 - 매번 새로 생성)"""
    data_dir = project_root / "data"
    return DataLoader(data_dir)




def main():
    """메인 함수"""
    # 카드 CSS 스타일 주입
    inject_card_styles()

    # 헤더
    st.markdown('<div class="main-header">🏀 BucketsVision</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI 기반 NBA 승부 예측 | V4.4 Logistic + Player EPM + B2B</div>', unsafe_allow_html=True)

    # 사이드바
    with st.sidebar:
        st.header("메뉴")

        # 페이지 모드 선택
        if "page_mode" not in st.session_state:
            st.session_state.page_mode = "predictions"

        page_mode = st.radio(
            "페이지 선택",
            options=["predictions", "team_roster"],
            format_func=lambda x: "🏀 경기 예측" if x == "predictions" else "👥 팀 로스터",
            key="page_mode_radio",
            horizontal=True,
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

        # 경기 예측
        for game in games:
            game_status = game.get("game_status", 1)

            home_id = game["home_team_id"]
            away_id = game["away_team_id"]

            home_info = TEAM_INFO.get(home_id, {})
            away_info = TEAM_INFO.get(away_id, {})

            home_abbr = home_info.get("abbr", "UNK")
            away_abbr = away_info.get("abbr", "UNK")

            # V4.3 피처 생성 (13개 = V4.2 11개 + 선수 EPM 2개)
            features = loader.build_v4_3_features(home_id, away_id, team_epm, game_date)

            # V4.3 기본 예측 (직접 확률 출력)
            base_prob = predictor.predict_proba(features)

            # B2B 정보
            home_b2b = game.get("home_b2b", False)
            away_b2b = game.get("away_b2b", False)

            # V4.4: B2B 보정 적용
            home_win_prob = apply_b2b_correction(base_prob, home_b2b, away_b2b)

            # 마진 근사값 (확률 -> 마진 역변환, UI 표시용)
            # 가비지 타임 압축: 75% 이상(또는 25% 이하)에서 0.85배 적용
            raw_margin = norm.ppf(home_win_prob) * 12.0
            if abs(home_win_prob - 0.5) > 0.25:  # 75% 이상 또는 25% 이하
                predicted_margin = raw_margin * 0.85
            else:
                predicted_margin = raw_margin

            # 경기 상태 및 점수
            game_status = game.get("game_status", 1)
            home_score = game.get("home_score")
            away_score = game.get("away_score")

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

            # 카드 렌더링 (커스텀 분석 포함)
            game_id = game.get("game_id", f"{home_abbr}_{away_abbr}")
            enable_custom = game_status == 1 and odds_info is not None

            render_game_card(
                home_team=home_abbr,
                away_team=away_abbr,
                home_name=home_info.get("name", "Unknown"),
                away_name=away_info.get("name", "Unknown"),
                home_color=home_info.get("color", "#666"),
                away_color=away_info.get("color", "#666"),
                game_time=game["game_time"],
                predicted_margin=round(predicted_margin, 1),
                home_win_prob=home_win_prob,
                game_status=game_status,
                home_score=home_score,
                away_score=away_score,
                home_b2b=home_b2b,
                away_b2b=away_b2b,
                hide_result=is_live_game,  # 라이브 경기는 적중 여부 숨김
                odds_info=odds_info,
                game_id=game_id,
                enable_custom_input=enable_custom,
            )

        # 일별 요약 (다중 날짜 모드에서도 각 날짜별로)
        if day_finished > 0 and date_mode != "daily":
            accuracy = day_correct / day_finished * 100
            mae = day_error / day_finished
            st.caption(f"📊 {day_finished}경기 중 {day_correct}경기 적중 ({accuracy:.1f}%) | MAE: {mae:.1f}pt")
            st.markdown("---")

    # 전체 적중률 요약
    if date_mode == "daily":
        # 일별 모드: 기존 방식
        if grand_total_finished > 0:
            mae = grand_total_error / grand_total_finished
            render_day_summary(grand_total_finished, grand_total_correct, mae)
    else:
        # 다중 날짜 모드: 전체 통계
        if grand_total_finished > 0:
            accuracy = grand_total_correct / grand_total_finished * 100
            mae = grand_total_error / grand_total_finished
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
                    <div style="font-size: 1rem; color: #94a3b8; margin-bottom: 12px;">
                        📊 전체 예측 성과
                    </div>
                    <div style="display: flex; justify-content: center; gap: 40px;">
                        <div>
                            <div style="font-size: 0.8rem; color: #64748b;">적중률</div>
                            <div style="font-size: 2.2rem; font-weight: 800; color: {'#22c55e' if accuracy >= 50 else '#ef4444'};">
                                {accuracy:.1f}%
                            </div>
                        </div>
                        <div>
                            <div style="font-size: 0.8rem; color: #64748b;">평균 오차</div>
                            <div style="font-size: 2.2rem; font-weight: 800; color: #9ca3af;">
                                {mae:.1f}pt
                            </div>
                        </div>
                    </div>
                    <div style="font-size: 0.9rem; color: #64748b; margin-top: 16px;">
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
