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
from app.components.game_card import render_game_card, render_no_games, render_day_summary

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
    # 헤더
    st.markdown('<div class="main-header">🏀 BucketsVision</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI 기반 NBA 승부 예측 | V4.4 Logistic + Player EPM + B2B</div>', unsafe_allow_html=True)

    # 사이드바
    with st.sidebar:
        st.header("설정")

        # 날짜 선택 (좌우 토글 방식)
        et_today = get_et_today()
        min_date = et_today - timedelta(days=30)  # 과거 30일
        max_date = et_today + timedelta(days=7)   # 미래 7일

        # 세션 스테이트 초기화
        if "selected_date" not in st.session_state:
            st.session_state.selected_date = et_today

        selected_date = st.session_state.selected_date

        # 범위 초과 시 보정
        if selected_date < min_date:
            selected_date = min_date
            st.session_state.selected_date = min_date
        elif selected_date > max_date:
            selected_date = max_date
            st.session_state.selected_date = max_date

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

    # 메인 컨텐츠 (한국 시간 기준 표시)
    st.subheader(f"📅 {format_date_kst(selected_date)} 경기 예측")

    # 서비스 로드
    predictor = get_prediction_service()
    loader = get_data_loader()

    # 팀 EPM 데이터 로드
    with st.spinner("팀 데이터 로딩 중..."):
        team_epm = loader.load_team_epm(selected_date)

    if not team_epm:
        st.warning("팀 EPM 데이터를 불러올 수 없습니다.")
        return

    # 경기 가져오기 (결과 포함)
    with st.spinner("경기 일정 로딩 중..."):
        games = loader.get_games(selected_date)

    if not games:
        render_no_games()
        return

    # 경기는 data_loader.get_games()에서 game_id 순으로 정렬됨

    # 경기 상태별 카운트
    live_count = sum(1 for g in games if g.get("game_status") == 2)
    scheduled_count = sum(1 for g in games if g.get("game_status") == 1)
    finished_count = sum(1 for g in games if g.get("game_status") == 3)

    # 상태 요약 표시
    status_parts = []
    if live_count > 0:
        status_parts.append(f"🔴 진행 {live_count}")
    if scheduled_count > 0:
        status_parts.append(f"⏰ 예정 {scheduled_count}")
    if finished_count > 0:
        status_parts.append(f"✅ 종료 {finished_count}")
    if status_parts:
        st.caption(" | ".join(status_parts))

    # 예측 적중 추적
    total_finished = 0
    total_correct = 0
    total_error = 0.0  # MAE 계산용

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
        features = loader.build_v4_3_features(home_id, away_id, team_epm, selected_date)

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
            total_finished += 1
            predicted_home_win = home_win_prob >= 0.5
            actual_home_win = home_score > away_score
            actual_margin = home_score - away_score

            # 적중 여부
            if predicted_home_win == actual_home_win:
                total_correct += 1

            # 오차 누적 (MAE용)
            total_error += abs(predicted_margin - actual_margin)

        # 라이브 경기(진행 중)는 적중 여부 숨김
        is_live_game = game_status == 2

        # 배당 정보 조회 (예정된 경기만)
        odds_info = None
        if game_status == 1:  # 예정된 경기만 배당 표시
            odds_info = loader.get_game_odds(home_abbr, away_abbr)

        # 카드 렌더링
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
            adjusted_margin=None,
            adjusted_win_prob=None,
            home_injuries=[],
            away_injuries=[],
            home_injury_impact=0.0,
            away_injury_impact=0.0,
            game_status=game_status,
            home_score=home_score,
            away_score=away_score,
            home_b2b=home_b2b,
            away_b2b=away_b2b,
            hide_result=is_live_game,  # 라이브 경기는 적중 여부 숨김
            odds_info=odds_info,
        )

    # 일별 적중률 요약 (종료된 경기가 있을 경우)
    if total_finished > 0:
        mae = total_error / total_finished  # 평균 절대 오차
        render_day_summary(total_finished, total_correct, mae)

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
