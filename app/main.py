"""
🏀 BucketsVision - NBA 승부 예측 서비스

Streamlit 메인 엔트리포인트

V5.4 모델 사용:
- 알고리즘: Logistic Regression (C=0.01)
- 피처: 5개 (team_epm_diff, sos_diff, bench_strength_diff, top5_epm_diff, ft_rate_diff)
- 정확도: 78.05% (저신뢰 71.4%, 고신뢰 87.9%)
- 부상 영향: 후행 지표로 예측 후 조정

리팩토링 Phase 4: UI 모듈화 적용.
"""

import streamlit as st

from app.theme import inject_all_styles, render_header, render_footer
from app.utils.date_utils import get_et_today, get_cache_date_key, get_cache_info
from app.utils.streamlit_utils import (
    get_prediction_service,
    get_data_loader,
    clear_all_caches,
    get_project_root,
)
from app.components.sidebar import (
    render_date_picker,
    render_model_info,
    render_cache_status,
    render_refresh_button,
)
from app.components.team_roster import get_team_options, render_team_roster_page
from app.pages.predictions_page import render_predictions_page
from app.pages.paper_betting_page import render_paper_betting_page
from app.services.data_loader import TEAM_INFO


# 페이지 설정
st.set_page_config(
    page_title="BucketsVision",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 주입
inject_all_styles()


def main():
    """메인 함수"""
    # 헤더
    render_header()

    # 사이드바 & 페이지 모드
    page_mode, date_selection = _render_sidebar()

    # 페이지 라우팅
    if page_mode == "paper_betting":
        render_paper_betting_page(get_project_root())
        render_footer("paper_betting")

    elif page_mode == "team_roster":
        _handle_team_roster_page()
        render_footer("team_roster")

    else:  # predictions
        _handle_predictions_page(date_selection)
        render_footer("predictions")


def _render_sidebar():
    """
    사이드바 렌더링.

    Returns:
        (page_mode, date_selection) 튜플
    """
    date_selection = None

    with st.sidebar:
        st.header("메뉴")

        # 페이지 모드 선택
        page_mode = st.radio(
            "페이지 선택",
            options=["predictions", "paper_betting", "team_roster"],
            format_func=_format_page_mode,
            key="page_mode_radio",
            horizontal=False,
            label_visibility="collapsed"
        )

        st.markdown("---")

        # 팀 로스터 모드
        if page_mode == "team_roster":
            _render_team_roster_sidebar()

        # 예측 모드 설정
        elif page_mode == "predictions":
            st.subheader("설정")

            # 날짜 선택
            et_today = get_et_today()
            date_selection = render_date_picker(et_today)

            st.markdown("---")

            # 모델 정보
            predictor = get_prediction_service()
            render_model_info(predictor.get_model_info())

            st.markdown("---")

            # 캐시 상태
            render_cache_status(get_cache_info())
            if render_refresh_button():
                clear_all_caches()
                st.rerun()

    return page_mode, date_selection


def _render_team_roster_sidebar():
    """팀 로스터 사이드바 렌더링"""
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


def _handle_team_roster_page():
    """팀 로스터 페이지 처리"""
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


def _handle_predictions_page(date_selection):
    """예측 페이지 처리"""
    if date_selection is None:
        st.warning("날짜를 선택해주세요.")
        return

    et_today = get_et_today()
    cache_key = get_cache_date_key()

    # 서비스 로드
    predictor = get_prediction_service()
    loader = get_data_loader(cache_key)

    # 팀 EPM 데이터 로드
    with st.spinner("팀 데이터 로딩 중..."):
        team_epm = loader.load_team_epm(et_today)

    if not team_epm:
        st.warning("팀 EPM 데이터를 불러올 수 없습니다.")
        return

    # 예측 페이지 렌더링
    render_predictions_page(
        loader=loader,
        predictor=predictor,
        date_selection=date_selection,
        team_epm=team_epm,
        et_today=et_today,
    )


def _format_page_mode(x):
    """페이지 모드 포맷팅"""
    modes = {
        "predictions": "🏀 경기 예측",
        "team_roster": "👥 팀 로스터",
        "paper_betting": "💰 Paper Betting",
    }
    return modes.get(x, x)


if __name__ == "__main__":
    main()
