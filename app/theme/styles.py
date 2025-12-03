"""
BucketsVision CSS 스타일 및 렌더 함수.

리팩토링 Phase 4: main.py에서 추출.
"""

import streamlit as st

from .colors import COLORS


# 메인 CSS 스타일
MAIN_CSS = f"""
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
"""


def inject_main_styles() -> None:
    """메인 CSS 스타일 주입"""
    st.markdown(MAIN_CSS, unsafe_allow_html=True)


def inject_all_styles() -> None:
    """모든 CSS 스타일 주입 (메인 + 게임카드)"""
    inject_main_styles()

    # 게임 카드 스타일도 주입
    from app.components.game_card_v2 import inject_card_styles
    inject_card_styles()


def render_header() -> None:
    """메인 헤더 렌더링"""
    st.markdown(
        '<div class="main-header">🏀 BucketsVision</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">AI 기반 NBA 승부 예측 | V5.4 Logistic Regression (78.05%)</div>',
        unsafe_allow_html=True
    )


def render_footer(page_type: str = "predictions") -> None:
    """푸터 렌더링"""
    st.markdown("---")

    if page_type == "paper_betting":
        st.markdown(
            """
            <div style="text-align: center; color: #666; font-size: 0.8rem;">
            ⚠️ Paper Betting은 가상 베팅입니다. 실제 베팅에 사용하지 마세요.<br>
            배당 출처: Pinnacle (The Odds API)
            </div>
            """,
            unsafe_allow_html=True
        )
    elif page_type == "team_roster":
        st.markdown(
            """
            <div style="text-align: center; color: #666; font-size: 0.8rem;">
            데이터 출처: NBA Stats API | 2025-26 시즌
            </div>
            """,
            unsafe_allow_html=True
        )
    else:  # predictions
        st.markdown(
            """
            <div style="text-align: center; color: #666; font-size: 0.8rem;">
            ⚠️ 본 예측은 참고용이며, 베팅 등의 목적으로 사용하지 마세요.<br>
            V5.4 Logistic Regression | 정확도: 78.05% | 학습 데이터: 3,643경기 (22-25 시즌)
            </div>
            """,
            unsafe_allow_html=True
        )
