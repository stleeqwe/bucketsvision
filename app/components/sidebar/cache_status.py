"""
캐시 상태 사이드바 컴포넌트.

리팩토링 Phase 4: main.py에서 추출.
"""

from typing import Dict

import streamlit as st

from app.utils.date_utils import get_current_time_kst


def render_cache_status(cache_info: Dict) -> None:
    """
    캐시 상태 표시.

    Args:
        cache_info: 캐시 정보 딕셔너리
    """
    # 마지막 갱신 시간 (session_state에서)
    last_refresh = st.session_state.get("last_refresh_time", "앱 시작 시")

    st.markdown("##### 📊 데이터 상태")
    st.markdown(
        f"""
        <div style="
            background: #1a1a2e;
            border-radius: 8px;
            padding: 12px;
            font-size: 0.8rem;
            margin-bottom: 10px;
        ">
            <div style="color: #9ca3af; margin-bottom: 4px;">
                🔄 마지막 갱신
            </div>
            <div style="color: #22c55e; font-weight: bold;">
                {last_refresh}
            </div>
            <div style="color: #6b7280; font-size: 0.7rem; margin-top: 8px;">
                현재: {cache_info.get('current_time_kst', '')}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_refresh_button() -> bool:
    """
    강제 새로고침 버튼 렌더링.

    버튼 클릭 시 session_state에 갱신 시간 기록.

    Returns:
        True if button was clicked
    """
    clicked = st.button(
        "🔄 강제 새로고침",
        help="캐시를 무시하고 최신 데이터를 가져옵니다"
    )

    if clicked:
        st.session_state["last_refresh_time"] = get_current_time_kst()

    return clicked
