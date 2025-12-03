"""
캐시 상태 사이드바 컴포넌트.

리팩토링 Phase 4: main.py에서 추출.
"""

from typing import Dict

import streamlit as st


def render_cache_status(cache_info: Dict) -> None:
    """
    캐시 상태 표시.

    Args:
        cache_info: 캐시 정보 딕셔너리
    """
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
                📅 데이터 기준
            </div>
            <div style="color: #22c55e; font-weight: bold;">
                {cache_info['cache_date']} 05:00 ET
            </div>
            <div style="color: #6b7280; font-size: 0.7rem; margin-top: 8px;">
                다음 갱신: {cache_info['next_refresh_et']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_refresh_button() -> bool:
    """
    강제 새로고침 버튼 렌더링.

    Returns:
        True if button was clicked
    """
    return st.button(
        "🔄 강제 새로고침",
        help="캐시를 무시하고 최신 데이터를 가져옵니다"
    )
