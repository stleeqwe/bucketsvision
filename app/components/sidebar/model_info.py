"""
모델 정보 사이드바 컴포넌트.

리팩토링 Phase 4: main.py에서 추출.
"""

from typing import Dict

import streamlit as st


def render_model_info(model_info: Dict) -> None:
    """
    모델 정보 표시.

    Args:
        model_info: 모델 정보 딕셔너리
    """
    st.subheader("모델 정보")

    st.metric("모델", model_info.get("model_version", "V5.4"))
    st.metric("피처 수", model_info.get("n_features", 5))

    overall_acc = model_info.get("overall_accuracy")
    if overall_acc:
        st.metric("검증 정확도", f"{overall_acc * 100:.1f}%")
