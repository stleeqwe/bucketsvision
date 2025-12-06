"""
모델 정보 사이드바 컴포넌트.

리팩토링 Phase 4: main.py에서 추출.
"""

from typing import Dict, Optional

import streamlit as st


def render_model_info(model_info: Dict, realtime_accuracy: Optional[Dict] = None) -> None:
    """
    모델 정보 표시.

    Args:
        model_info: 모델 정보 딕셔너리
        realtime_accuracy: 실시간 정확도 데이터 (선택)
    """
    st.subheader("모델 정보")

    st.metric("모델", model_info.get("model_version", "V5.4"))
    st.metric("피처 수", model_info.get("n_features", 5))

    # 캘리브레이션 상태 표시
    if model_info.get("calibration_enabled"):
        factor = model_info.get("calibration_factor", 1.15)
        st.caption(f"📊 캘리브레이션: ×{factor} (활성)")

    # 실시간 정확도가 있으면 표시, 없으면 메타데이터 정확도 표시
    if realtime_accuracy and realtime_accuracy.get("accuracy"):
        acc = realtime_accuracy["accuracy"]
        total = realtime_accuracy.get("total_games", 0)
        st.metric(
            "시즌 정확도",
            f"{acc * 100:.1f}%",
            delta=f"{total}경기",
            delta_color="off"
        )
        # 고신뢰/저신뢰 정확도 표시 (기준: 70%)
        high_acc = realtime_accuracy.get("high_conf_accuracy")
        high_n = realtime_accuracy.get("high_conf_games", 0)
        low_acc = realtime_accuracy.get("low_conf_accuracy")
        low_n = realtime_accuracy.get("low_conf_games", 0)

        if high_acc is not None and low_acc is not None:
            st.caption("신뢰도별 정확도 (기준: 70%)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**≥70%**: {high_acc*100:.0f}%  \n({high_n}경기)")
            with col2:
                st.markdown(f"**<70%**: {low_acc*100:.0f}%  \n({low_n}경기)")
    else:
        # 메타데이터 정확도 (학습 시 검증)
        overall_acc = model_info.get("overall_accuracy")
        if overall_acc:
            st.metric("검증 정확도", f"{overall_acc * 100:.1f}%")
