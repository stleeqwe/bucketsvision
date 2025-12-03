"""
Streamlit 관련 유틸리티.

리팩토링 Phase 4: main.py에서 추출.
"""

from pathlib import Path

import streamlit as st


# 프로젝트 루트 (이 파일 기준 3단계 상위)
PROJECT_ROOT = Path(__file__).parent.parent.parent


@st.cache_resource
def get_prediction_service():
    """
    V5.4 예측 서비스 로드 (캐시).

    Streamlit의 cache_resource를 사용하여
    앱 재실행 시에도 모델을 재로드하지 않음.

    Returns:
        V5PredictionService 인스턴스
    """
    from app.services.predictor_v5 import V5PredictionService

    model_dir = PROJECT_ROOT / "bucketsvision_v4" / "models"
    return V5PredictionService(model_dir)


@st.cache_resource
def get_data_loader(_cache_key: str):
    """
    데이터 로더 (ET 5AM 기준 일일 캐시).

    Args:
        _cache_key: 캐시 키 (언더스코어 prefix로 해싱 제외)
                   ET 5AM에 키가 변경되어 새 인스턴스 생성

    Returns:
        DataLoader 인스턴스
    """
    from app.services.data_loader import DataLoader

    data_dir = PROJECT_ROOT / "data"
    return DataLoader(data_dir)


@st.cache_resource
def get_prediction_pipeline(_cache_key: str):
    """
    예측 파이프라인 로드 (캐시).

    Args:
        _cache_key: 캐시 키

    Returns:
        PredictionPipeline 인스턴스
    """
    from app.services.prediction_pipeline import PredictionPipeline

    data_dir = PROJECT_ROOT / "data"
    model_dir = PROJECT_ROOT / "bucketsvision_v4" / "models"
    return PredictionPipeline(data_dir, model_dir)


def clear_all_caches():
    """모든 Streamlit 캐시 클리어"""
    st.cache_data.clear()
    st.cache_resource.clear()


def get_project_root() -> Path:
    """프로젝트 루트 경로 반환"""
    return PROJECT_ROOT
