"""
BucketsVision 통합 테스트 설정.

Pytest 설정 및 공통 Fixture 정의.
"""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, Optional

import pytest
import pytz

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Pytest 마커 설정
# =============================================================================

def pytest_configure(config):
    """Pytest 마커 등록"""
    config.addinivalue_line("markers", "smoke: 핵심 기능 빠른 검증 테스트")
    config.addinivalue_line("markers", "api: 외부 API 연동 테스트")
    config.addinivalue_line("markers", "slow: 실행 시간이 긴 테스트")
    config.addinivalue_line("markers", "e2e: End-to-End 통합 테스트")
    config.addinivalue_line("markers", "accuracy: 모델 정확도 검증 테스트")
    config.addinivalue_line("markers", "unit: 단위 테스트")


# =============================================================================
# 프로젝트 경로 Fixture
# =============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """프로젝트 루트 경로"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def project_paths(project_root) -> Dict[str, Path]:
    """프로젝트 주요 경로"""
    return {
        "root": project_root,
        "model_dir": project_root / "bucketsvision_v4" / "models",
        "data_dir": project_root / "data",
        "tests_dir": project_root / "tests",
    }


# =============================================================================
# 서비스 Fixture
# =============================================================================

@pytest.fixture(scope="module")
def predictor(project_paths):
    """V5.4 예측 서비스"""
    from app.services.predictor_v5 import V5PredictionService
    return V5PredictionService(project_paths["model_dir"])


@pytest.fixture(scope="module")
def loader(project_paths):
    """데이터 로더"""
    from app.services.data_loader import DataLoader
    return DataLoader(project_paths["data_dir"])


@pytest.fixture(scope="module")
def dnt_client():
    """DNT API 클라이언트"""
    from app.services.dnt_api import DNTApiClient
    return DNTApiClient()


@pytest.fixture(scope="module")
def espn_client():
    """ESPN 부상 클라이언트"""
    from src.data_collection.espn_injury_client import ESPNInjuryClient
    return ESPNInjuryClient()


@pytest.fixture(scope="module")
def nba_client():
    """NBA Stats API 클라이언트"""
    from src.data_collection.nba_stats_client import NBAStatsClient
    return NBAStatsClient()


# =============================================================================
# 날짜/시간 Fixture
# =============================================================================

@pytest.fixture
def et_today() -> date:
    """미국 동부 시간 기준 오늘"""
    et = pytz.timezone('America/New_York')
    return datetime.now(et).date()


@pytest.fixture
def et_yesterday(et_today) -> date:
    """미국 동부 시간 기준 어제"""
    return et_today - timedelta(days=1)


@pytest.fixture
def et_tomorrow(et_today) -> date:
    """미국 동부 시간 기준 내일"""
    return et_today + timedelta(days=1)


@pytest.fixture
def current_season(et_today) -> int:
    """현재 시즌 연도"""
    from src.utils.helpers import get_season_from_date
    return get_season_from_date(et_today)


# =============================================================================
# 데이터 Fixture
# =============================================================================

@pytest.fixture(scope="module")
def team_info():
    """팀 정보 딕셔너리"""
    from config.constants import TEAM_INFO
    return TEAM_INFO


@pytest.fixture(scope="module")
def abbr_to_id():
    """팀 약어 → ID 매핑"""
    from config.constants import ABBR_TO_ID
    return ABBR_TO_ID


@pytest.fixture
def team_epm(loader, et_today) -> Dict[int, Dict]:
    """팀 EPM 데이터"""
    return loader.load_team_epm(et_today)


@pytest.fixture
def sample_game(loader, et_today) -> Optional[Dict]:
    """샘플 경기 데이터"""
    for offset in [0, 1, -1, 2, -2]:
        test_date = et_today + timedelta(days=offset)
        games = loader.get_games(test_date)
        if games:
            return {"date": test_date, "game": games[0]}
    return None


# =============================================================================
# Mock 데이터 Fixture
# =============================================================================

@pytest.fixture
def mock_balanced_features() -> Dict[str, float]:
    """균형잡힌 경기 피처 (모든 값 0)"""
    return {
        'team_epm_diff': 0.0,
        'sos_diff': 0.0,
        'bench_strength_diff': 0.0,
        'top5_epm_diff': 0.0,
        'ft_rate_diff': 0.0,
    }


@pytest.fixture
def mock_strong_home_features() -> Dict[str, float]:
    """강팀 홈 경기 피처"""
    return {
        'team_epm_diff': 8.0,
        'sos_diff': 0.5,
        'bench_strength_diff': 3.0,
        'top5_epm_diff': 4.0,
        'ft_rate_diff': 0.02,
    }


@pytest.fixture
def mock_weak_home_features() -> Dict[str, float]:
    """약팀 홈 경기 피처"""
    return {
        'team_epm_diff': -8.0,
        'sos_diff': -0.5,
        'bench_strength_diff': -3.0,
        'top5_epm_diff': -4.0,
        'ft_rate_diff': -0.02,
    }


# =============================================================================
# V5.4 피처 목록
# =============================================================================

@pytest.fixture(scope="session")
def v54_feature_names() -> list:
    """V5.4 피처 이름 목록"""
    return [
        'team_epm_diff',
        'sos_diff',
        'bench_strength_diff',
        'top5_epm_diff',
        'ft_rate_diff',
    ]


# =============================================================================
# 헬퍼 함수
# =============================================================================

def find_game_with_status(loader, et_today, status: int, max_offset: int = 7) -> Optional[Dict]:
    """
    특정 상태의 경기 찾기.

    Args:
        loader: DataLoader 인스턴스
        et_today: 오늘 날짜
        status: 경기 상태 (1=예정, 2=진행중, 3=종료)
        max_offset: 최대 날짜 오프셋

    Returns:
        경기 딕셔너리 또는 None
    """
    for offset in range(-max_offset, max_offset + 1):
        test_date = et_today + timedelta(days=offset)
        games = loader.get_games(test_date)
        for game in games or []:
            if game.get('game_status') == status:
                return {"date": test_date, "game": game}
    return None


@pytest.fixture
def finished_game(loader, et_today) -> Optional[Dict]:
    """종료된 경기 (status=3)"""
    return find_game_with_status(loader, et_today, status=3)


@pytest.fixture
def scheduled_game(loader, et_today) -> Optional[Dict]:
    """예정된 경기 (status=1)"""
    return find_game_with_status(loader, et_today, status=1)
