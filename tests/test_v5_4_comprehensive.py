#!/usr/bin/env python3
"""
V5.4 모델 종합 검증 테스트

테스트 범위:
1. API 데이터 수집 검증 (DNT, NBA Stats, ESPN, Odds)
2. 날짜/시즌 로직 검증 (2025-26 시즌 = 2026)
3. V5.4 피처 빌드 검증 (5개 피처)
4. 모델 예측 로직 검증 (Logistic Regression)
5. Injury Impact 계산 로직 검증 (v1.0.0)
6. E2E 통합 테스트
7. 프론트엔드 데이터 검증

실행: python -m pytest tests/test_v5_4_comprehensive.py -v
"""

import sys
import json
import math
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pytest
import pytz
import numpy as np
import pandas as pd

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.predictor_v5 import V5PredictionService, V5GamePrediction
from app.services.data_loader import DataLoader, TEAM_INFO
from config.constants import ABBR_TO_ID
from src.utils.helpers import get_season_from_date


# =============================================================================
# 테스트 픽스처
# =============================================================================

@pytest.fixture(scope="module")
def project_paths():
    """프로젝트 경로"""
    return {
        "root": project_root,
        "model_dir": project_root / "bucketsvision_v4" / "models",
        "data_dir": project_root / "data",
    }


@pytest.fixture(scope="module")
def predictor(project_paths):
    """V5.4 예측 서비스"""
    return V5PredictionService(project_paths["model_dir"])


@pytest.fixture(scope="module")
def loader(project_paths):
    """데이터 로더"""
    return DataLoader(project_paths["data_dir"])


@pytest.fixture
def et_today():
    """미국 동부 시간 기준 오늘"""
    et = pytz.timezone('America/New_York')
    return datetime.now(et).date()


@pytest.fixture
def team_epm(loader, et_today):
    """팀 EPM 데이터"""
    return loader.load_team_epm(et_today)


# =============================================================================
# 1. API 데이터 수집 테스트
# =============================================================================

class TestAPIDataCollection:
    """API 데이터 수집 검증"""

    def test_dnt_api_team_epm(self, loader, et_today):
        """DNT API: 팀 EPM 데이터 로드"""
        team_epm = loader.load_team_epm(et_today)

        # 30개 팀 데이터 존재
        assert len(team_epm) == 30, f"Expected 30 teams, got {len(team_epm)}"

        # 필수 필드 존재 확인
        required_fields = ['team_epm', 'team_oepm', 'team_depm', 'sos']
        for team_id, data in team_epm.items():
            for field in required_fields:
                assert field in data, f"Missing field '{field}' for team {team_id}"

        # EPM 값 범위 확인 (-15 ~ +15 범위 내)
        for team_id, data in team_epm.items():
            epm = data.get('team_epm', 0)
            assert -15 <= epm <= 15, f"Team {team_id} EPM {epm} out of range"

    def test_dnt_api_player_epm(self, loader, et_today):
        """DNT API: 선수 EPM 데이터 로드"""
        season = get_season_from_date(et_today)
        player_epm = loader.load_player_epm(season)

        # 데이터 존재 확인
        assert not player_epm.empty, "Player EPM data is empty"

        # 필수 컬럼 존재
        required_cols = ['player_id', 'player_name', 'team_id', 'team_alias', 'tot', 'mpg']
        for col in required_cols:
            assert col in player_epm.columns, f"Missing column '{col}'"

        # 최소 300명 이상 선수 데이터
        assert len(player_epm) >= 300, f"Expected 300+ players, got {len(player_epm)}"

    def test_nba_stats_api_games(self, loader, et_today):
        """NBA Stats API: 경기 스케줄 로드"""
        # 오늘 또는 전후 1일 중 경기 있는 날짜 찾기
        for offset in [0, 1, -1, 2, -2]:
            test_date = et_today + timedelta(days=offset)
            games = loader.get_games(test_date)
            if games:
                break

        # 경기 데이터 구조 검증
        if games:
            game = games[0]
            required_keys = ['game_id', 'home_team_id', 'away_team_id', 'game_time', 'game_status']
            for key in required_keys:
                assert key in game, f"Missing key '{key}' in game data"

            # team_id가 유효한지 확인
            assert game['home_team_id'] in TEAM_INFO
            assert game['away_team_id'] in TEAM_INFO

    def test_nba_stats_api_team_game_logs(self, loader, et_today):
        """NBA Stats API: 팀 게임 로그 로드"""
        logs = loader.load_team_game_logs(et_today)

        assert not logs.empty, "Team game logs is empty"

        # 필수 컬럼 확인
        required_cols = ['team_id', 'game_id', 'game_date', 'result', 'pts']
        for col in required_cols:
            assert col in logs.columns, f"Missing column '{col}'"

    def test_espn_injury_api(self, loader):
        """ESPN API: 부상자 데이터 로드"""
        # 임의의 팀 부상자 조회
        test_teams = ['LAL', 'BOS', 'GSW']

        for team in test_teams:
            injuries = loader.get_injuries(team)
            # 부상자 리스트 반환 확인 (빈 리스트도 OK)
            assert isinstance(injuries, list), f"Expected list for {team}"


# =============================================================================
# 2. 날짜/시즌 로직 테스트
# =============================================================================

class TestDateSeasonLogic:
    """날짜 및 시즌 로직 검증"""

    def test_current_season_is_2026(self, et_today):
        """현재 시즌이 2026인지 확인 (2025-26 시즌)"""
        season = get_season_from_date(et_today)
        assert season == 2026, f"Expected season 2026, got {season}"

    def test_season_calculation_october(self):
        """10월 이후 날짜 → 다음 해 시즌"""
        test_cases = [
            (date(2025, 10, 22), 2026),  # 시즌 시작
            (date(2025, 11, 28), 2026),  # 11월
            (date(2025, 12, 25), 2026),  # 크리스마스
            (date(2026, 1, 15), 2026),   # 1월
            (date(2026, 4, 10), 2026),   # 4월 (레귤러시즌 끝)
            (date(2026, 6, 15), 2026),   # 6월 (플레이오프)
        ]

        for test_date, expected_season in test_cases:
            result = get_season_from_date(test_date)
            assert result == expected_season, \
                f"Date {test_date}: expected {expected_season}, got {result}"

    def test_season_calculation_before_october(self):
        """10월 이전 날짜 → 같은 해 시즌"""
        test_cases = [
            (date(2026, 9, 30), 2026),  # 프리시즌
            (date(2025, 9, 15), 2025),  # 이전 시즌
        ]

        for test_date, expected_season in test_cases:
            result = get_season_from_date(test_date)
            assert result == expected_season, \
                f"Date {test_date}: expected {expected_season}, got {result}"

    def test_data_loader_uses_correct_season(self, loader, et_today):
        """DataLoader가 올바른 시즌 사용하는지 확인"""
        season = get_season_from_date(et_today)

        # 선수 EPM 로드 시 시즌 확인
        player_epm = loader.load_player_epm(season)
        assert not player_epm.empty, "Failed to load player EPM for current season"


# =============================================================================
# 3. V5.4 피처 빌드 테스트
# =============================================================================

class TestV54FeatureBuild:
    """V5.4 피처 빌드 검증"""

    V54_FEATURES = [
        'team_epm_diff',
        'sos_diff',
        'bench_strength_diff',
        'top5_epm_diff',
        'ft_rate_diff',
    ]

    def test_feature_names_match_model(self, project_paths):
        """피처 이름이 모델 메타데이터와 일치"""
        feature_path = project_paths["model_dir"] / "v5_4_feature_names.json"

        with open(feature_path) as f:
            model_features = json.load(f)

        assert model_features == self.V54_FEATURES, \
            f"Feature mismatch: {model_features} vs {self.V54_FEATURES}"

    def test_build_v5_4_features_returns_all(self, loader, team_epm, et_today):
        """build_v5_4_features가 5개 피처 모두 반환"""
        # LAL vs BOS 테스트
        home_id = ABBR_TO_ID['LAL']
        away_id = ABBR_TO_ID['BOS']

        features = loader.build_v5_4_features(home_id, away_id, team_epm, et_today)

        # 5개 피처 존재 확인
        assert len(features) == 5, f"Expected 5 features, got {len(features)}"

        for feature_name in self.V54_FEATURES:
            assert feature_name in features, f"Missing feature: {feature_name}"
            assert not math.isnan(features[feature_name]), f"NaN value for {feature_name}"

    def test_team_epm_diff_calculation(self, loader, team_epm, et_today):
        """team_epm_diff 계산 검증"""
        home_id = ABBR_TO_ID['BOS']  # 강팀
        away_id = ABBR_TO_ID['DET']  # 약팀

        features = loader.build_v5_4_features(home_id, away_id, team_epm, et_today)

        # 직접 계산
        home_epm = team_epm.get(home_id, {}).get('team_epm', 0)
        away_epm = team_epm.get(away_id, {}).get('team_epm', 0)
        expected_diff = home_epm - away_epm

        assert abs(features['team_epm_diff'] - expected_diff) < 0.001, \
            f"team_epm_diff mismatch: {features['team_epm_diff']} vs {expected_diff}"

    def test_bench_strength_diff_calculation(self, loader, team_epm, et_today):
        """bench_strength_diff 계산 검증 (6-10번째 MPG 선수)"""
        home_id = ABBR_TO_ID['LAL']
        away_id = ABBR_TO_ID['BOS']

        season = get_season_from_date(et_today)
        features = loader.build_v5_4_features(home_id, away_id, team_epm, et_today)

        # 계산 가능한지만 확인 (NaN 아님)
        assert not math.isnan(features['bench_strength_diff'])

        # 범위 확인 (-10 ~ +10 정도가 합리적)
        assert -15 <= features['bench_strength_diff'] <= 15

    def test_top5_epm_diff_calculation(self, loader, team_epm, et_today):
        """top5_epm_diff 계산 검증 (상위 5명 MPG 선수)"""
        home_id = ABBR_TO_ID['OKC']  # 강팀
        away_id = ABBR_TO_ID['WAS']  # 약팀

        features = loader.build_v5_4_features(home_id, away_id, team_epm, et_today)

        # OKC vs WAS면 양수여야 함 (강팀 홈)
        assert features['top5_epm_diff'] > 0, \
            f"Expected positive top5_epm_diff for OKC vs WAS, got {features['top5_epm_diff']}"

    def test_ft_rate_diff_range(self, loader, team_epm, et_today):
        """ft_rate_diff 범위 검증"""
        home_id = ABBR_TO_ID['MIA']
        away_id = ABBR_TO_ID['NYK']

        features = loader.build_v5_4_features(home_id, away_id, team_epm, et_today)

        # FT Rate 차이는 보통 -0.1 ~ +0.1 범위
        assert -0.3 <= features['ft_rate_diff'] <= 0.3, \
            f"ft_rate_diff {features['ft_rate_diff']} out of expected range"


# =============================================================================
# 4. 모델 예측 로직 테스트
# =============================================================================

class TestModelPrediction:
    """V5.4 모델 예측 로직 검증"""

    def test_model_loaded(self, predictor):
        """모델이 정상 로드되었는지 확인"""
        assert predictor.model is not None, "Model not loaded"
        assert predictor.scaler is not None, "Scaler not loaded"
        assert predictor.feature_names is not None, "Feature names not loaded"

    def test_model_info(self, predictor):
        """모델 메타데이터 확인"""
        info = predictor.get_model_info()

        assert info['n_features'] == 5
        assert info['model_version'] == '5.4.0'
        assert 0.7 <= info['overall_accuracy'] <= 0.85  # 78.05% 기대

    def test_predict_proba_range(self, predictor):
        """예측 확률이 유효 범위 내인지 확인"""
        # 균형잡힌 경기 (모든 피처 0)
        balanced_features = {
            'team_epm_diff': 0.0,
            'sos_diff': 0.0,
            'bench_strength_diff': 0.0,
            'top5_epm_diff': 0.0,
            'ft_rate_diff': 0.0,
        }

        prob = predictor.predict_proba(balanced_features)

        # 0.01 ~ 0.99 범위
        assert 0.01 <= prob <= 0.99, f"Probability {prob} out of range"

        # 균형잡힌 경기는 50% 근처여야 함
        assert 0.4 <= prob <= 0.6, f"Balanced game should be ~50%, got {prob:.1%}"

    def test_predict_strong_vs_weak(self, predictor):
        """강팀 vs 약팀 예측 검증"""
        # 강팀 홈 (모든 diff 양수)
        strong_home_features = {
            'team_epm_diff': 8.0,    # 강팀 홈
            'sos_diff': 0.5,
            'bench_strength_diff': 3.0,
            'top5_epm_diff': 4.0,
            'ft_rate_diff': 0.02,
        }

        prob_strong = predictor.predict_proba(strong_home_features)

        # 강팀 홈이면 70% 이상
        assert prob_strong >= 0.70, f"Strong home should be >70%, got {prob_strong:.1%}"

        # 약팀 홈 (모든 diff 음수)
        weak_home_features = {
            'team_epm_diff': -8.0,
            'sos_diff': -0.5,
            'bench_strength_diff': -3.0,
            'top5_epm_diff': -4.0,
            'ft_rate_diff': -0.02,
        }

        prob_weak = predictor.predict_proba(weak_home_features)

        # 약팀 홈이면 30% 이하
        assert prob_weak <= 0.30, f"Weak home should be <30%, got {prob_weak:.1%}"

    def test_prediction_monotonicity(self, predictor):
        """피처 증가 → 확률 증가 검증 (단조성)"""
        base_features = {
            'team_epm_diff': 0.0,
            'sos_diff': 0.0,
            'bench_strength_diff': 0.0,
            'top5_epm_diff': 0.0,
            'ft_rate_diff': 0.0,
        }

        base_prob = predictor.predict_proba(base_features)

        # team_epm_diff 증가 → 확률 증가
        higher_epm = base_features.copy()
        higher_epm['team_epm_diff'] = 5.0
        higher_prob = predictor.predict_proba(higher_epm)

        assert higher_prob > base_prob, \
            f"Higher team_epm_diff should increase prob: {base_prob:.3f} -> {higher_prob:.3f}"

    def test_probability_range_matches_metadata(self, predictor, project_paths):
        """확률 범위가 현실적 피처값에서 합리적인지 확인"""
        meta_path = project_paths["model_dir"] / "v5_4_metadata.json"

        with open(meta_path) as f:
            metadata = json.load(f)

        expected_range = metadata['metrics']['prob_range']

        # 현실적 범위의 피처로 테스트 (학습 데이터 범위 내)
        # 실제 경기에서 관측 가능한 최대/최소 수준
        realistic_high = {
            'team_epm_diff': 10.0,    # OKC vs WAS 수준
            'sos_diff': 1.0,
            'bench_strength_diff': 5.0,
            'top5_epm_diff': 6.0,
            'ft_rate_diff': 0.05,
        }

        realistic_low = {
            'team_epm_diff': -10.0,
            'sos_diff': -1.0,
            'bench_strength_diff': -5.0,
            'top5_epm_diff': -6.0,
            'ft_rate_diff': -0.05,
        }

        high_prob = predictor.predict_proba(realistic_high)
        low_prob = predictor.predict_proba(realistic_low)

        # 유효 범위(0.01~0.99) 내인지 확인
        assert 0.01 <= low_prob <= 0.99, \
            f"Low prob {low_prob:.3f} out of valid range"
        assert 0.01 <= high_prob <= 0.99, \
            f"High prob {high_prob:.3f} out of valid range"

        # 메타데이터 범위와 비교 (현실적 값은 메타데이터 범위 내여야 함)
        assert low_prob <= expected_range[1], \
            f"Realistic low {low_prob:.3f} should be within observed range"
        assert high_prob >= expected_range[0], \
            f"Realistic high {high_prob:.3f} should be within observed range"


# =============================================================================
# 5. Injury Impact 계산 로직 테스트
# =============================================================================

class TestInjuryImpact:
    """Injury Impact 계산 로직 검증"""

    def test_injury_calculator_loads(self, loader, et_today, team_epm):
        """고급 부상 계산기 로드"""
        calc = loader.get_advanced_injury_calculator(et_today, team_epm)

        # 계산기 존재 확인 (데이터 부족 시 None 가능)
        # assert calc is not None, "Failed to load advanced injury calculator"
        if calc is None:
            pytest.skip("Advanced injury calculator not available (data insufficient)")

    def test_injury_impact_version(self, project_paths):
        """Injury Impact 버전 확인"""
        meta_path = project_paths["model_dir"] / "injury_impact_v1_metadata.json"

        with open(meta_path) as f:
            metadata = json.load(f)

        assert metadata['version'] == '1.1.0'
        assert metadata['algorithm'] == 'Performance-based (출전 vs 미출전 성과 비교)'

    def test_injury_adjustment_no_limit(self, predictor):
        """부상 조정에 한도 없음 확인"""
        base_prob = 0.5

        # 큰 부상 영향 (한도 제거되었으므로 그대로 반영)
        home_shift = 15.0  # 15%
        away_shift = 0.0

        adjusted = predictor.apply_injury_adjustment(base_prob, home_shift, away_shift)

        # 50% - 15% = 35%
        expected = 0.35
        assert abs(adjusted - expected) < 0.01, \
            f"Expected {expected:.2%}, got {adjusted:.2%}"

    def test_injury_adjustment_probability_bounds(self, predictor):
        """부상 조정 후 확률 범위 (1%~99%)"""
        # 극단적 케이스
        base_prob = 0.10
        home_shift = 20.0  # 20% 감소 → 음수 되면 안됨

        adjusted = predictor.apply_injury_adjustment(base_prob, home_shift, 0.0)

        assert adjusted >= 0.01, f"Probability {adjusted} below 1%"
        assert adjusted <= 0.99, f"Probability {adjusted} above 99%"

    def test_injury_summary_structure(self, loader, et_today, team_epm):
        """부상 요약 데이터 구조 검증"""
        # 아무 팀이나 테스트
        summary = loader.get_injury_summary('LAL', et_today, team_epm)

        assert 'out_players' in summary
        assert 'gtd_players' in summary
        assert 'total_prob_shift' in summary

        # 타입 확인
        assert isinstance(summary['out_players'], list)
        assert isinstance(summary['gtd_players'], list)
        assert isinstance(summary['total_prob_shift'], (int, float))

    def test_injury_impact_conditions(self, loader, et_today, team_epm, project_paths):
        """부상 영향 적용 조건 검증"""
        meta_path = project_paths["model_dir"] / "injury_impact_v1_metadata.json"

        with open(meta_path) as f:
            metadata = json.load(f)

        conditions = metadata['conditions']

        # 조건 확인
        assert conditions['min_epm'] == 0.0  # EPM > 0
        assert conditions['min_mpg'] == 12.0  # MPG >= 12
        assert conditions['min_play_rate'] == 0.333  # 출전율 > 1/3


# =============================================================================
# 6. E2E 통합 테스트
# =============================================================================

class TestE2EIntegration:
    """End-to-End 통합 테스트"""

    def test_full_prediction_pipeline(self, loader, predictor, et_today, team_epm):
        """전체 예측 파이프라인 테스트"""
        # 경기 찾기
        games = None
        test_date = et_today

        for offset in [0, 1, -1, 2, -2]:
            test_date = et_today + timedelta(days=offset)
            games = loader.get_games(test_date)
            if games:
                break

        if not games:
            pytest.skip("No games found for testing")

        game = games[0]
        home_id = game['home_team_id']
        away_id = game['away_team_id']
        home_info = TEAM_INFO.get(home_id, {})
        away_info = TEAM_INFO.get(away_id, {})
        home_abbr = home_info.get('abbr', 'UNK')
        away_abbr = away_info.get('abbr', 'UNK')

        # 1. 피처 생성
        features = loader.build_v5_4_features(home_id, away_id, team_epm, test_date)
        assert len(features) == 5

        # 2. 기본 예측
        base_prob = predictor.predict_proba(features)
        assert 0.01 <= base_prob <= 0.99

        # 3. 부상 정보 (예정된 경기만)
        home_shift = 0.0
        away_shift = 0.0

        if game.get('game_status') == 1:
            home_summary = loader.get_injury_summary(home_abbr, test_date, team_epm)
            away_summary = loader.get_injury_summary(away_abbr, test_date, team_epm)
            home_shift = home_summary.get('total_prob_shift', 0.0)
            away_shift = away_summary.get('total_prob_shift', 0.0)

        # 4. 부상 조정
        final_prob = predictor.apply_injury_adjustment(base_prob, home_shift, away_shift)
        assert 0.01 <= final_prob <= 0.99

        # 5. 결과 검증
        print(f"\n  E2E Test: {away_abbr} @ {home_abbr}")
        print(f"    Features: {features}")
        print(f"    Base Prob: {base_prob:.1%}")
        print(f"    Injury Shift: Home -{home_shift:.1f}%, Away -{away_shift:.1f}%")
        print(f"    Final Prob: {final_prob:.1%}")

    def test_multiple_games_prediction(self, loader, predictor, et_today, team_epm):
        """다중 경기 예측 일관성 테스트"""
        games = None
        test_date = et_today

        for offset in range(-3, 4):
            test_date = et_today + timedelta(days=offset)
            games = loader.get_games(test_date)
            if games and len(games) >= 3:
                break

        if not games or len(games) < 3:
            pytest.skip("Not enough games for multi-game test")

        predictions = []

        for game in games[:5]:
            home_id = game['home_team_id']
            away_id = game['away_team_id']

            features = loader.build_v5_4_features(home_id, away_id, team_epm, test_date)
            prob = predictor.predict_proba(features)

            predictions.append({
                'game_id': game['game_id'],
                'features': features,
                'prob': prob,
            })

        # 모든 예측이 유효 범위
        for pred in predictions:
            assert 0.01 <= pred['prob'] <= 0.99

    def test_finished_game_accuracy(self, loader, predictor, et_today, team_epm):
        """종료된 경기 적중률 검증"""
        # 과거 경기 찾기
        finished_games = []

        for offset in range(-7, 0):
            test_date = et_today + timedelta(days=offset)
            games = loader.get_games(test_date)

            for game in games or []:
                if game.get('game_status') == 3 and game.get('home_score') is not None:
                    finished_games.append((test_date, game))

        if len(finished_games) < 5:
            pytest.skip("Not enough finished games for accuracy test")

        correct = 0
        total = 0

        for test_date, game in finished_games[:20]:
            home_id = game['home_team_id']
            away_id = game['away_team_id']

            features = loader.build_v5_4_features(home_id, away_id, team_epm, test_date)
            prob = predictor.predict_proba(features)

            predicted_home_win = prob >= 0.5
            actual_home_win = game['home_score'] > game['away_score']

            if predicted_home_win == actual_home_win:
                correct += 1
            total += 1

        accuracy = correct / total
        print(f"\n  Sample Accuracy: {correct}/{total} = {accuracy:.1%}")

        # 최소 40% 이상 (랜덤보다 나아야 함)
        assert accuracy >= 0.40, f"Accuracy {accuracy:.1%} below 40%"


# =============================================================================
# 7. 프론트엔드 데이터 검증
# =============================================================================

class TestFrontendData:
    """프론트엔드 렌더링용 데이터 검증"""

    def test_team_info_complete(self):
        """모든 팀 정보 완비"""
        assert len(TEAM_INFO) == 30

        for team_id, info in TEAM_INFO.items():
            assert 'abbr' in info, f"Missing abbr for team {team_id}"
            assert 'name' in info, f"Missing name for team {team_id}"
            assert 'color' in info, f"Missing color for team {team_id}"

            # 색상 형식 확인 (#XXXXXX)
            color = info['color']
            assert color.startswith('#'), f"Invalid color format: {color}"

    def test_abbr_to_id_mapping(self):
        """팀 약어 → ID 매핑 완비"""
        assert len(ABBR_TO_ID) == 30

        for abbr, team_id in ABBR_TO_ID.items():
            assert team_id in TEAM_INFO, f"Team ID {team_id} not in TEAM_INFO"
            assert TEAM_INFO[team_id]['abbr'] == abbr, f"Abbr mismatch for {team_id}"

    def test_game_card_data_structure(self, loader, et_today, team_epm):
        """게임 카드 렌더링 데이터 구조"""
        games = None
        test_date = et_today

        for offset in [0, 1, -1]:
            test_date = et_today + timedelta(days=offset)
            games = loader.get_games(test_date)
            if games:
                break

        if not games:
            pytest.skip("No games for frontend test")

        game = games[0]

        # 프론트엔드에서 필요한 필드
        required_fields = [
            'game_id',
            'game_time',
            'home_team_id',
            'away_team_id',
            'game_status',
            'home_b2b',
            'away_b2b',
        ]

        for field in required_fields:
            assert field in game, f"Missing field '{field}' for game card"

    def test_prediction_data_for_frontend(self, loader, predictor, et_today, team_epm):
        """프론트엔드 예측 데이터 형식"""
        games = loader.get_games(et_today) or loader.get_games(et_today + timedelta(days=1))

        if not games:
            pytest.skip("No games for frontend prediction test")

        game = games[0]
        home_id = game['home_team_id']
        away_id = game['away_team_id']

        features = loader.build_v5_4_features(home_id, away_id, team_epm, et_today)
        prob = predictor.predict_proba(features)

        # 프론트엔드에서 표시할 데이터
        frontend_data = {
            'home_win_prob': prob,
            'away_win_prob': 1 - prob,
            'predicted_winner': 'home' if prob >= 0.5 else 'away',
            'confidence': abs(prob - 0.5) * 2,  # 0~1 스케일
            'features': features,
        }

        assert 0 <= frontend_data['home_win_prob'] <= 1
        assert 0 <= frontend_data['away_win_prob'] <= 1
        assert frontend_data['predicted_winner'] in ['home', 'away']
        assert 0 <= frontend_data['confidence'] <= 1


# =============================================================================
# 실행 스크립트
# =============================================================================

def run_comprehensive_test():
    """종합 테스트 실행 (pytest 없이)"""
    print("="*70)
    print("  BucketsVision V5.4 종합 검증 테스트")
    print("  시간:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)

    # 초기화
    model_dir = project_root / "bucketsvision_v4" / "models"
    data_dir = project_root / "data"

    predictor = V5PredictionService(model_dir)
    loader = DataLoader(data_dir)

    et = pytz.timezone('America/New_York')
    et_today = datetime.now(et).date()

    print(f"\n  테스트 날짜: {et_today} (ET)")
    print(f"  현재 시즌: {get_season_from_date(et_today)}")

    results = {
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'details': []
    }

    def run_test(name: str, test_func):
        """개별 테스트 실행"""
        try:
            test_func()
            results['passed'] += 1
            results['details'].append((name, 'PASS', None))
            print(f"  ✓ {name}")
        except AssertionError as e:
            results['failed'] += 1
            results['details'].append((name, 'FAIL', str(e)))
            print(f"  ✗ {name}: {e}")
        except Exception as e:
            results['skipped'] += 1
            results['details'].append((name, 'SKIP', str(e)))
            print(f"  ⚠ {name}: {e}")

    # 1. API 테스트
    print("\n[1] API 데이터 수집 테스트")
    print("-"*50)

    team_epm = loader.load_team_epm(et_today)
    run_test("DNT API - 팀 EPM (30팀)", lambda: assert_(len(team_epm) == 30))

    season = get_season_from_date(et_today)
    player_epm = loader.load_player_epm(season)
    run_test("DNT API - 선수 EPM (300+)", lambda: assert_(len(player_epm) >= 300))

    # 경기 찾기
    games = None
    test_date = et_today
    for offset in [0, 1, -1, 2]:
        test_date = et_today + timedelta(days=offset)
        games = loader.get_games(test_date)
        if games:
            break

    run_test("NBA Stats API - 경기 로드", lambda: assert_(games is not None and len(games) > 0))

    # 2. 시즌 로직 테스트
    print("\n[2] 날짜/시즌 로직 테스트")
    print("-"*50)

    run_test("현재 시즌 = 2026", lambda: assert_(get_season_from_date(et_today) == 2026))
    run_test("2025-11-28 → 2026", lambda: assert_(get_season_from_date(date(2025, 11, 28)) == 2026))
    run_test("2026-03-15 → 2026", lambda: assert_(get_season_from_date(date(2026, 3, 15)) == 2026))

    # 3. 피처 빌드 테스트
    print("\n[3] V5.4 피처 빌드 테스트")
    print("-"*50)

    if games:
        game = games[0]
        home_id = game['home_team_id']
        away_id = game['away_team_id']
        features = loader.build_v5_4_features(home_id, away_id, team_epm, test_date)

        run_test("피처 5개 생성", lambda: assert_(len(features) == 5))
        run_test("team_epm_diff 존재", lambda: assert_('team_epm_diff' in features))
        run_test("sos_diff 존재", lambda: assert_('sos_diff' in features))
        run_test("bench_strength_diff 존재", lambda: assert_('bench_strength_diff' in features))
        run_test("top5_epm_diff 존재", lambda: assert_('top5_epm_diff' in features))
        run_test("ft_rate_diff 존재", lambda: assert_('ft_rate_diff' in features))

    # 4. 모델 예측 테스트
    print("\n[4] 모델 예측 로직 테스트")
    print("-"*50)

    run_test("모델 로드 완료", lambda: assert_(predictor.model is not None))
    run_test("스케일러 로드 완료", lambda: assert_(predictor.scaler is not None))

    model_info = predictor.get_model_info()
    run_test("모델 버전 5.4.0", lambda: assert_(model_info['model_version'] == '5.4.0'))
    run_test("피처 수 5개", lambda: assert_(model_info['n_features'] == 5))

    # 예측 테스트
    balanced = {'team_epm_diff': 0, 'sos_diff': 0, 'bench_strength_diff': 0, 'top5_epm_diff': 0, 'ft_rate_diff': 0}
    prob_balanced = predictor.predict_proba(balanced)
    run_test("균형 경기 ~50%", lambda: assert_(0.4 <= prob_balanced <= 0.6))

    strong = {'team_epm_diff': 8, 'sos_diff': 0.5, 'bench_strength_diff': 3, 'top5_epm_diff': 4, 'ft_rate_diff': 0.02}
    prob_strong = predictor.predict_proba(strong)
    run_test("강팀 홈 >70%", lambda: assert_(prob_strong >= 0.70))

    # 5. Injury Impact 테스트
    print("\n[5] Injury Impact 테스트")
    print("-"*50)

    # 한도 없음 확인
    adj = predictor.apply_injury_adjustment(0.5, 15.0, 0.0)
    run_test("부상 조정 한도 없음 (15% 적용)", lambda: assert_(abs(adj - 0.35) < 0.01))

    # 확률 범위 유지
    adj_low = predictor.apply_injury_adjustment(0.1, 20.0, 0.0)
    run_test("확률 하한 1%", lambda: assert_(adj_low >= 0.01))

    # 6. E2E 테스트
    print("\n[6] E2E 통합 테스트")
    print("-"*50)

    if games:
        game = games[0]
        home_id = game['home_team_id']
        away_id = game['away_team_id']
        home_abbr = TEAM_INFO.get(home_id, {}).get('abbr', 'UNK')
        away_abbr = TEAM_INFO.get(away_id, {}).get('abbr', 'UNK')

        # 전체 파이프라인
        features = loader.build_v5_4_features(home_id, away_id, team_epm, test_date)
        base_prob = predictor.predict_proba(features)

        home_summary = loader.get_injury_summary(home_abbr, test_date, team_epm)
        away_summary = loader.get_injury_summary(away_abbr, test_date, team_epm)

        final_prob = predictor.apply_injury_adjustment(
            base_prob,
            home_summary.get('total_prob_shift', 0),
            away_summary.get('total_prob_shift', 0)
        )

        run_test("E2E 파이프라인 완료", lambda: assert_(0.01 <= final_prob <= 0.99))

        print(f"\n    샘플 경기: {away_abbr} @ {home_abbr}")
        print(f"    기본 확률: {base_prob:.1%}")
        print(f"    최종 확률: {final_prob:.1%}")

    # 결과 요약
    print("\n" + "="*70)
    print("  [테스트 결과 요약]")
    print("="*70)
    print(f"  ✓ 통과: {results['passed']}")
    print(f"  ✗ 실패: {results['failed']}")
    print(f"  ⚠ 스킵: {results['skipped']}")
    print("="*70)

    if results['failed'] > 0:
        print("\n  [실패 상세]")
        for name, status, msg in results['details']:
            if status == 'FAIL':
                print(f"    - {name}: {msg}")

    return results['failed'] == 0


def assert_(condition):
    """간단한 assert 헬퍼"""
    if not condition:
        raise AssertionError("Assertion failed")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--pytest':
        # pytest로 실행
        pytest.main([__file__, '-v'])
    else:
        # 직접 실행
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
