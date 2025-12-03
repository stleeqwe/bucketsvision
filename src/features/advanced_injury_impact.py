"""
Advanced Injury Impact Calculator.

기대 대비 성과(Performance) 기반 선수 영향력 계산.

버전: 1.0.0

핵심 공식:
1. 기대 승률 = 0.5 - (상대팀 EPM × 0.03)
   - 상대 EPM +5 → 기대 승률 35%
   - 상대 EPM -5 → 기대 승률 65%

2. 경기 성과(performance) = 실제 결과(1/0) - 기대 승률
   - 강팀(+5) 이김: 1 - 0.35 = +0.65 (기대 이상!)
   - 강팀(+5) 짐: 0 - 0.35 = -0.35 (기대한 결과)
   - 약팀(-5) 이김: 1 - 0.65 = +0.35 (기대한 결과)
   - 약팀(-5) 짐: 0 - 0.65 = -0.65 (기대 이하!)

3. 선수 영향력 = 출전경기 평균 성과 - 미출전경기 평균 성과
   - 양수: 선수가 있을 때 팀이 기대 이상 성과 → 영향력 인정
   - 음수: 선수가 없어도 팀 성과 비슷/더 좋음 → 영향력 없음

4. prob_shift = player_epm × 0.02 × normalized_diff
   - 결장 데이터 없으면 폴백: prob_shift = player_epm × 0.02
   - 한도 없음 (확률 경계 1%~99%만 유지)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
import numpy as np

from src.utils.logger import logger


@dataclass
class PlayerImpactResult:
    """선수 영향력 계산 결과"""
    player_name: str
    team_abbr: str
    epm: float
    mpg: float

    # 출전 경기
    played_games: int
    played_avg_value: float  # 출전 경기 평균 가치

    # 미출전 경기
    missed_games: int
    missed_avg_value: float  # 미출전 경기 평균 가치

    # 영향력
    schedule_diff: float  # 출전 평균 - 미출전 평균
    prob_shift: float  # 승률 변화 (0~1)

    is_valid: bool
    skip_reason: Optional[str] = None


class AdvancedInjuryImpactCalculator:
    """
    고급 부상 영향력 계산기.

    기대 대비 성과(Performance) 기반 선수 영향력 계산.
    """

    # 버전 정보
    VERSION = "1.0.0"
    VERSION_DATE = "2025-12-03"

    # 적용 조건
    MIN_EPM = 0.0  # EPM > 0
    MIN_MPG = 12.0  # 로테이션 선수
    MIN_PLAY_RATE = 1/3  # 출전율 > 1/3
    MIN_MISSED_GAMES = 2  # 미출전 최소 2경기 (폴백 로직 있음)

    # EPM → 기대 승률 변환 계수
    EPM_TO_WIN_PROB = 0.03  # EPM 1점 ≈ 3% 승률 변화

    # 선수 EPM → 승률 변환 계수
    PLAYER_EPM_TO_PROB = 0.02  # 선수 EPM 1점 ≈ 2%p

    # 한도 없음 (확률 경계 1%~99%만 유지)

    @classmethod
    def get_version_info(cls) -> Dict:
        """버전 정보 반환"""
        return {
            "version": cls.VERSION,
            "version_date": cls.VERSION_DATE,
            "algorithm": "Performance-based (출전 vs 미출전 성과 비교)",
            "conditions": {
                "min_epm": cls.MIN_EPM,
                "min_mpg": cls.MIN_MPG,
                "min_play_rate": cls.MIN_PLAY_RATE,
                "min_missed_games": cls.MIN_MISSED_GAMES,
            },
            "coefficients": {
                "epm_to_win_prob": cls.EPM_TO_WIN_PROB,
                "player_epm_to_prob": cls.PLAYER_EPM_TO_PROB,
            },
            "features": [
                "결장 데이터 없으면 EPM 기반 폴백",
                "이적 선수 현재 팀 기준 처리",
                "한도 없음 (확률 경계만 유지)",
            ],
        }

    def __init__(
        self,
        player_epm_df: pd.DataFrame,
        team_game_logs: pd.DataFrame,
        player_game_logs: pd.DataFrame,
        team_epm: Dict[int, Dict]
    ):
        self.player_epm = player_epm_df.copy()
        self.team_game_logs = team_game_logs.copy()
        self.player_game_logs = player_game_logs.copy()
        self.team_epm = team_epm

        # 팀 약어 → EPM 매핑 및 팀 약어 → 팀 ID 매핑
        self.team_alias_to_epm = {}
        self.team_alias_to_id = {}
        for tid, info in team_epm.items():
            team_alias = info.get('team_alias', '')
            team_epm_value = info.get('team_epm', 0)
            if team_alias:
                self.team_alias_to_epm[team_alias] = team_epm_value
                self.team_alias_to_id[team_alias] = tid

        logger.info(f"Team EPM mapping: {len(self.team_alias_to_epm)} teams")

        # 이름 정규화
        self.player_epm['player_name_normalized'] = self.player_epm['player_name'].apply(
            self._normalize_name
        )

        # 캐시
        self._team_games_cache: Dict[int, int] = {}
        self._player_games_cache: Dict[int, set] = {}
        self._team_results_cache: Dict[int, Dict] = {}

        self._build_caches()
        self._validate_data()

        logger.info(
            f"AdvancedInjuryCalculator V3 initialized: "
            f"{len(self.player_epm)} players, "
            f"{len(self.team_game_logs)} team games, "
            f"{len(self.player_game_logs)} player games"
        )

    def _validate_data(self) -> None:
        """데이터 검증"""
        if not self.team_game_logs.empty and 'game_date' in self.team_game_logs.columns:
            dates = pd.to_datetime(self.team_game_logs['game_date'])
            min_date = dates.min()
            max_date = dates.max()
            logger.info(f"Team game logs date range: {min_date.date()} ~ {max_date.date()}")

            season_start = pd.Timestamp('2025-10-01')
            recent_games = (dates >= season_start).sum()
            logger.info(f"25-26 season games: {recent_games} / {len(self.team_game_logs)}")

        if not self.player_game_logs.empty:
            unique_games = self.player_game_logs['GAME_ID'].nunique()
            unique_players = self.player_game_logs['PLAYER_ID'].nunique()
            logger.info(f"Player game logs: {unique_players} players, {unique_games} games")

    def get_data_stats(self) -> Dict:
        """데이터 통계 반환"""
        stats = {
            "player_epm_count": len(self.player_epm),
            "team_game_logs_count": len(self.team_game_logs),
            "player_game_logs_count": len(self.player_game_logs),
            "teams_with_games": len(self._team_games_cache),
            "players_with_games": len(self._player_games_cache),
        }

        if not self.team_game_logs.empty and 'game_date' in self.team_game_logs.columns:
            dates = pd.to_datetime(self.team_game_logs['game_date'])
            stats["game_date_range"] = f"{dates.min().date()} ~ {dates.max().date()}"
            season_start = pd.Timestamp('2025-10-01')
            stats["season_25_26_games"] = int((dates >= season_start).sum())

        return stats

    def _normalize_name(self, name: str) -> str:
        """이름 정규화"""
        if pd.isna(name):
            return ""
        return name.lower().strip().replace(".", "").replace("'", "").replace("-", " ")

    def _build_caches(self) -> None:
        """캐시 구축"""
        if not self.team_game_logs.empty:
            team_counts = self.team_game_logs.groupby('team_id')['game_id'].nunique()
            for team_id, count in team_counts.items():
                self._team_games_cache[team_id] = count

        if not self.team_game_logs.empty:
            for _, row in self.team_game_logs.iterrows():
                team_id = row['team_id']
                game_id = row['game_id']

                if team_id not in self._team_results_cache:
                    self._team_results_cache[team_id] = {}

                matchup = row.get('matchup', '')
                opp_abbr = self._extract_opponent(matchup)
                result = row.get('result', 'L')

                # 기대 대비 성과 계산
                opp_epm = self._get_opponent_epm(opp_abbr)

                # 1. 기대 승률 계산 (상대 EPM 기준)
                #    상대가 강하면(+EPM) 기대 승률 낮음
                #    상대가 약하면(-EPM) 기대 승률 높음
                expected_win_prob = 0.5 - opp_epm * self.EPM_TO_WIN_PROB
                expected_win_prob = max(0.1, min(0.9, expected_win_prob))  # 10%~90% 클리핑

                # 2. 실제 결과
                actual_result = 1.0 if result == 'W' else 0.0

                # 3. 성과 = 실제 - 기대
                performance = actual_result - expected_win_prob

                self._team_results_cache[team_id][game_id] = {
                    'matchup': matchup,
                    'opp_abbr': opp_abbr,
                    'result': result,
                    'opp_epm': opp_epm,
                    'expected_win_prob': expected_win_prob,
                    'performance': performance,  # 기대 대비 성과
                }

        if not self.player_game_logs.empty:
            for player_id, group in self.player_game_logs.groupby('PLAYER_ID'):
                self._player_games_cache[player_id] = set(group['GAME_ID'].unique())

    def _extract_opponent(self, matchup: str) -> str:
        """매치업에서 상대팀 약어 추출"""
        if not matchup:
            return ""
        if '@' in matchup:
            return matchup.split(' @ ')[-1].strip()
        elif 'vs.' in matchup:
            return matchup.split(' vs. ')[-1].strip()
        return ""

    def _get_opponent_epm(self, opp_abbr: str) -> float:
        """상대팀 EPM 조회"""
        return self.team_alias_to_epm.get(opp_abbr, 0.0)

    def find_player(
        self,
        player_name: str,
        team_abbr: Optional[str] = None
    ) -> Optional[pd.Series]:
        """
        선수 찾기 (개선된 매칭 로직).

        매칭 우선순위:
        1. 정확한 이름 매칭 (정규화 후) + 팀 필터
        2. 성(Last Name) 매칭 + 팀 필터
        3. 퍼지 매칭 (유사도 0.8 이상) + 팀 필터
        4. [Fallback] 팀 필터 없이 전체 검색 (이적 선수 대응)

        Args:
            player_name: 선수 이름
            team_abbr: 팀 약어 (선택)

        Returns:
            매칭된 선수 정보 또는 None
        """
        normalized = self._normalize_name(player_name)

        # 팀 필터로 먼저 시도, 실패하면 전체 검색
        result = self._find_player_internal(normalized, player_name, team_abbr)

        # 팀 필터로 못 찾으면 전체 검색 (Fallback)
        if result is None and team_abbr is not None:
            logger.info(
                f"팀 필터({team_abbr})로 선수를 찾지 못함. 전체 검색 시도: {player_name}"
            )
            result = self._find_player_internal(normalized, player_name, None)

            if result is not None:
                actual_team = result.get('team_alias', 'UNK')
                logger.warning(
                    f"⚠️ 팀 불일치: '{player_name}' 검색 팀={team_abbr}, "
                    f"실제 팀={actual_team} (이적 가능성)"
                )

        if result is None:
            logger.warning(f"선수를 찾을 수 없음: {player_name} (팀: {team_abbr})")

        return result

    def _find_player_internal(
        self,
        normalized: str,
        player_name: str,
        team_abbr: Optional[str]
    ) -> Optional[pd.Series]:
        """
        내부 선수 검색 로직.

        Args:
            normalized: 정규화된 선수 이름
            player_name: 원본 선수 이름 (로깅용)
            team_abbr: 팀 약어 (None이면 전체 검색)

        Returns:
            매칭된 선수 정보 또는 None
        """
        # 검색 풀 결정
        if team_abbr:
            search_pool = self.player_epm[self.player_epm['team_alias'] == team_abbr]
        else:
            search_pool = self.player_epm

        if len(search_pool) == 0:
            return None

        # 1단계: 정확한 이름 매칭
        exact_matches = search_pool[
            search_pool['player_name_normalized'] == normalized
        ]

        if len(exact_matches) == 1:
            logger.debug(f"정확한 매칭: {player_name} -> {exact_matches.iloc[0]['player_name']}")
            return exact_matches.iloc[0]
        elif len(exact_matches) > 1:
            logger.warning(
                f"여러 선수가 매칭됨 ({player_name}): "
                f"{[r['player_name'] for _, r in exact_matches.iterrows()]}"
            )
            return exact_matches.iloc[0]

        # 2단계: 성(Last Name) 매칭
        name_parts = normalized.split()
        if len(name_parts) >= 2:
            last_name = name_parts[-1]  # 성 (마지막 단어)

            # 성으로 매칭 (정확히 성이 포함된 경우만)
            last_name_matches = search_pool[
                search_pool['player_name_normalized'].str.contains(
                    rf'\b{last_name}\b', regex=True, na=False
                )
            ]

            if len(last_name_matches) == 1:
                matched = last_name_matches.iloc[0]
                logger.debug(f"성 매칭: {player_name} -> {matched['player_name']}")
                return matched
            elif len(last_name_matches) > 1:
                # 성이 같은 선수가 여러 명이면 이름(first name)으로 추가 필터
                first_name = name_parts[0]
                refined = last_name_matches[
                    last_name_matches['player_name_normalized'].str.contains(first_name, na=False)
                ]
                if len(refined) == 1:
                    matched = refined.iloc[0]
                    logger.debug(f"이름+성 매칭: {player_name} -> {matched['player_name']}")
                    return matched
                elif len(refined) > 1:
                    logger.warning(
                        f"여러 선수가 매칭됨 ({player_name}): "
                        f"{[r['player_name'] for _, r in refined.iterrows()]}"
                    )
                    return refined.iloc[0]

        # 3단계: 퍼지 매칭 (유사도 0.8 이상)
        best_match = None
        best_ratio = 0.0

        for _, player in search_pool.iterrows():
            ratio = SequenceMatcher(
                None, normalized, player['player_name_normalized']
            ).ratio()

            if ratio > best_ratio and ratio > 0.8:
                best_ratio = ratio
                best_match = player

        if best_match is not None:
            logger.debug(
                f"퍼지 매칭: {player_name} -> {best_match['player_name']} "
                f"(유사도: {best_ratio:.2f})"
            )

        return best_match

    def calculate_player_impact(
        self,
        player_name: str,
        team_abbr: str
    ) -> PlayerImpactResult:
        """
        선수의 부상 영향력 계산 (V4).

        V4 로직 (기대 대비 성과):
        1. 기대 승률 = 0.5 - (상대팀 EPM × 0.03)
        2. 성과 = 실제 결과(1/0) - 기대 승률
        3. 선수 영향력 = 출전경기 평균 성과 - 미출전경기 평균 성과
        """
        def invalid_result(reason: str) -> PlayerImpactResult:
            return PlayerImpactResult(
                player_name=player_name,
                team_abbr=team_abbr,
                epm=0.0, mpg=0.0,
                played_games=0, played_avg_value=0.0,
                missed_games=0, missed_avg_value=0.0,
                schedule_diff=0.0, prob_shift=0.0,
                is_valid=False, skip_reason=reason
            )

        # 1. 선수 찾기
        player = self.find_player(player_name, team_abbr)
        if player is None:
            return invalid_result("선수를 찾을 수 없음")

        # 매칭된 선수 정보 확인 (검증용 로깅)
        matched_name = player.get('player_name', 'Unknown')
        actual_team = player.get('team_alias', team_abbr)  # 실제 팀 약어

        if matched_name.lower() != player_name.lower():
            logger.info(
                f"선수 매칭: '{player_name}' -> '{matched_name}' "
                f"(검색 팀: {team_abbr}, 실제 팀: {actual_team})"
            )

        # 팀이 다르면 현재 팀(team_abbr) 기준으로 처리 (이적 선수)
        if actual_team != team_abbr:
            logger.info(
                f"이적 선수 처리: {matched_name} (DNT: {actual_team} → ESPN: {team_abbr})"
            )
            # 현재 팀의 team_id 사용
            current_team_id = self.team_alias_to_id.get(team_abbr)
            if current_team_id is None:
                return invalid_result(f"현재 팀 {team_abbr} ID를 찾을 수 없음")
            team_id = current_team_id
            actual_team = team_abbr  # 현재 팀으로 업데이트
        else:
            team_id = player.get('team_id')

        player_epm = player.get('tot', 0.0)
        player_mpg = player.get('mpg', 0.0)
        player_id = player.get('player_id')

        if pd.isna(player_epm) or pd.isna(player_mpg):
            return invalid_result("EPM/MPG 데이터 없음")

        # 조건 1: EPM > 0
        if player_epm <= self.MIN_EPM:
            return invalid_result(f"EPM 조건 미충족 ({player_epm:.2f})")

        # 조건 2: MPG >= 12
        if player_mpg < self.MIN_MPG:
            return invalid_result(f"MPG 조건 미충족 ({player_mpg:.1f})")

        # 팀 경기 결과 조회
        team_results = self._team_results_cache.get(team_id, {})
        if not team_results:
            return invalid_result("팀 경기 데이터 없음")

        team_total_games = len(team_results)

        # 선수 출전 경기
        player_game_ids = self._player_games_cache.get(player_id, set())
        played_count = len(player_game_ids)

        # 조건 3: 출전율 > 1/3
        play_rate = played_count / team_total_games if team_total_games > 0 else 0
        if play_rate <= self.MIN_PLAY_RATE:
            return invalid_result(f"출전율 조건 미충족 ({play_rate:.1%})")

        # 미출전 경기
        all_game_ids = set(team_results.keys())
        missed_game_ids = all_game_ids - player_game_ids
        missed_count = len(missed_game_ids)

        # 조건 4: 미출전 >= 2경기 → 폴백 로직 적용
        if missed_count < self.MIN_MISSED_GAMES:
            # 폴백: 미출전 데이터 없으면 EPM 기반 기본 계산
            prob_shift = player_epm * self.PLAYER_EPM_TO_PROB
            return PlayerImpactResult(
                player_name=matched_name,
                team_abbr=actual_team,
                epm=player_epm,
                mpg=player_mpg,
                played_games=played_count,
                played_avg_value=0.0,
                missed_games=missed_count,
                missed_avg_value=0.0,
                schedule_diff=0.0,
                prob_shift=prob_shift,
                is_valid=True,
                skip_reason=None
            )

        # 출전 경기 성과 수집
        played_performances = []
        for game_id in player_game_ids:
            if game_id in team_results:
                played_performances.append(team_results[game_id]['performance'])

        # 미출전 경기 성과 수집
        missed_performances = []
        for game_id in missed_game_ids:
            if game_id in team_results:
                missed_performances.append(team_results[game_id]['performance'])

        # 평균 성과 계산
        played_avg = np.mean(played_performances) if played_performances else 0.0
        missed_avg = np.mean(missed_performances) if missed_performances else 0.0

        # 성과 차이: 출전 경기 평균 성과 - 미출전 경기 평균 성과
        # 양수: 선수 출전 시 팀이 기대 이상 성과 → 선수 영향력 있음
        # 음수: 선수 없어도 팀 성과 비슷/더 좋음 → 영향력 없음
        schedule_diff = played_avg - missed_avg

        # 승률 변화 계산
        if schedule_diff > 0:
            # 선수 EPM 기반 승률 변화
            # 성과 차이가 클수록 더 큰 영향력 (정규화: 0.5가 최대)
            normalized_diff = min(schedule_diff / 0.5, 1.0)  # 최대 1.0으로 제한
            prob_shift = player_epm * self.PLAYER_EPM_TO_PROB * normalized_diff
            # 한도 제거 - 계산된 그대로 적용
        else:
            prob_shift = 0.0

        return PlayerImpactResult(
            player_name=matched_name,  # 실제 매칭된 선수 이름
            team_abbr=actual_team,  # 실제 팀 (이적 선수 대응)
            epm=player_epm,
            mpg=player_mpg,
            played_games=len(played_performances),
            played_avg_value=played_avg,  # 평균 성과 (performance)
            missed_games=len(missed_performances),
            missed_avg_value=missed_avg,  # 평균 성과 (performance)
            schedule_diff=schedule_diff,  # 성과 차이
            prob_shift=prob_shift,
            is_valid=True,
            skip_reason=None
        )

    def get_game_injury_impact(
        self,
        home_team: str,
        away_team: str,
        home_out_players: List[str],
        away_out_players: List[str],
        home_gtd_players: Optional[List[str]] = None,
        away_gtd_players: Optional[List[str]] = None,
    ) -> Tuple[float, float, List[PlayerImpactResult]]:
        """
        경기별 부상 영향 계산.

        Returns:
            (home_prob_shift, away_prob_shift, all_player_details)
        """
        home_total_shift = 0.0
        away_total_shift = 0.0
        all_details = []

        for player_name in home_out_players:
            result = self.calculate_player_impact(player_name, home_team)
            all_details.append(result)
            if result.is_valid:
                home_total_shift += result.prob_shift

        for player_name in away_out_players:
            result = self.calculate_player_impact(player_name, away_team)
            all_details.append(result)
            if result.is_valid:
                away_total_shift += result.prob_shift

        if home_gtd_players:
            for player_name in home_gtd_players:
                result = self.calculate_player_impact(player_name, home_team)
                all_details.append(result)

        if away_gtd_players:
            for player_name in away_gtd_players:
                result = self.calculate_player_impact(player_name, away_team)
                all_details.append(result)

        return home_total_shift, away_total_shift, all_details


def create_advanced_injury_calculator(
    player_epm_df: pd.DataFrame,
    team_game_logs: pd.DataFrame,
    player_game_logs: pd.DataFrame,
    team_epm: Dict[int, Dict],
) -> Optional[AdvancedInjuryImpactCalculator]:
    """
    팩토리 함수.

    Args:
        player_epm_df: 선수 EPM DataFrame (DNT API에서 로드)
        team_game_logs: 팀 경기 로그
        player_game_logs: 선수 경기 로그
        team_epm: 팀 EPM 딕셔너리

    Returns:
        AdvancedInjuryImpactCalculator 또는 None
    """
    try:
        if player_epm_df is None or player_epm_df.empty:
            logger.warning("Player EPM data is empty")
            return None

        logger.info(f"Creating advanced injury calculator with {len(player_epm_df)} players")

        return AdvancedInjuryImpactCalculator(
            player_epm_df=player_epm_df,
            team_game_logs=team_game_logs,
            player_game_logs=player_game_logs,
            team_epm=team_epm
        )
    except Exception as e:
        logger.error(f"Error creating advanced injury calculator: {e}")
        return None
