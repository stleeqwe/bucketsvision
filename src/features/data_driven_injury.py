"""
Data-Driven Injury Impact Calculator.

On/Off 분석 결과를 기반으로 부상 영향도를 계산합니다.
EPM 기반 추정치 대신 실제 경기 데이터에서 도출된 영향도를 사용합니다.

핵심 기능:
- 선수별 데이터 기반 영향도 조회
- EPM fallback 지원 (데이터 부족 시)
- 복수 부상자 효과 합산
- 신뢰도 기반 가중 평균
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from difflib import SequenceMatcher

import pandas as pd
import numpy as np

from src.features.injury_impact import InjuryImpactCalculator, load_player_epm
from src.utils.logger import logger


@dataclass
class PlayerImpactInfo:
    """선수 영향도 정보"""
    player_id: int
    player_name: str
    team_id: int
    team_abbr: str
    impact: float  # 부재 시 예상 마진 변화 (음수 = 팀에 불리)
    source: str  # "data" or "epm"
    confidence: float  # 0.0 ~ 1.0
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


class DataDrivenInjuryCalculator:
    """
    데이터 기반 부상 영향도 계산기.

    On/Off 분석 결과를 사용하여 선수 부재 시 팀에 미치는
    영향을 데이터 기반으로 계산합니다.
    """

    def __init__(
        self,
        player_impact_df: pd.DataFrame,
        fallback_calculator: Optional[InjuryImpactCalculator] = None,
        use_fallback: bool = True,
        confidence_threshold: float = 0.3
    ):
        """
        Args:
            player_impact_df: On/Off 분석 결과 DataFrame
                필수 컬럼: player_id, player_name, team_id, team_abbr,
                          adjusted_impact, ci_lower, ci_upper, sample_quality
            fallback_calculator: EPM 기반 계산기 (fallback용)
            use_fallback: fallback 사용 여부
            confidence_threshold: 최소 신뢰도 임계값
        """
        self.player_impact_df = player_impact_df.copy()
        self.fallback = fallback_calculator
        self.use_fallback = use_fallback
        self.confidence_threshold = confidence_threshold

        # 빠른 조회를 위한 인덱스 구축
        self._build_index()

        logger.info(f"DataDrivenInjuryCalculator initialized:")
        logger.info(f"  Players with data: {len(self.player_impact_df)}")
        logger.info(f"  Fallback enabled: {use_fallback}")

    def _build_index(self):
        """빠른 조회를 위한 인덱스 구축"""
        # player_id 기반 조회
        self._id_lookup = {}
        for _, row in self.player_impact_df.iterrows():
            self._id_lookup[row["player_id"]] = {
                "player_name": row.get("player_name", ""),
                "team_id": row.get("team_id", 0),
                "team_abbr": row.get("team_abbr", ""),
                "adjusted_impact": row.get("adjusted_impact", 0),
                "ci_lower": row.get("ci_lower"),
                "ci_upper": row.get("ci_upper"),
                "sample_quality": row.get("sample_quality", "low"),
                "is_significant": row.get("is_significant", False),
            }

        # 이름 기반 조회 (fuzzy matching용)
        self._name_lookup = {}
        for _, row in self.player_impact_df.iterrows():
            name = self._normalize_name(row.get("player_name", ""))
            team_abbr = row.get("team_abbr", "")
            key = (name, team_abbr)
            self._name_lookup[key] = row["player_id"]

    def _normalize_name(self, name: str) -> str:
        """이름 정규화"""
        if pd.isna(name):
            return ""
        return name.lower().strip().replace(".", "").replace("'", "").replace("-", " ")

    def _calculate_confidence(self, sample_quality: str, is_significant: bool) -> float:
        """신뢰도 계산"""
        base_confidence = {
            "high": 0.9,
            "medium": 0.6,
            "low": 0.3
        }.get(sample_quality, 0.3)

        # 통계적 유의성 보너스
        if is_significant:
            base_confidence = min(1.0, base_confidence + 0.1)

        return base_confidence

    def get_player_impact_by_id(
        self,
        player_id: int
    ) -> Optional[PlayerImpactInfo]:
        """
        player_id로 영향도 조회.

        Args:
            player_id: 선수 ID

        Returns:
            PlayerImpactInfo 또는 None
        """
        if player_id in self._id_lookup:
            data = self._id_lookup[player_id]
            confidence = self._calculate_confidence(
                data["sample_quality"],
                data["is_significant"]
            )

            # 영향도 부호 변환:
            # On/Off 분석에서 adjusted_impact = margin_on - margin_off
            # 즉, 양수 = 출전 시 더 좋은 성적
            # 부재 시 영향 = -adjusted_impact (양수 = 팀에 불리)
            impact = -data["adjusted_impact"]

            return PlayerImpactInfo(
                player_id=player_id,
                player_name=data["player_name"],
                team_id=data["team_id"],
                team_abbr=data["team_abbr"],
                impact=impact,
                source="data",
                confidence=confidence,
                ci_lower=-data["ci_upper"] if data["ci_upper"] is not None else None,
                ci_upper=-data["ci_lower"] if data["ci_lower"] is not None else None,
            )

        return None

    def get_player_impact_by_name(
        self,
        player_name: str,
        team_abbr: Optional[str] = None
    ) -> Optional[PlayerImpactInfo]:
        """
        선수 이름으로 영향도 조회.

        Args:
            player_name: 선수 이름
            team_abbr: 팀 약어 (선택)

        Returns:
            PlayerImpactInfo 또는 None
        """
        normalized = self._normalize_name(player_name)

        # 정확한 매칭
        if team_abbr:
            key = (normalized, team_abbr)
            if key in self._name_lookup:
                return self.get_player_impact_by_id(self._name_lookup[key])

        # 팀 무관 매칭
        for (name, team), player_id in self._name_lookup.items():
            if name == normalized:
                if team_abbr is None or team == team_abbr:
                    return self.get_player_impact_by_id(player_id)

        # Fuzzy 매칭
        best_match = None
        best_ratio = 0.0

        for (name, team), player_id in self._name_lookup.items():
            if team_abbr and team != team_abbr:
                continue

            ratio = SequenceMatcher(None, normalized, name).ratio()
            if ratio > best_ratio and ratio > 0.7:
                best_ratio = ratio
                best_match = player_id

        if best_match:
            return self.get_player_impact_by_id(best_match)

        return None

    def get_player_impact(
        self,
        player_identifier: Union[int, str],
        team_abbr: Optional[str] = None
    ) -> float:
        """
        선수 부재 시 예상 마진 변화.

        Args:
            player_identifier: 선수 ID 또는 이름
            team_abbr: 팀 약어 (이름으로 조회 시)

        Returns:
            마진 변화 (음수 = 팀에 불리)
        """
        # 데이터 기반 조회 시도
        if isinstance(player_identifier, int):
            info = self.get_player_impact_by_id(player_identifier)
        else:
            info = self.get_player_impact_by_name(player_identifier, team_abbr)

        if info is not None:
            # 신뢰도 임계값 확인
            if info.confidence >= self.confidence_threshold:
                return info.impact

        # Fallback: EPM 기반
        if self.use_fallback and self.fallback is not None:
            if isinstance(player_identifier, str):
                epm_impact = self.fallback.calculate_player_impact(
                    player_identifier,
                    team_abbr or ""
                )
                # EPM 기반 영향도는 양수가 팀에 불리
                return -epm_impact

        return 0.0

    def calculate_game_adjustment(
        self,
        home_team_id: int,
        away_team_id: int,
        home_out_players: List[Union[int, str]],
        away_out_players: List[Union[int, str]],
        home_team_abbr: Optional[str] = None,
        away_team_abbr: Optional[str] = None
    ) -> float:
        """
        경기별 부상 조정값 계산.

        Args:
            home_team_id: 홈팀 ID
            away_team_id: 원정팀 ID
            home_out_players: 홈팀 결장 선수 리스트
            away_out_players: 원정팀 결장 선수 리스트
            home_team_abbr: 홈팀 약어
            away_team_abbr: 원정팀 약어

        Returns:
            조정 마진 (양수 = 홈팀에 유리)
        """
        home_total_impact = 0.0
        away_total_impact = 0.0

        # 홈팀 결장자 영향
        for player in home_out_players:
            impact = self.get_player_impact(player, home_team_abbr)
            home_total_impact += impact

        # 원정팀 결장자 영향
        for player in away_out_players:
            impact = self.get_player_impact(player, away_team_abbr)
            away_total_impact += impact

        # 최종 조정값:
        # home_total_impact < 0 이면 홈팀에 불리 → 마진 감소
        # away_total_impact < 0 이면 원정팀에 불리 → 마진 증가 (홈팀 유리)
        adjustment = home_total_impact - away_total_impact

        return adjustment

    def get_game_adjustment_detail(
        self,
        home_out_players: List[Union[int, str]],
        away_out_players: List[Union[int, str]],
        home_team_abbr: Optional[str] = None,
        away_team_abbr: Optional[str] = None
    ) -> Dict:
        """
        경기별 부상 조정 상세 정보.

        Returns:
            {
                "home_adjustment": 홈팀 조정값,
                "away_adjustment": 원정팀 조정값,
                "total_adjustment": 최종 조정값,
                "home_details": [각 선수별 상세],
                "away_details": [각 선수별 상세]
            }
        """
        home_details = []
        away_details = []

        for player in home_out_players:
            if isinstance(player, int):
                info = self.get_player_impact_by_id(player)
            else:
                info = self.get_player_impact_by_name(player, home_team_abbr)

            if info:
                home_details.append({
                    "player_name": info.player_name,
                    "impact": info.impact,
                    "source": info.source,
                    "confidence": info.confidence,
                })
            else:
                # Fallback 사용
                impact = self.get_player_impact(player, home_team_abbr)
                home_details.append({
                    "player_name": player if isinstance(player, str) else f"ID:{player}",
                    "impact": impact,
                    "source": "fallback" if impact != 0 else "not_found",
                    "confidence": 0.5 if impact != 0 else 0.0,
                })

        for player in away_out_players:
            if isinstance(player, int):
                info = self.get_player_impact_by_id(player)
            else:
                info = self.get_player_impact_by_name(player, away_team_abbr)

            if info:
                away_details.append({
                    "player_name": info.player_name,
                    "impact": info.impact,
                    "source": info.source,
                    "confidence": info.confidence,
                })
            else:
                impact = self.get_player_impact(player, away_team_abbr)
                away_details.append({
                    "player_name": player if isinstance(player, str) else f"ID:{player}",
                    "impact": impact,
                    "source": "fallback" if impact != 0 else "not_found",
                    "confidence": 0.5 if impact != 0 else 0.0,
                })

        home_adj = sum(d["impact"] for d in home_details)
        away_adj = sum(d["impact"] for d in away_details)

        return {
            "home_adjustment": home_adj,
            "away_adjustment": away_adj,
            "total_adjustment": home_adj - away_adj,
            "home_details": home_details,
            "away_details": away_details,
        }

    def get_top_impact_players(self, n: int = 20) -> pd.DataFrame:
        """영향도 상위 선수 조회"""
        df = self.player_impact_df.copy()
        # 부재 시 영향이 큰 순서 (adjusted_impact가 큰 = 출전 시 좋은 = 부재 시 안좋음)
        df["absence_impact"] = -df["adjusted_impact"]
        return df.nlargest(n, "adjusted_impact")[
            ["player_name", "team_abbr", "adjusted_impact", "sample_quality", "is_significant"]
        ]

    @classmethod
    def load(
        cls,
        data_dir: Path,
        season: int,
        use_fallback: bool = True
    ) -> "DataDrivenInjuryCalculator":
        """
        저장된 데이터에서 계산기 로드.

        Args:
            data_dir: 데이터 디렉토리
            season: 시즌 연도
            use_fallback: EPM fallback 사용 여부

        Returns:
            DataDrivenInjuryCalculator 인스턴스
        """
        # On/Off 분석 결과 로드
        impact_path = data_dir / "processed" / "player_impact" / f"season_{season}.parquet"
        if not impact_path.exists():
            logger.warning(f"Player impact data not found: {impact_path}")
            player_impact_df = pd.DataFrame()
        else:
            player_impact_df = pd.read_parquet(impact_path)

        # Fallback 계산기 생성
        fallback = None
        if use_fallback:
            try:
                player_epm = load_player_epm(data_dir, season)
                fallback = InjuryImpactCalculator(player_epm)
            except FileNotFoundError:
                logger.warning(f"Player EPM not found for fallback (season {season})")

        return cls(
            player_impact_df=player_impact_df,
            fallback_calculator=fallback,
            use_fallback=use_fallback
        )

    def save_impact_data(self, output_path: Path):
        """영향도 데이터 저장"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.player_impact_df.to_parquet(output_path, index=False)
        logger.info(f"Saved player impact data to {output_path}")


def create_data_driven_calculator(
    data_dir: Path,
    season: int,
    games_df: pd.DataFrame,
    player_games_df: pd.DataFrame,
    team_epm_df: Optional[pd.DataFrame] = None,
    min_games: int = 10,
    use_fallback: bool = True
) -> DataDrivenInjuryCalculator:
    """
    데이터에서 On/Off 분석을 수행하고 계산기를 생성.

    Args:
        data_dir: 데이터 디렉토리
        season: 시즌 연도
        games_df: 경기 데이터
        player_games_df: 선수 경기 데이터
        team_epm_df: 팀 EPM 데이터
        min_games: 최소 출전 경기 수
        use_fallback: EPM fallback 사용 여부

    Returns:
        DataDrivenInjuryCalculator 인스턴스
    """
    from src.analysis.player_on_off_analyzer import PlayerOnOffAnalyzer

    # On/Off 분석 수행
    analyzer = PlayerOnOffAnalyzer(
        games_df=games_df,
        player_games_df=player_games_df,
        team_epm_df=team_epm_df,
        min_games_on=min_games
    )

    player_impact_df = analyzer.get_all_players_impact()

    # 결과 저장
    output_dir = data_dir / "processed" / "player_impact"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"season_{season}.parquet"
    player_impact_df.to_parquet(output_path, index=False)

    # Fallback 계산기
    fallback = None
    if use_fallback:
        try:
            player_epm = load_player_epm(data_dir, season)
            fallback = InjuryImpactCalculator(player_epm)
        except FileNotFoundError:
            pass

    return DataDrivenInjuryCalculator(
        player_impact_df=player_impact_df,
        fallback_calculator=fallback,
        use_fallback=use_fallback
    )
