"""
Feature configuration and definitions.

이 모듈은 모델에서 사용하는 모든 피처의 정의와 메타데이터를 관리합니다.
피처 추가/수정 시 이 파일만 업데이트하면 됩니다.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any


class FeatureCategory(Enum):
    """피처 카테고리"""
    FOUR_FACTORS = "four_factors"
    REST_FATIGUE = "rest_fatigue"
    TEAM_STRENGTH = "team_strength"
    PLAYER_EPM = "player_epm"
    LINEUP = "lineup"


class FeatureSource(Enum):
    """피처 데이터 소스"""
    DNT_TEAM_EPM = "dnt_team_epm"
    DNT_PLAYER_EPM = "dnt_player_epm"
    DNT_SEASON_EPM = "dnt_season_epm"
    NBA_STATS_BOXSCORE = "nba_stats_boxscore"
    NBA_STATS_SCHEDULE = "nba_stats_schedule"
    NBA_STATS_LINEUP = "nba_stats_lineup"
    COMPUTED = "computed"


@dataclass
class FeatureDefinition:
    """개별 피처 정의"""
    name: str
    category: FeatureCategory
    source: FeatureSource
    description: str
    formula: Optional[str] = None
    rolling_window: Optional[int] = None
    is_diff: bool = True  # Home - Away 차분 여부
    nullable: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def __post_init__(self):
        """유효성 검증"""
        if self.min_value is not None and self.max_value is not None:
            if self.min_value >= self.max_value:
                raise ValueError(f"min_value must be less than max_value for {self.name}")


@dataclass
class FeatureSet:
    """피처 세트 관리"""
    features: List[FeatureDefinition] = field(default_factory=list)

    def add(self, feature: FeatureDefinition) -> None:
        """피처 추가"""
        self.features.append(feature)

    def get_by_category(self, category: FeatureCategory) -> List[FeatureDefinition]:
        """카테고리별 피처 조회"""
        return [f for f in self.features if f.category == category]

    def get_by_source(self, source: FeatureSource) -> List[FeatureDefinition]:
        """소스별 피처 조회"""
        return [f for f in self.features if f.source == source]

    def get_feature_names(self) -> List[str]:
        """전체 피처명 목록"""
        return [f.name for f in self.features]

    def get_dnt_features(self) -> List[FeatureDefinition]:
        """D&T 기반 피처 조회"""
        dnt_sources = {
            FeatureSource.DNT_TEAM_EPM,
            FeatureSource.DNT_PLAYER_EPM,
            FeatureSource.DNT_SEASON_EPM
        }
        return [f for f in self.features if f.source in dnt_sources]

    def __len__(self) -> int:
        return len(self.features)

    def summary(self) -> Dict[str, int]:
        """카테고리별 피처 수 요약"""
        summary = {}
        for cat in FeatureCategory:
            count = len(self.get_by_category(cat))
            if count > 0:
                summary[cat.value] = count
        return summary


# ===================
# 피처 정의
# ===================

def create_feature_set() -> FeatureSet:
    """전체 피처 세트 생성"""
    fs = FeatureSet()

    # -----------------------
    # Four Factors (8개)
    # -----------------------
    four_factors_metrics = [
        ("eFG", "Effective Field Goal Percentage", "(FG + 0.5 * 3P) / FGA"),
        ("TOV", "Turnover Percentage", "TOV / (FGA + 0.44 * FTA + TOV)"),
        ("ORB", "Offensive Rebound Percentage", "ORB / (ORB + Opp_DRB)"),
        ("FTR", "Free Throw Rate", "FT / FGA"),
    ]

    for metric, desc, formula in four_factors_metrics:
        for window in [5, 10]:
            fs.add(FeatureDefinition(
                name=f"{metric}_diff_{window}G",
                category=FeatureCategory.FOUR_FACTORS,
                source=FeatureSource.NBA_STATS_BOXSCORE,
                description=f"{desc} difference (last {window} games)",
                formula=f"Home_{metric}%_{window}G - Away_{metric}%_allowed_{window}G",
                rolling_window=window,
                is_diff=True,
                min_value=-0.5,
                max_value=0.5
            ))

    # -----------------------
    # Rest/Fatigue (7개)
    # -----------------------
    fs.add(FeatureDefinition(
        name="rest_days_home",
        category=FeatureCategory.REST_FATIGUE,
        source=FeatureSource.NBA_STATS_SCHEDULE,
        description="Home team rest days (capped at 4+)",
        is_diff=False,
        min_value=0,
        max_value=4
    ))

    fs.add(FeatureDefinition(
        name="rest_days_away",
        category=FeatureCategory.REST_FATIGUE,
        source=FeatureSource.NBA_STATS_SCHEDULE,
        description="Away team rest days (capped at 4+)",
        is_diff=False,
        min_value=0,
        max_value=4
    ))

    fs.add(FeatureDefinition(
        name="rest_diff",
        category=FeatureCategory.REST_FATIGUE,
        source=FeatureSource.COMPUTED,
        description="Rest days difference (home - away)",
        formula="rest_days_home - rest_days_away",
        is_diff=True,
        min_value=-4,
        max_value=4
    ))

    fs.add(FeatureDefinition(
        name="back_to_back",
        category=FeatureCategory.REST_FATIGUE,
        source=FeatureSource.NBA_STATS_SCHEDULE,
        description="Back-to-back status (0: none, 1: away only, 2: home only, 3: both)",
        is_diff=False,
        min_value=0,
        max_value=3
    ))

    fs.add(FeatureDefinition(
        name="travel_km_away",
        category=FeatureCategory.REST_FATIGUE,
        source=FeatureSource.COMPUTED,
        description="Away team travel distance from last game (km)",
        is_diff=False,
        min_value=0,
        max_value=5000
    ))

    fs.add(FeatureDefinition(
        name="games_7d_diff",
        category=FeatureCategory.REST_FATIGUE,
        source=FeatureSource.COMPUTED,
        description="Games played in last 7 days difference",
        formula="Away_games_in_7days - Home_games_in_7days",
        is_diff=True,
        min_value=-4,
        max_value=4
    ))

    fs.add(FeatureDefinition(
        name="timezone_shift_away",
        category=FeatureCategory.REST_FATIGUE,
        source=FeatureSource.COMPUTED,
        description="Away team timezone shift (East->West: +, West->East: -)",
        is_diff=False,
        min_value=-3,
        max_value=3
    ))

    # -----------------------
    # Team Strength - D&T EPM/SOS (7개)
    # -----------------------
    fs.add(FeatureDefinition(
        name="team_epm_diff",
        category=FeatureCategory.TEAM_STRENGTH,
        source=FeatureSource.DNT_TEAM_EPM,
        description="Team EPM difference (D&T team-epm)",
        formula="Home_team_epm - Away_team_epm",
        is_diff=True,
        min_value=-20,
        max_value=20
    ))

    fs.add(FeatureDefinition(
        name="sos_diff",
        category=FeatureCategory.TEAM_STRENGTH,
        source=FeatureSource.DNT_TEAM_EPM,
        description="Strength of Schedule difference",
        formula="Home_sos - Away_sos",
        is_diff=True,
        min_value=-5,
        max_value=5
    ))

    fs.add(FeatureDefinition(
        name="sos_o_diff",
        category=FeatureCategory.TEAM_STRENGTH,
        source=FeatureSource.DNT_TEAM_EPM,
        description="Offensive Strength of Schedule difference",
        formula="Home_sos_o - Away_sos_o",
        is_diff=True,
        min_value=-5,
        max_value=5
    ))

    fs.add(FeatureDefinition(
        name="sos_d_diff",
        category=FeatureCategory.TEAM_STRENGTH,
        source=FeatureSource.DNT_TEAM_EPM,
        description="Defensive Strength of Schedule difference",
        formula="Home_sos_d - Away_sos_d",
        is_diff=True,
        min_value=-5,
        max_value=5
    ))

    fs.add(FeatureDefinition(
        name="win_pct_10G_diff",
        category=FeatureCategory.TEAM_STRENGTH,
        source=FeatureSource.NBA_STATS_SCHEDULE,
        description="Win percentage difference (last 10 games)",
        formula="Home_Win%_10G - Away_Win%_10G",
        rolling_window=10,
        is_diff=True,
        min_value=-1,
        max_value=1
    ))

    fs.add(FeatureDefinition(
        name="home_record",
        category=FeatureCategory.TEAM_STRENGTH,
        source=FeatureSource.NBA_STATS_SCHEDULE,
        description="Home team's home game win percentage",
        is_diff=False,
        min_value=0,
        max_value=1
    ))

    fs.add(FeatureDefinition(
        name="away_record",
        category=FeatureCategory.TEAM_STRENGTH,
        source=FeatureSource.NBA_STATS_SCHEDULE,
        description="Away team's away game win percentage",
        is_diff=False,
        min_value=0,
        max_value=1
    ))

    # -----------------------
    # Player EPM - D&T (3개)
    # -----------------------
    fs.add(FeatureDefinition(
        name="top5_epm_diff",
        category=FeatureCategory.PLAYER_EPM,
        source=FeatureSource.DNT_PLAYER_EPM,
        description="Top 5 players EPM difference (weighted by expected minutes)",
        formula="Home_Top5_EPM - Away_Top5_EPM",
        is_diff=True,
        min_value=-30,
        max_value=30
    ))

    fs.add(FeatureDefinition(
        name="top8_epm_diff",
        category=FeatureCategory.PLAYER_EPM,
        source=FeatureSource.DNT_PLAYER_EPM,
        description="Top 8 rotation players EPM difference",
        formula="Home_8man_EPM - Away_8man_EPM",
        is_diff=True,
        min_value=-40,
        max_value=40
    ))

    fs.add(FeatureDefinition(
        name="bench_epm_diff",
        category=FeatureCategory.PLAYER_EPM,
        source=FeatureSource.DNT_PLAYER_EPM,
        description="Bench players (6-10) EPM difference",
        formula="Home_Bench_EPM - Away_Bench_EPM",
        is_diff=True,
        min_value=-15,
        max_value=15
    ))

    # -----------------------
    # Lineup (4개)
    # -----------------------
    fs.add(FeatureDefinition(
        name="starter_netrtg_diff",
        category=FeatureCategory.LINEUP,
        source=FeatureSource.NBA_STATS_LINEUP,
        description="Starting 5 Net Rating difference",
        formula="Home_Starter5_NetRtg - Away_Starter5_NetRtg",
        is_diff=True,
        nullable=True,  # 100 포제션 미달 시 null 가능
        min_value=-50,
        max_value=50
    ))

    fs.add(FeatureDefinition(
        name="bench_netrtg_diff",
        category=FeatureCategory.LINEUP,
        source=FeatureSource.NBA_STATS_LINEUP,
        description="Bench unit Net Rating difference",
        formula="Home_Bench_NetRtg - Away_Bench_NetRtg",
        is_diff=True,
        nullable=True,
        min_value=-50,
        max_value=50
    ))

    fs.add(FeatureDefinition(
        name="depth_diff",
        category=FeatureCategory.LINEUP,
        source=FeatureSource.DNT_PLAYER_EPM,
        description="8-man rotation depth EPM difference",
        formula="Home_8man_avg_EPM - Away_8man_avg_EPM",
        is_diff=True,
        min_value=-10,
        max_value=10
    ))

    fs.add(FeatureDefinition(
        name="best_duo_diff",
        category=FeatureCategory.LINEUP,
        source=FeatureSource.NBA_STATS_LINEUP,
        description="Best 2-man combination Net Rating difference",
        formula="Home_BestDuo_NetRtg - Away_BestDuo_NetRtg",
        is_diff=True,
        nullable=True,
        min_value=-50,
        max_value=50
    ))

    return fs


# 전역 피처 세트 인스턴스
FEATURE_SET = create_feature_set()

# 피처 이름 상수
FEATURE_NAMES = FEATURE_SET.get_feature_names()

# 피처 수 요약
FEATURE_SUMMARY = FEATURE_SET.summary()


# ===================
# NBA Team IDs
# ===================
NBA_TEAMS = {
    1610612737: {"abbr": "ATL", "name": "Atlanta Hawks", "timezone": "ET"},
    1610612738: {"abbr": "BOS", "name": "Boston Celtics", "timezone": "ET"},
    1610612739: {"abbr": "CLE", "name": "Cleveland Cavaliers", "timezone": "ET"},
    1610612740: {"abbr": "NOP", "name": "New Orleans Pelicans", "timezone": "CT"},
    1610612741: {"abbr": "CHI", "name": "Chicago Bulls", "timezone": "CT"},
    1610612742: {"abbr": "DAL", "name": "Dallas Mavericks", "timezone": "CT"},
    1610612743: {"abbr": "DEN", "name": "Denver Nuggets", "timezone": "MT"},
    1610612744: {"abbr": "GSW", "name": "Golden State Warriors", "timezone": "PT"},
    1610612745: {"abbr": "HOU", "name": "Houston Rockets", "timezone": "CT"},
    1610612746: {"abbr": "LAC", "name": "LA Clippers", "timezone": "PT"},
    1610612747: {"abbr": "LAL", "name": "Los Angeles Lakers", "timezone": "PT"},
    1610612748: {"abbr": "MIA", "name": "Miami Heat", "timezone": "ET"},
    1610612749: {"abbr": "MIL", "name": "Milwaukee Bucks", "timezone": "CT"},
    1610612750: {"abbr": "MIN", "name": "Minnesota Timberwolves", "timezone": "CT"},
    1610612751: {"abbr": "BKN", "name": "Brooklyn Nets", "timezone": "ET"},
    1610612752: {"abbr": "NYK", "name": "New York Knicks", "timezone": "ET"},
    1610612753: {"abbr": "ORL", "name": "Orlando Magic", "timezone": "ET"},
    1610612754: {"abbr": "IND", "name": "Indiana Pacers", "timezone": "ET"},
    1610612755: {"abbr": "PHI", "name": "Philadelphia 76ers", "timezone": "ET"},
    1610612756: {"abbr": "PHX", "name": "Phoenix Suns", "timezone": "MT"},
    1610612757: {"abbr": "POR", "name": "Portland Trail Blazers", "timezone": "PT"},
    1610612758: {"abbr": "SAC", "name": "Sacramento Kings", "timezone": "PT"},
    1610612759: {"abbr": "SAS", "name": "San Antonio Spurs", "timezone": "CT"},
    1610612760: {"abbr": "OKC", "name": "Oklahoma City Thunder", "timezone": "CT"},
    1610612761: {"abbr": "TOR", "name": "Toronto Raptors", "timezone": "ET"},
    1610612762: {"abbr": "UTA", "name": "Utah Jazz", "timezone": "MT"},
    1610612763: {"abbr": "MEM", "name": "Memphis Grizzlies", "timezone": "CT"},
    1610612764: {"abbr": "WAS", "name": "Washington Wizards", "timezone": "ET"},
    1610612765: {"abbr": "DET", "name": "Detroit Pistons", "timezone": "ET"},
    1610612766: {"abbr": "CHA", "name": "Charlotte Hornets", "timezone": "ET"},
}

# Team ID <-> Abbreviation 변환
TEAM_ID_TO_ABBR = {k: v["abbr"] for k, v in NBA_TEAMS.items()}
TEAM_ABBR_TO_ID = {v["abbr"]: k for k, v in NBA_TEAMS.items()}

# Timezone offset (hours from ET)
TIMEZONE_OFFSETS = {
    "ET": 0,
    "CT": -1,
    "MT": -2,
    "PT": -3,
}
