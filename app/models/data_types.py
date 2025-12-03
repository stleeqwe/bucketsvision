"""
BucketsVision 데이터 타입 정의.

리팩토링 Phase 1: 원시 타입을 객체로 대체.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional


@dataclass
class TeamInfo:
    """팀 정보"""
    team_id: int
    abbr: str
    name: str
    color: str

    @classmethod
    def from_dict(cls, team_id: int, data: Dict) -> 'TeamInfo':
        return cls(
            team_id=team_id,
            abbr=data.get('abbr', 'UNK'),
            name=data.get('name', 'Unknown'),
            color=data.get('color', '#666666'),
        )


@dataclass
class PlayerInjuryDetail:
    """선수 부상 상세 정보"""
    name: str
    status: str  # "Out" | "GTD"
    detail: str
    epm: float = 0.0
    mpg: float = 0.0
    prob_shift_pct: float = 0.0
    played_games: int = 0
    missed_games: int = 0
    schedule_diff: float = 0.0
    skip_reason: Optional[str] = None

    @classmethod
    def from_injury_result(cls, injury, result, status: str) -> 'PlayerInjuryDetail':
        """팩토리 메서드: InjuryResult로부터 생성"""
        if result.is_valid:
            return cls(
                name=injury.player_name,
                status=status,
                detail=injury.detail,
                epm=round(result.epm, 2),
                mpg=round(result.mpg, 1),
                prob_shift_pct=round(result.prob_shift * 100, 1),
                played_games=result.played_games,
                missed_games=result.missed_games,
                schedule_diff=round(result.schedule_diff, 3),
            )
        else:
            return cls(
                name=injury.player_name,
                status=status,
                detail=injury.detail,
                skip_reason=result.skip_reason,
            )

    def to_dict(self) -> Dict:
        """UI 렌더링용 딕셔너리 변환"""
        return asdict(self)


@dataclass
class InjurySummary:
    """팀 부상 요약"""
    out_players: List[PlayerInjuryDetail] = field(default_factory=list)
    gtd_players: List[PlayerInjuryDetail] = field(default_factory=list)
    total_prob_shift: float = 0.0

    def to_dict(self) -> Dict:
        """UI 렌더링용 딕셔너리 변환"""
        return {
            'out_players': [p.to_dict() for p in self.out_players],
            'gtd_players': [p.to_dict() for p in self.gtd_players],
            'total_prob_shift': self.total_prob_shift,
        }


@dataclass
class PredictionResult:
    """예측 결과"""
    game_id: str
    base_prob: float
    adjusted_prob: float
    features: Dict[str, float]
    home_injury: InjurySummary
    away_injury: InjurySummary

    @property
    def home_injury_shift(self) -> float:
        return self.home_injury.total_prob_shift

    @property
    def away_injury_shift(self) -> float:
        return self.away_injury.total_prob_shift

    @property
    def predicted_winner(self) -> str:
        return 'home' if self.adjusted_prob >= 0.5 else 'away'

    @property
    def confidence(self) -> float:
        """신뢰도 (0~1 스케일)"""
        return abs(self.adjusted_prob - 0.5) * 2


@dataclass
class GameCardData:
    """게임 카드 렌더링 데이터"""
    # 필수 필드 (기본값 없음)
    game_id: str
    game_time: str
    game_status: int  # 1=예정, 2=진행중, 3=종료
    home_abbr: str
    home_name: str
    home_color: str
    away_abbr: str
    away_name: str
    away_color: str

    # 선택 필드 (기본값 있음)
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    home_b2b: bool = False
    away_b2b: bool = False
    home_win_prob: float = 0.5
    predicted_margin: float = 0.0
    home_injury_summary: Optional[Dict] = None
    away_injury_summary: Optional[Dict] = None
    odds_info: Optional[Dict] = None
    hide_result: bool = False
    enable_custom_input: bool = False

    @classmethod
    def from_game_and_prediction(cls, game: Dict, prediction: PredictionResult,
                                 home_info: TeamInfo, away_info: TeamInfo,
                                 odds_info: Optional[Dict] = None) -> 'GameCardData':
        """팩토리 메서드"""
        return cls(
            game_id=game['game_id'],
            game_time=game.get('game_time', ''),
            game_status=game.get('game_status', 1),
            home_abbr=home_info.abbr,
            home_name=home_info.name,
            home_color=home_info.color,
            home_score=game.get('home_score'),
            home_b2b=game.get('home_b2b', False),
            away_abbr=away_info.abbr,
            away_name=away_info.name,
            away_color=away_info.color,
            away_score=game.get('away_score'),
            away_b2b=game.get('away_b2b', False),
            home_win_prob=prediction.adjusted_prob,
            predicted_margin=0.0,  # 계산 필요 시 추가
            home_injury_summary=prediction.home_injury.to_dict(),
            away_injury_summary=prediction.away_injury.to_dict(),
            odds_info=odds_info,
        )
