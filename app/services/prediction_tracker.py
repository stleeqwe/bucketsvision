"""
예측 기록 추적 및 저장 모듈.

경기 시작 1시간 전 기준으로 예측을 기록하고,
경기 종료 후 실제 결과와 비교하여 적중 여부를 저장합니다.

저장 위치: data/predictions/
"""

import json
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
import pytz

from src.utils.logger import logger


@dataclass
class PredictionRecord:
    """단일 경기 예측 기록"""
    # 경기 정보
    game_id: str
    game_date: str  # YYYY-MM-DD
    game_time_et: str  # 예: "7:00 PM ET"
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int

    # 예측 정보
    prediction_time: str  # ISO format (경기 1시간 전 기준)
    model_version: str
    home_win_prob: float
    predicted_winner: str  # 팀 약어
    predicted_margin: float  # 예상 점수차

    # 주요 피처 (분석용)
    team_epm_diff: float = 0.0
    player_rotation_epm_diff: float = 0.0
    bench_strength_diff: float = 0.0

    # 실제 결과 (경기 종료 후 업데이트)
    actual_home_score: Optional[int] = None
    actual_away_score: Optional[int] = None
    actual_winner: Optional[str] = None
    actual_margin: Optional[int] = None
    is_correct: Optional[bool] = None
    result_updated_at: Optional[str] = None

    # 메타데이터
    season: int = 2026
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class PredictionTracker:
    """예측 기록 추적기"""

    def __init__(self, data_dir: Path):
        """
        Args:
            data_dir: 데이터 디렉토리 (project_root/data)
        """
        self.data_dir = data_dir
        self.predictions_dir = data_dir / "predictions"
        self.records_dir = self.predictions_dir / "records"
        self.daily_dir = self.predictions_dir / "daily"

        # 디렉토리 생성
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.daily_dir.mkdir(parents=True, exist_ok=True)

        self.et_tz = pytz.timezone('America/New_York')

    def get_season_file(self, season: int) -> Path:
        """시즌별 기록 파일 경로"""
        return self.records_dir / f"season_{season}.json"

    def get_daily_file(self, game_date: date) -> Path:
        """일별 기록 파일 경로"""
        return self.daily_dir / f"{game_date.strftime('%Y-%m-%d')}.json"

    # =========================================================================
    # 데이터 로드/저장
    # =========================================================================

    def load_season_records(self, season: int = 2026) -> Dict[str, PredictionRecord]:
        """
        시즌 전체 기록 로드.

        Returns:
            {game_id: PredictionRecord} 딕셔너리
        """
        file_path = self.get_season_file(season)
        if not file_path.exists():
            return {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            records = {}
            for game_id, record_data in data.items():
                records[game_id] = PredictionRecord(**record_data)

            return records
        except Exception as e:
            logger.error(f"Error loading season records: {e}")
            return {}

    def save_season_records(self, records: Dict[str, PredictionRecord], season: int = 2026) -> None:
        """시즌 전체 기록 저장"""
        file_path = self.get_season_file(season)

        data = {game_id: asdict(record) for game_id, record in records.items()}

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(records)} records to {file_path}")

    def save_daily_snapshot(self, records: List[PredictionRecord], game_date: date) -> None:
        """일별 스냅샷 저장 (백업용)"""
        file_path = self.get_daily_file(game_date)

        data = [asdict(record) for record in records]

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved daily snapshot: {file_path}")

    # =========================================================================
    # 예측 기록
    # =========================================================================

    def record_prediction(
        self,
        game_id: str,
        game_date: str,
        game_time_et: str,
        home_team: str,
        away_team: str,
        home_team_id: int,
        away_team_id: int,
        home_win_prob: float,
        predicted_margin: float,
        model_version: str,
        features: Dict[str, float],
        season: int = 2026
    ) -> PredictionRecord:
        """
        단일 경기 예측 기록.

        Args:
            game_id: 경기 ID
            game_date: 경기 날짜 (YYYY-MM-DD)
            game_time_et: 경기 시간 (ET)
            home_team: 홈팀 약어
            away_team: 원정팀 약어
            home_team_id: 홈팀 ID
            away_team_id: 원정팀 ID
            home_win_prob: 홈팀 승리 확률
            predicted_margin: 예상 점수차
            model_version: 모델 버전
            features: 피처 딕셔너리
            season: 시즌 연도

        Returns:
            생성된 PredictionRecord
        """
        predicted_winner = home_team if home_win_prob >= 0.5 else away_team

        record = PredictionRecord(
            game_id=game_id,
            game_date=game_date,
            game_time_et=game_time_et,
            home_team=home_team,
            away_team=away_team,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            prediction_time=datetime.now(self.et_tz).isoformat(),
            model_version=model_version,
            home_win_prob=round(home_win_prob, 4),
            predicted_winner=predicted_winner,
            predicted_margin=round(predicted_margin, 1),
            team_epm_diff=round(features.get('team_epm_diff', 0), 4),
            player_rotation_epm_diff=round(features.get('player_rotation_epm_diff', 0), 4),
            bench_strength_diff=round(features.get('bench_strength_diff', 0), 4),
            season=season,
        )

        # 시즌 기록에 추가
        records = self.load_season_records(season)
        records[game_id] = record
        self.save_season_records(records, season)

        logger.info(f"Recorded prediction: {away_team} @ {home_team} -> {predicted_winner} ({home_win_prob:.1%})")

        return record

    def record_game_predictions(
        self,
        games: List[Dict],
        predictions: List[Tuple[float, float, Dict]],  # (prob, margin, features)
        model_version: str,
        season: int = 2026
    ) -> List[PredictionRecord]:
        """
        여러 경기 예측 일괄 기록.

        Args:
            games: 경기 정보 리스트
            predictions: (확률, 마진, 피처) 튜플 리스트
            model_version: 모델 버전
            season: 시즌 연도

        Returns:
            생성된 PredictionRecord 리스트
        """
        records = self.load_season_records(season)
        new_records = []

        for game, (prob, margin, features) in zip(games, predictions):
            game_id = game['game_id']

            # 이미 기록된 경기는 건너뛰기
            if game_id in records:
                logger.debug(f"Game {game_id} already recorded, skipping")
                continue

            from config.constants import TEAM_INFO
            home_info = TEAM_INFO.get(game['home_team_id'], {})
            away_info = TEAM_INFO.get(game['away_team_id'], {})

            predicted_winner = home_info.get('abbr', 'UNK') if prob >= 0.5 else away_info.get('abbr', 'UNK')

            record = PredictionRecord(
                game_id=game_id,
                game_date=game.get('game_date', ''),
                game_time_et=game.get('game_time', ''),
                home_team=home_info.get('abbr', 'UNK'),
                away_team=away_info.get('abbr', 'UNK'),
                home_team_id=game['home_team_id'],
                away_team_id=game['away_team_id'],
                prediction_time=datetime.now(self.et_tz).isoformat(),
                model_version=model_version,
                home_win_prob=round(prob, 4),
                predicted_winner=predicted_winner,
                predicted_margin=round(margin, 1),
                team_epm_diff=round(features.get('team_epm_diff', 0), 4),
                player_rotation_epm_diff=round(features.get('player_rotation_epm_diff', 0), 4),
                bench_strength_diff=round(features.get('bench_strength_diff', 0), 4),
                season=season,
            )

            records[game_id] = record
            new_records.append(record)

        if new_records:
            self.save_season_records(records, season)

            # 일별 스냅샷도 저장
            if new_records:
                game_date = datetime.strptime(new_records[0].game_date, '%Y-%m-%d').date()
                self.save_daily_snapshot(new_records, game_date)

        logger.info(f"Recorded {len(new_records)} new predictions")
        return new_records

    # =========================================================================
    # 결과 업데이트
    # =========================================================================

    def update_result(
        self,
        game_id: str,
        home_score: int,
        away_score: int,
        season: int = 2026
    ) -> Optional[PredictionRecord]:
        """
        경기 결과 업데이트.

        Args:
            game_id: 경기 ID
            home_score: 홈팀 점수
            away_score: 원정팀 점수
            season: 시즌 연도

        Returns:
            업데이트된 PredictionRecord (없으면 None)
        """
        records = self.load_season_records(season)

        if game_id not in records:
            logger.warning(f"Game {game_id} not found in records")
            return None

        record = records[game_id]

        # 이미 결과가 업데이트되었으면 건너뛰기
        if record.actual_home_score is not None:
            logger.debug(f"Game {game_id} already has result, skipping")
            return record

        # 결과 업데이트
        record.actual_home_score = home_score
        record.actual_away_score = away_score
        record.actual_margin = home_score - away_score
        record.actual_winner = record.home_team if home_score > away_score else record.away_team
        record.is_correct = (record.predicted_winner == record.actual_winner)
        record.result_updated_at = datetime.now(self.et_tz).isoformat()

        records[game_id] = record
        self.save_season_records(records, season)

        status = "✓" if record.is_correct else "✗"
        logger.info(f"Updated result: {record.away_team} @ {record.home_team} = "
                   f"{away_score}-{home_score} | Predicted: {record.predicted_winner} | {status}")

        return record

    def update_results_batch(
        self,
        results: List[Dict],  # [{game_id, home_score, away_score}]
        season: int = 2026
    ) -> Tuple[int, int]:
        """
        여러 경기 결과 일괄 업데이트.

        Returns:
            (업데이트 수, 적중 수)
        """
        updated = 0
        correct = 0

        for result in results:
            record = self.update_result(
                game_id=result['game_id'],
                home_score=result['home_score'],
                away_score=result['away_score'],
                season=season
            )

            if record and record.result_updated_at:
                updated += 1
                if record.is_correct:
                    correct += 1

        logger.info(f"Batch update: {updated} games, {correct} correct ({correct/updated*100:.1f}%)" if updated else "No updates")
        return updated, correct

    # =========================================================================
    # 통계 조회
    # =========================================================================

    def get_stats(self, season: int = 2026) -> Dict:
        """
        시즌 통계 조회.

        Returns:
            통계 딕셔너리
        """
        records = self.load_season_records(season)

        if not records:
            return {
                'season': season,
                'total_predictions': 0,
                'completed_games': 0,
                'correct_predictions': 0,
                'accuracy': 0.0,
                'pending_games': 0,
            }

        total = len(records)
        completed = [r for r in records.values() if r.actual_home_score is not None]
        correct = [r for r in completed if r.is_correct]
        pending = total - len(completed)

        # 신뢰도별 정확도
        confidence_stats = {}
        for threshold in [0.55, 0.60, 0.65, 0.70, 0.75]:
            filtered = [r for r in completed
                       if r.home_win_prob >= threshold or r.home_win_prob <= (1 - threshold)]
            if filtered:
                conf_correct = len([r for r in filtered if r.is_correct])
                confidence_stats[f'conf_{int(threshold*100)}'] = {
                    'count': len(filtered),
                    'correct': conf_correct,
                    'accuracy': conf_correct / len(filtered),
                }

        return {
            'season': season,
            'total_predictions': total,
            'completed_games': len(completed),
            'correct_predictions': len(correct),
            'accuracy': len(correct) / len(completed) if completed else 0.0,
            'pending_games': pending,
            'confidence_stats': confidence_stats,
            'last_updated': datetime.now().isoformat(),
        }

    def get_recent_predictions(
        self,
        n: int = 10,
        season: int = 2026,
        only_completed: bool = False
    ) -> List[PredictionRecord]:
        """최근 예측 조회"""
        records = self.load_season_records(season)

        sorted_records = sorted(
            records.values(),
            key=lambda r: r.game_date,
            reverse=True
        )

        if only_completed:
            sorted_records = [r for r in sorted_records if r.actual_home_score is not None]

        return sorted_records[:n]

    def get_predictions_by_date(
        self,
        game_date: date,
        season: int = 2026
    ) -> List[PredictionRecord]:
        """특정 날짜 예측 조회"""
        records = self.load_season_records(season)
        date_str = game_date.strftime('%Y-%m-%d')

        return [r for r in records.values() if r.game_date == date_str]

    def get_model_performance_by_date(self, season: int = 2026) -> List[Dict]:
        """날짜별 성과 조회"""
        records = self.load_season_records(season)
        completed = [r for r in records.values() if r.actual_home_score is not None]

        # 날짜별 그룹화
        by_date = {}
        for r in completed:
            if r.game_date not in by_date:
                by_date[r.game_date] = {'total': 0, 'correct': 0}
            by_date[r.game_date]['total'] += 1
            if r.is_correct:
                by_date[r.game_date]['correct'] += 1

        result = []
        for date_str, stats in sorted(by_date.items()):
            result.append({
                'date': date_str,
                'total': stats['total'],
                'correct': stats['correct'],
                'accuracy': stats['correct'] / stats['total'] if stats['total'] else 0,
            })

        return result
