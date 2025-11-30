"""
Data validation utilities.

수집된 데이터의 품질을 검증하고 이상치를 탐지합니다.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.logger import logger
from config.feature_config import NBA_TEAMS, TEAM_ID_TO_ABBR


class ValidationLevel(Enum):
    """검증 수준"""
    ERROR = "error"      # 치명적 오류 - 데이터 사용 불가
    WARNING = "warning"  # 경고 - 주의 필요
    INFO = "info"        # 정보 - 참고용


@dataclass
class ValidationResult:
    """검증 결과"""
    is_valid: bool
    level: ValidationLevel
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationReport:
    """검증 리포트"""
    results: List[ValidationResult] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """전체 유효성"""
        return all(r.level != ValidationLevel.ERROR for r in self.results)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.level == ValidationLevel.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results if r.level == ValidationLevel.WARNING)

    def add(self, result: ValidationResult) -> None:
        """결과 추가"""
        self.results.append(result)

    def get_errors(self) -> List[ValidationResult]:
        """에러만 반환"""
        return [r for r in self.results if r.level == ValidationLevel.ERROR]

    def get_warnings(self) -> List[ValidationResult]:
        """경고만 반환"""
        return [r for r in self.results if r.level == ValidationLevel.WARNING]

    def summary(self) -> str:
        """요약 문자열"""
        return f"Validation: {len(self.results)} checks, {self.error_count} errors, {self.warning_count} warnings"


class DataValidator:
    """데이터 검증기 기본 클래스"""

    def validate(self, data: Any) -> ValidationReport:
        """데이터 검증"""
        raise NotImplementedError


class TeamEPMValidator(DataValidator):
    """팀 EPM 데이터 검증기"""

    # 필수 필드
    REQUIRED_FIELDS = [
        "team_id", "game_dt", "team_epm", "team_oepm", "team_depm",
        "sos", "sos_o", "sos_d", "team_alias"
    ]

    # EPM 값 범위
    EPM_MIN = -15.0
    EPM_MAX = 15.0

    # SOS 값 범위
    SOS_MIN = -5.0
    SOS_MAX = 5.0

    def validate(self, data: List[Dict[str, Any]]) -> ValidationReport:
        """
        팀 EPM 데이터 검증.

        Args:
            data: 팀 EPM 레코드 리스트

        Returns:
            검증 리포트
        """
        report = ValidationReport()

        if not data:
            report.add(ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Empty data",
            ))
            return report

        # 1. 데이터 크기 검증
        report.add(self._validate_size(data))

        # 2. 필수 필드 검증
        report.add(self._validate_required_fields(data))

        # 3. 팀 ID 검증
        report.add(self._validate_team_ids(data))

        # 4. 값 범위 검증
        report.add(self._validate_value_ranges(data))

        # 5. 날짜 형식 검증
        report.add(self._validate_dates(data))

        # 6. 중복 검증
        report.add(self._validate_duplicates(data))

        return report

    def _validate_size(self, data: List[Dict]) -> ValidationResult:
        """데이터 크기 검증"""
        size = len(data)

        if size == 0:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="No records found"
            )

        # 시즌 데이터는 최소 30팀 * 150일 = 4500 레코드 예상
        if size < 100:
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.WARNING,
                message=f"Low record count: {size}",
                details={"count": size}
            )

        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message=f"Record count: {size}",
            details={"count": size}
        )

    def _validate_required_fields(self, data: List[Dict]) -> ValidationResult:
        """필수 필드 존재 검증"""
        if not data:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="No data to validate"
            )

        sample = data[0]
        missing_fields = [f for f in self.REQUIRED_FIELDS if f not in sample]

        if missing_fields:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Missing required fields: {missing_fields}",
                details={"missing": missing_fields}
            )

        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="All required fields present"
        )

    def _validate_team_ids(self, data: List[Dict]) -> ValidationResult:
        """팀 ID 유효성 검증"""
        valid_team_ids = set(NBA_TEAMS.keys())
        data_team_ids = set(r.get("team_id") for r in data if r.get("team_id"))

        invalid_ids = data_team_ids - valid_team_ids

        if invalid_ids:
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.WARNING,
                message=f"Unknown team IDs found: {invalid_ids}",
                details={"invalid_ids": list(invalid_ids)}
            )

        # 30팀 모두 존재하는지 확인
        missing_teams = valid_team_ids - data_team_ids

        if missing_teams:
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.WARNING,
                message=f"Missing teams: {len(missing_teams)}",
                details={"missing_team_ids": list(missing_teams)}
            )

        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="All 30 teams present"
        )

    def _validate_value_ranges(self, data: List[Dict]) -> ValidationResult:
        """값 범위 검증"""
        out_of_range = []

        for i, record in enumerate(data):
            # EPM 범위 검증
            for field in ["team_epm", "team_oepm", "team_depm"]:
                value = record.get(field)
                if value is not None and (value < self.EPM_MIN or value > self.EPM_MAX):
                    out_of_range.append({
                        "index": i,
                        "field": field,
                        "value": value,
                        "team_id": record.get("team_id")
                    })

            # SOS 범위 검증
            for field in ["sos", "sos_o", "sos_d"]:
                value = record.get(field)
                if value is not None and (value < self.SOS_MIN or value > self.SOS_MAX):
                    out_of_range.append({
                        "index": i,
                        "field": field,
                        "value": value,
                        "team_id": record.get("team_id")
                    })

        if out_of_range:
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.WARNING,
                message=f"Values out of expected range: {len(out_of_range)}",
                details={"out_of_range": out_of_range[:10]}  # 최대 10개만
            )

        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="All values within expected range"
        )

    def _validate_dates(self, data: List[Dict]) -> ValidationResult:
        """날짜 형식 검증"""
        invalid_dates = []

        for i, record in enumerate(data):
            date_str = record.get("game_dt")
            if date_str:
                try:
                    pd.to_datetime(date_str)
                except:
                    invalid_dates.append({"index": i, "date": date_str})

        if invalid_dates:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Invalid date formats: {len(invalid_dates)}",
                details={"invalid_dates": invalid_dates[:10]}
            )

        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="All dates valid"
        )

    def _validate_duplicates(self, data: List[Dict]) -> ValidationResult:
        """중복 레코드 검증"""
        seen = set()
        duplicates = []

        for i, record in enumerate(data):
            key = (record.get("team_id"), record.get("game_dt"))
            if key in seen:
                duplicates.append({
                    "index": i,
                    "team_id": record.get("team_id"),
                    "game_dt": record.get("game_dt")
                })
            seen.add(key)

        if duplicates:
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.WARNING,
                message=f"Duplicate records found: {len(duplicates)}",
                details={"duplicates": duplicates[:10]}
            )

        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message="No duplicates found"
        )


class PlayerEPMValidator(DataValidator):
    """선수 EPM 데이터 검증기"""

    REQUIRED_FIELDS = [
        "player_id", "player_name", "team_id", "epm", "oepm", "depm", "p_mp_48"
    ]

    EPM_MIN = -10.0
    EPM_MAX = 12.0

    def validate(self, data: List[Dict[str, Any]]) -> ValidationReport:
        """선수 EPM 데이터 검증"""
        report = ValidationReport()

        if not data:
            report.add(ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Empty data"
            ))
            return report

        # 필수 필드 검증
        sample = data[0]
        missing = [f for f in self.REQUIRED_FIELDS if f not in sample]

        if missing:
            report.add(ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Missing fields: {missing}"
            ))
        else:
            report.add(ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message="All required fields present"
            ))

        # EPM 값 범위 검증
        out_of_range = []
        for record in data:
            epm = record.get("epm")
            if epm is not None and (epm < self.EPM_MIN or epm > self.EPM_MAX):
                out_of_range.append({
                    "player_name": record.get("player_name"),
                    "epm": epm
                })

        if out_of_range:
            report.add(ValidationResult(
                is_valid=True,
                level=ValidationLevel.WARNING,
                message=f"EPM values out of range: {len(out_of_range)}",
                details={"samples": out_of_range[:5]}
            ))

        return report


class GameDataValidator(DataValidator):
    """경기 데이터 검증기"""

    def validate(self, games: pd.DataFrame) -> ValidationReport:
        """경기 DataFrame 검증"""
        report = ValidationReport()

        if games.empty:
            report.add(ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Empty DataFrame"
            ))
            return report

        # 필수 컬럼 검증
        required_cols = ["game_id", "game_date", "team_id"]
        missing = [c for c in required_cols if c not in games.columns]

        if missing:
            report.add(ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Missing columns: {missing}"
            ))
        else:
            report.add(ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message=f"DataFrame shape: {games.shape}"
            ))

        # 결측치 검증
        null_counts = games.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]

        if len(cols_with_nulls) > 0:
            report.add(ValidationResult(
                is_valid=True,
                level=ValidationLevel.WARNING,
                message=f"Columns with null values: {len(cols_with_nulls)}",
                details={"null_counts": cols_with_nulls.to_dict()}
            ))

        return report


# ===================
# Validation Functions
# ===================

def validate_team_epm_data(data: List[Dict[str, Any]]) -> ValidationReport:
    """팀 EPM 데이터 검증 헬퍼 함수"""
    validator = TeamEPMValidator()
    return validator.validate(data)


def validate_player_epm_data(data: List[Dict[str, Any]]) -> ValidationReport:
    """선수 EPM 데이터 검증 헬퍼 함수"""
    validator = PlayerEPMValidator()
    return validator.validate(data)


def validate_game_data(games: pd.DataFrame) -> ValidationReport:
    """경기 데이터 검증 헬퍼 함수"""
    validator = GameDataValidator()
    return validator.validate(games)


def log_validation_report(report: ValidationReport, name: str = "Data") -> None:
    """검증 리포트 로깅"""
    logger.info(f"{name} validation: {report.summary()}")

    for result in report.get_errors():
        logger.error(f"  ERROR: {result.message}")

    for result in report.get_warnings():
        logger.warning(f"  WARNING: {result.message}")
