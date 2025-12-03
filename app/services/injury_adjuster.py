"""
부상 조정 서비스.

리팩토링 Phase 1: 중복 로직 통합 (Single Source of Truth).

기존 위치:
- app/main.py: apply_injury_correction() (31-67)
- app/services/predictor_v5.py: apply_injury_adjustment() (111-149)
"""


class InjuryAdjuster:
    """
    부상 영향 조정 서비스.

    V5.4 모델의 후행 지표(trailing indicator)로,
    기본 예측 후 부상 정보를 반영하여 최종 확률을 조정합니다.
    """

    # 확률 경계
    MIN_PROB = 0.01  # 1%
    MAX_PROB = 0.99  # 99%

    @staticmethod
    def apply(base_prob: float, home_shift: float, away_shift: float) -> float:
        """
        부상 조정 적용.

        Args:
            base_prob: 기본 예측 확률 (홈팀 승리, 0-1)
            home_shift: 홈팀 부상 영향 (%, 양수 = 홈팀 불리)
            away_shift: 원정팀 부상 영향 (%, 양수 = 원정팀 불리)

        Returns:
            조정된 확률 (1% ~ 99% 범위)

        공식:
            - home_shift% → 홈팀 승률 감소
            - away_shift% → 원정팀 승률 감소 (= 홈팀 승률 증가)
            - net_shift = (away_shift - home_shift) / 100
            - adjusted = base_prob + net_shift
        """
        # % 단위를 소수로 변환
        home_shift_decimal = max(home_shift, 0) / 100.0
        away_shift_decimal = max(away_shift, 0) / 100.0

        # 순 영향 계산 (양수 = 홈팀 유리)
        net_shift = away_shift_decimal - home_shift_decimal

        if net_shift == 0:
            return base_prob

        # 확률 조정 (한도 없음)
        adjusted = base_prob + net_shift

        # 확률 경계 유지
        return max(InjuryAdjuster.MIN_PROB,
                  min(InjuryAdjuster.MAX_PROB, adjusted))

    @staticmethod
    def calculate_net_shift(home_shift: float, away_shift: float) -> float:
        """
        순 부상 영향 계산.

        Args:
            home_shift: 홈팀 부상 영향 (%)
            away_shift: 원정팀 부상 영향 (%)

        Returns:
            순 영향 (%, 양수 = 홈팀 유리)
        """
        return max(away_shift, 0) - max(home_shift, 0)

    @staticmethod
    def describe_adjustment(base_prob: float, home_shift: float,
                           away_shift: float) -> str:
        """
        조정 내용 설명 (디버깅/로깅용).

        Returns:
            조정 설명 문자열
        """
        adjusted = InjuryAdjuster.apply(base_prob, home_shift, away_shift)
        net_shift = InjuryAdjuster.calculate_net_shift(home_shift, away_shift)

        return (
            f"Base: {base_prob:.1%} → Adjusted: {adjusted:.1%} "
            f"(Home Shift: -{home_shift:.1f}%, Away Shift: -{away_shift:.1f}%, "
            f"Net: {net_shift:+.1f}%)"
        )
