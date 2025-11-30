"""
Bayesian Hierarchical Model for Player Impact.

EPM을 사전 정보로 활용하고 실제 경기 데이터로
사후 분포를 업데이트하여 선수 영향도를 추정합니다.

핵심 개념:
- Prior: EPM 기반 사전 분포 (선수 영향도에 대한 초기 추정)
- Likelihood: 실제 On/Off 경기 데이터
- Posterior: 업데이트된 영향도 분포 (불확실성 포함)

수학적 모델:
θᵢ ~ N(μ_prior, σ²_prior)  # 선수 i의 영향도 사전분포
y_on ~ N(μ + θᵢ, σ²)       # 출전 시 마진
y_off ~ N(μ, σ²)           # 미출전 시 마진
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import logger


@dataclass
class BayesianEstimate:
    """베이지안 영향도 추정치"""
    player_id: int
    player_name: str
    team_id: int

    # 사전 분포 (EPM 기반)
    prior_mean: float
    prior_std: float

    # 우도 (데이터)
    likelihood_mean: float
    likelihood_std: float
    n_games_on: int
    n_games_off: int

    # 사후 분포
    posterior_mean: float
    posterior_std: float

    # 신뢰구간 (HPD)
    hpd_95_lower: float
    hpd_95_upper: float

    # 확률
    prob_positive: float  # P(θ > 0)
    prob_large_effect: float  # P(|θ| > 3)

    # 불확실성 감소
    uncertainty_reduction: float  # 1 - posterior_std / prior_std


class BayesianPlayerImpactModel:
    """
    베이지안 계층 모델을 사용한 선수 영향도 추정.

    EPM 기반 사전 정보와 실제 On/Off 데이터를 결합하여
    불확실성을 포함한 영향도를 추정합니다.
    """

    # 기본 하이퍼파라미터
    DEFAULT_PRIOR_STD = 5.0  # EPM 기반 사전분포의 표준편차
    DEFAULT_LIKELIHOOD_STD = 12.0  # 경기 마진의 표준편차 (관찰)
    EPM_TO_MARGIN_SCALE = 0.67  # EPM을 마진으로 변환하는 스케일

    def __init__(
        self,
        prior_std: float = DEFAULT_PRIOR_STD,
        likelihood_std: float = DEFAULT_LIKELIHOOD_STD,
        epm_scale: float = EPM_TO_MARGIN_SCALE,
        min_games: int = 5
    ):
        """
        Args:
            prior_std: 사전분포 표준편차
            likelihood_std: 우도 표준편차
            epm_scale: EPM -> 마진 변환 스케일
            min_games: 베이지안 업데이트를 위한 최소 경기 수
        """
        self.prior_std = prior_std
        self.likelihood_std = likelihood_std
        self.epm_scale = epm_scale
        self.min_games = min_games

        self._estimates: Dict[int, BayesianEstimate] = {}

    def calculate_prior(
        self,
        player_epm: float,
        mpg: float = 30.0
    ) -> Tuple[float, float]:
        """
        EPM 기반 사전분포 계산.

        Args:
            player_epm: 선수 EPM (Tot)
            mpg: 평균 출전 시간

        Returns:
            (prior_mean, prior_std)
        """
        # EPM을 예상 마진 영향으로 변환
        # EPM = 48분 기준 순 기여도
        # 실제 출전 시간 비율 고려
        minutes_ratio = mpg / 48.0

        # 선수 부재 시 예상 마진 변화
        # 양의 EPM 선수가 빠지면 팀에 불리 → 음의 영향
        prior_mean = -player_epm * self.epm_scale * minutes_ratio

        # 불확실성은 EPM의 절대값에 비례 (스타 선수일수록 변동성 큼)
        prior_std = self.prior_std * (1 + 0.1 * abs(player_epm))

        return prior_mean, prior_std

    def calculate_likelihood(
        self,
        margins_on: List[float],
        margins_off: List[float]
    ) -> Tuple[float, float, int, int]:
        """
        관찰 데이터 기반 우도 계산.

        Args:
            margins_on: 출전 경기 마진 리스트
            margins_off: 미출전 경기 마진 리스트

        Returns:
            (likelihood_mean, likelihood_std, n_on, n_off)
        """
        n_on = len(margins_on)
        n_off = len(margins_off)

        if n_on < 1 or n_off < 1:
            return 0.0, self.likelihood_std, n_on, n_off

        # 관찰된 영향도
        mean_on = np.mean(margins_on)
        mean_off = np.mean(margins_off)
        observed_effect = mean_on - mean_off

        # 표본 표준오차
        # SE = sqrt(var_on/n_on + var_off/n_off)
        var_on = np.var(margins_on, ddof=1) if n_on > 1 else self.likelihood_std ** 2
        var_off = np.var(margins_off, ddof=1) if n_off > 1 else self.likelihood_std ** 2

        se = np.sqrt(var_on / n_on + var_off / n_off)

        return observed_effect, se, n_on, n_off

    def update_posterior(
        self,
        prior_mean: float,
        prior_std: float,
        likelihood_mean: float,
        likelihood_std: float
    ) -> Tuple[float, float]:
        """
        베이지안 업데이트 (정규-정규 켤레).

        Posterior = Prior × Likelihood

        Args:
            prior_mean: 사전 평균
            prior_std: 사전 표준편차
            likelihood_mean: 우도 평균 (관찰된 효과)
            likelihood_std: 우도 표준편차

        Returns:
            (posterior_mean, posterior_std)
        """
        prior_var = prior_std ** 2
        likelihood_var = likelihood_std ** 2

        # 사후 정밀도 = 사전 정밀도 + 우도 정밀도
        posterior_precision = 1 / prior_var + 1 / likelihood_var
        posterior_var = 1 / posterior_precision
        posterior_std = np.sqrt(posterior_var)

        # 사후 평균 = 정밀도 가중 평균
        posterior_mean = (
            prior_mean / prior_var + likelihood_mean / likelihood_var
        ) / posterior_precision

        return posterior_mean, posterior_std

    def estimate_player(
        self,
        player_id: int,
        player_name: str,
        team_id: int,
        player_epm: float,
        mpg: float,
        margins_on: List[float],
        margins_off: List[float]
    ) -> BayesianEstimate:
        """
        단일 선수 베이지안 추정.

        Args:
            player_id: 선수 ID
            player_name: 선수 이름
            team_id: 팀 ID
            player_epm: EPM (Tot)
            mpg: 평균 출전 시간
            margins_on: 출전 경기 마진
            margins_off: 미출전 경기 마진

        Returns:
            BayesianEstimate
        """
        # 사전분포
        prior_mean, prior_std = self.calculate_prior(player_epm, mpg)

        # 우도
        likelihood_mean, likelihood_std, n_on, n_off = self.calculate_likelihood(
            margins_on, margins_off
        )

        # 데이터가 부족하면 사전분포 그대로 사용
        if n_on < self.min_games or n_off < 1:
            posterior_mean = prior_mean
            posterior_std = prior_std
        else:
            # 사후분포
            posterior_mean, posterior_std = self.update_posterior(
                prior_mean, prior_std,
                likelihood_mean, likelihood_std
            )

        # 95% HPD (Highest Posterior Density)
        hpd_lower = posterior_mean - 1.96 * posterior_std
        hpd_upper = posterior_mean + 1.96 * posterior_std

        # 확률 계산
        prob_positive = 1 - stats.norm.cdf(0, posterior_mean, posterior_std)
        prob_large_effect = (
            stats.norm.cdf(-3, posterior_mean, posterior_std) +
            1 - stats.norm.cdf(3, posterior_mean, posterior_std)
        )

        # 불확실성 감소
        uncertainty_reduction = 1 - posterior_std / prior_std

        estimate = BayesianEstimate(
            player_id=player_id,
            player_name=player_name,
            team_id=team_id,
            prior_mean=prior_mean,
            prior_std=prior_std,
            likelihood_mean=likelihood_mean,
            likelihood_std=likelihood_std,
            n_games_on=n_on,
            n_games_off=n_off,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            hpd_95_lower=hpd_lower,
            hpd_95_upper=hpd_upper,
            prob_positive=prob_positive,
            prob_large_effect=prob_large_effect,
            uncertainty_reduction=uncertainty_reduction,
        )

        self._estimates[player_id] = estimate

        return estimate

    def fit(
        self,
        player_epm_df: pd.DataFrame,
        on_off_results: pd.DataFrame
    ) -> pd.DataFrame:
        """
        전체 선수 베이지안 추정.

        Args:
            player_epm_df: 선수 EPM DataFrame (player_id, tot, mpg, player_name, team_id)
            on_off_results: On/Off 분석 결과 DataFrame (player_id, margins_on, margins_off)
                          또는 (player_id, margin_on, margin_off, games_on, games_off)

        Returns:
            추정 결과 DataFrame
        """
        results = []

        for _, player_row in player_epm_df.iterrows():
            player_id = player_row["player_id"]

            # EPM 정보
            epm = player_row.get("tot", player_row.get("epm", 0))
            mpg = player_row.get("mpg", 30)
            player_name = player_row.get("player_name", f"Player_{player_id}")
            team_id = player_row.get("team_id", 0)

            # On/Off 데이터 조회
            on_off_row = on_off_results[on_off_results["player_id"] == player_id]

            if on_off_row.empty:
                # On/Off 데이터 없으면 EPM만으로 추정
                margins_on = []
                margins_off = []
            else:
                on_off_row = on_off_row.iloc[0]

                # margins_on/off 직접 제공된 경우
                if "margins_on" in on_off_row and isinstance(on_off_row["margins_on"], list):
                    margins_on = on_off_row["margins_on"]
                    margins_off = on_off_row["margins_off"]
                else:
                    # 평균값과 경기 수만 있는 경우 - 시뮬레이션
                    n_on = int(on_off_row.get("games_on", 0))
                    n_off = int(on_off_row.get("games_off", 0))
                    mean_on = on_off_row.get("margin_on", 0)
                    mean_off = on_off_row.get("margin_off", 0)

                    # 경기별 마진 시뮬레이션 (평균과 표준 NBA 분산 사용)
                    if n_on > 0:
                        margins_on = list(np.random.normal(mean_on, 12, n_on))
                    else:
                        margins_on = []

                    if n_off > 0:
                        margins_off = list(np.random.normal(mean_off, 12, n_off))
                    else:
                        margins_off = []

            estimate = self.estimate_player(
                player_id=player_id,
                player_name=player_name,
                team_id=team_id,
                player_epm=epm,
                mpg=mpg,
                margins_on=margins_on,
                margins_off=margins_off
            )

            results.append(estimate)

        # DataFrame 변환
        df = pd.DataFrame([
            {
                "player_id": e.player_id,
                "player_name": e.player_name,
                "team_id": e.team_id,
                "prior_mean": e.prior_mean,
                "prior_std": e.prior_std,
                "likelihood_mean": e.likelihood_mean,
                "likelihood_std": e.likelihood_std,
                "n_games_on": e.n_games_on,
                "n_games_off": e.n_games_off,
                "posterior_mean": e.posterior_mean,
                "posterior_std": e.posterior_std,
                "hpd_95_lower": e.hpd_95_lower,
                "hpd_95_upper": e.hpd_95_upper,
                "prob_positive": e.prob_positive,
                "prob_large_effect": e.prob_large_effect,
                "uncertainty_reduction": e.uncertainty_reduction,
            }
            for e in results
        ])

        logger.info(f"Completed Bayesian estimation for {len(df)} players")

        return df.sort_values("posterior_mean")  # 부재 시 불리한 순

    def get_estimate(self, player_id: int) -> Optional[BayesianEstimate]:
        """선수 추정치 조회"""
        return self._estimates.get(player_id)

    def get_impact(self, player_id: int) -> float:
        """선수 영향도 (posterior mean) 반환"""
        estimate = self._estimates.get(player_id)
        if estimate:
            return estimate.posterior_mean
        return 0.0

    def get_uncertainty(self, player_id: int) -> float:
        """선수 불확실성 (posterior std) 반환"""
        estimate = self._estimates.get(player_id)
        if estimate:
            return estimate.posterior_std
        return self.prior_std

    def save_results(self, output_path: Path, results_df: pd.DataFrame):
        """결과 저장"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(output_path, index=False)
        logger.info(f"Saved Bayesian estimates to {output_path}")

    @classmethod
    def load_results(cls, path: Path) -> pd.DataFrame:
        """저장된 결과 로드"""
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)


def create_bayesian_model_from_data(
    data_dir: Path,
    season: int,
    min_games: int = 10
) -> Tuple[BayesianPlayerImpactModel, pd.DataFrame]:
    """
    데이터에서 베이지안 모델 생성 및 적합.

    Args:
        data_dir: 데이터 디렉토리
        season: 시즌 연도
        min_games: 최소 경기 수

    Returns:
        (BayesianPlayerImpactModel, results_df)
    """
    # EPM 데이터 로드
    epm_path = data_dir / "raw" / "dnt" / "season_epm" / f"season_{season}.parquet"
    if not epm_path.exists():
        raise FileNotFoundError(f"Player EPM not found: {epm_path}")

    player_epm_df = pd.read_parquet(epm_path)

    # On/Off 분석 결과 로드
    on_off_path = data_dir / "processed" / "player_impact" / f"season_{season}.parquet"
    if on_off_path.exists():
        on_off_df = pd.read_parquet(on_off_path)
    else:
        on_off_df = pd.DataFrame()
        logger.warning(f"On/Off results not found: {on_off_path}")

    # 모델 생성 및 적합
    model = BayesianPlayerImpactModel(min_games=min_games)
    results_df = model.fit(player_epm_df, on_off_df)

    # 결과 저장
    output_path = data_dir / "processed" / "bayesian_impact" / f"season_{season}.parquet"
    model.save_results(output_path, results_df)

    return model, results_df
