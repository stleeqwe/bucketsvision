"""
Mixed Effects Model for Player Impact.

팀 효과, 상대팀 효과를 랜덤 효과로 모델링하여
더 정밀한 선수 영향도를 추정합니다.

모델:
margin_i = β₀ + Σⱼ(player_out_jᵢ × βⱼ) + γ_team + γ_opponent + ε_i

- β₀: 절편 (홈 어드밴티지 포함)
- βⱼ: 선수 j의 부재 효과 (고정 효과)
- γ_team: 팀 랜덤 효과
- γ_opponent: 상대팀 랜덤 효과
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from src.utils.logger import logger

# statsmodels는 선택적 의존성
try:
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not available, using Ridge approximation")


@dataclass
class PlayerEffect:
    """선수별 효과 추정치"""
    player_id: int
    player_name: str
    team_id: int
    effect: float  # 부재 시 예상 마진 변화
    std_error: float
    z_score: float
    p_value: float
    ci_lower: float
    ci_upper: float


@dataclass
class ModelFitResult:
    """모델 적합 결과"""
    n_observations: int
    n_players: int
    n_teams: int

    # 모델 적합도
    log_likelihood: float
    aic: float
    bic: float

    # 분산 성분
    team_variance: float
    opponent_variance: float
    residual_variance: float

    # ICC (Intraclass Correlation)
    icc_team: float
    icc_opponent: float

    # 선수 효과
    player_effects: Dict[int, PlayerEffect]


class MixedEffectsPlayerModel:
    """
    Mixed Effects 모델을 사용한 선수 영향도 추정.

    팀 효과와 상대팀 효과를 통제하면서
    개별 선수의 부재 효과를 추정합니다.
    """

    def __init__(
        self,
        regularization: float = 1.0,
        min_games_for_effect: int = 5
    ):
        """
        Args:
            regularization: 정규화 강도 (Ridge)
            min_games_for_effect: 효과 추정을 위한 최소 경기 수
        """
        self.regularization = regularization
        self.min_games_for_effect = min_games_for_effect

        self._is_fitted = False
        self._player_effects: Dict[int, PlayerEffect] = {}
        self._player_id_to_name: Dict[int, str] = {}
        self._player_id_to_team: Dict[int, int] = {}

    def prepare_data(
        self,
        games_df: pd.DataFrame,
        player_games_df: pd.DataFrame,
        target_players: Optional[List[int]] = None
    ) -> Tuple[pd.DataFrame, List[int]]:
        """
        모델링을 위한 데이터 준비.

        각 경기에 대해:
        - 어떤 선수가 결장했는지 (해당 팀 전체 로스터 대비)
        - 경기 결과 (마진)
        - 팀 ID, 상대팀 ID

        Args:
            games_df: 경기 결과 DataFrame
            player_games_df: 선수 경기 출전 DataFrame
            target_players: 분석 대상 선수 ID 리스트 (None이면 자동 선택)

        Returns:
            (모델 데이터 DataFrame, 분석 대상 선수 ID 리스트)
        """
        # 선수별 경기 수 계산
        player_games_count = (
            player_games_df[player_games_df["played"] == True]
            .groupby("player_id")
            .size()
        )

        # 분석 대상 선수 선정
        if target_players is None:
            target_players = player_games_count[
                player_games_count >= self.min_games_for_effect
            ].index.tolist()

        logger.info(f"Target players: {len(target_players)}")

        # 선수 정보 저장
        for _, row in player_games_df.drop_duplicates("player_id").iterrows():
            pid = row["player_id"]
            self._player_id_to_name[pid] = row.get("player_name", f"Player_{pid}")
            self._player_id_to_team[pid] = row.get("team_id", 0)

        # 경기별 결장 선수 매트릭스 구축
        model_data = []

        for _, game in games_df.iterrows():
            game_id = game["game_id"]
            home_team_id = game["home_team_id"]
            away_team_id = game["away_team_id"]
            margin = game["margin"]

            # 해당 경기 출전 선수
            game_players = player_games_df[
                (player_games_df["game_id"] == game_id) &
                (player_games_df["played"] == True)
            ]

            played_player_ids = set(game_players["player_id"].unique())

            # 각 팀의 잠재적 로스터 (시즌 동안 한 번이라도 출전)
            home_roster = set(
                player_games_df[
                    (player_games_df["team_id"] == home_team_id) &
                    (player_games_df["played"] == True)
                ]["player_id"].unique()
            )

            away_roster = set(
                player_games_df[
                    (player_games_df["team_id"] == away_team_id) &
                    (player_games_df["played"] == True)
                ]["player_id"].unique()
            )

            # 결장 선수 (로스터에 있지만 이 경기 미출전)
            home_out = home_roster - played_player_ids
            away_out = away_roster - played_player_ids

            row_data = {
                "game_id": game_id,
                "margin": margin,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
            }

            # 각 대상 선수의 결장 여부
            for player_id in target_players:
                player_team = self._player_id_to_team.get(player_id, 0)

                if player_team == home_team_id:
                    # 홈팀 선수가 결장 → 마진 감소 예상
                    row_data[f"out_{player_id}"] = 1 if player_id in home_out else 0
                elif player_team == away_team_id:
                    # 원정팀 선수가 결장 → 마진 증가 예상 (홈팀 유리)
                    row_data[f"out_{player_id}"] = -1 if player_id in away_out else 0
                else:
                    row_data[f"out_{player_id}"] = 0

            model_data.append(row_data)

        df = pd.DataFrame(model_data)

        return df, target_players

    def fit(
        self,
        model_data: pd.DataFrame,
        target_players: List[int]
    ) -> ModelFitResult:
        """
        Mixed Effects 모델 적합.

        Args:
            model_data: prepare_data에서 생성된 DataFrame
            target_players: 대상 선수 ID 리스트

        Returns:
            ModelFitResult
        """
        y = model_data["margin"].values

        # 고정 효과 (선수 결장 변수)
        out_cols = [f"out_{pid}" for pid in target_players]
        X_fixed = model_data[out_cols].values

        # 그룹 변수 (팀, 상대팀)
        team_ids = model_data["home_team_id"].values
        opponent_ids = model_data["away_team_id"].values

        if HAS_STATSMODELS:
            result = self._fit_mixed_lm(
                y, X_fixed, team_ids, opponent_ids, target_players, model_data
            )
        else:
            result = self._fit_ridge_approximation(
                y, X_fixed, team_ids, opponent_ids, target_players, model_data
            )

        self._is_fitted = True
        self._player_effects = result.player_effects

        return result

    def _fit_mixed_lm(
        self,
        y: np.ndarray,
        X_fixed: np.ndarray,
        team_ids: np.ndarray,
        opponent_ids: np.ndarray,
        target_players: List[int],
        model_data: pd.DataFrame
    ) -> ModelFitResult:
        """statsmodels MixedLM 사용"""
        # 데이터프레임 구성
        df = model_data.copy()
        df["y"] = y

        out_cols = [f"out_{pid}" for pid in target_players]
        formula = f"y ~ {' + '.join(out_cols)}"

        try:
            # 팀을 랜덤 효과로
            model = MixedLM.from_formula(
                formula,
                data=df,
                groups=df["home_team_id"],
                re_formula="1"
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(method="lbfgs", maxiter=100)

            # 선수 효과 추출
            player_effects = {}
            for i, pid in enumerate(target_players):
                param_name = f"out_{pid}"
                if param_name in result.params:
                    effect = result.params[param_name]
                    std_err = result.bse.get(param_name, 0.1)
                    z_score = effect / std_err if std_err > 0 else 0
                    p_value = result.pvalues.get(param_name, 1.0)

                    player_effects[pid] = PlayerEffect(
                        player_id=pid,
                        player_name=self._player_id_to_name.get(pid, ""),
                        team_id=self._player_id_to_team.get(pid, 0),
                        effect=effect,
                        std_error=std_err,
                        z_score=z_score,
                        p_value=p_value,
                        ci_lower=effect - 1.96 * std_err,
                        ci_upper=effect + 1.96 * std_err,
                    )

            # 분산 성분
            team_var = result.cov_re.iloc[0, 0] if hasattr(result, 'cov_re') else 0
            resid_var = result.scale

            return ModelFitResult(
                n_observations=len(y),
                n_players=len(target_players),
                n_teams=len(np.unique(team_ids)),
                log_likelihood=result.llf,
                aic=result.aic,
                bic=result.bic,
                team_variance=team_var,
                opponent_variance=0,  # 단순 모델에서는 미포함
                residual_variance=resid_var,
                icc_team=team_var / (team_var + resid_var) if (team_var + resid_var) > 0 else 0,
                icc_opponent=0,
                player_effects=player_effects,
            )

        except Exception as e:
            logger.warning(f"MixedLM failed: {e}, falling back to Ridge")
            return self._fit_ridge_approximation(
                y, X_fixed, team_ids, opponent_ids, target_players, model_data
            )

    def _fit_ridge_approximation(
        self,
        y: np.ndarray,
        X_fixed: np.ndarray,
        team_ids: np.ndarray,
        opponent_ids: np.ndarray,
        target_players: List[int],
        model_data: pd.DataFrame
    ) -> ModelFitResult:
        """Ridge 회귀를 사용한 근사"""
        # 팀 더미 변수 추가
        unique_teams = np.unique(np.concatenate([team_ids, opponent_ids]))
        team_dummies = np.zeros((len(y), len(unique_teams)))

        for i, team in enumerate(unique_teams):
            # 홈팀이면 +, 원정팀이면 -
            team_dummies[:, i] = (team_ids == team).astype(float) - (opponent_ids == team).astype(float)

        # 전체 디자인 매트릭스
        X = np.hstack([X_fixed, team_dummies])

        # Ridge 회귀
        model = Ridge(alpha=self.regularization)
        model.fit(X, y)

        # 잔차 분산 추정
        y_pred = model.predict(X)
        residuals = y - y_pred
        resid_var = np.var(residuals, ddof=X.shape[1])

        # 선수 효과 추출
        player_effects = {}
        for i, pid in enumerate(target_players):
            effect = model.coef_[i]

            # Bootstrap으로 표준오차 추정
            std_err = self._bootstrap_se(X_fixed[:, i], y, effect)

            z_score = effect / std_err if std_err > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score))) if std_err > 0 else 1.0

            player_effects[pid] = PlayerEffect(
                player_id=pid,
                player_name=self._player_id_to_name.get(pid, ""),
                team_id=self._player_id_to_team.get(pid, 0),
                effect=effect,
                std_error=std_err,
                z_score=z_score,
                p_value=p_value,
                ci_lower=effect - 1.96 * std_err,
                ci_upper=effect + 1.96 * std_err,
            )

        # 팀 분산 추정 (팀 더미 계수의 분산)
        team_effects = model.coef_[len(target_players):]
        team_var = np.var(team_effects) if len(team_effects) > 0 else 0

        # 모델 적합도
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        n = len(y)
        k = X.shape[1]
        aic = n * np.log(ss_res / n) + 2 * k
        bic = n * np.log(ss_res / n) + k * np.log(n)

        return ModelFitResult(
            n_observations=len(y),
            n_players=len(target_players),
            n_teams=len(unique_teams),
            log_likelihood=-0.5 * n * (1 + np.log(2 * np.pi * ss_res / n)),
            aic=aic,
            bic=bic,
            team_variance=team_var,
            opponent_variance=0,
            residual_variance=resid_var,
            icc_team=team_var / (team_var + resid_var) if (team_var + resid_var) > 0 else 0,
            icc_opponent=0,
            player_effects=player_effects,
        )

    def _bootstrap_se(
        self,
        x: np.ndarray,
        y: np.ndarray,
        observed_effect: float,
        n_bootstrap: int = 500
    ) -> float:
        """Bootstrap 표준오차 추정"""
        from scipy import stats

        n = len(y)
        boot_effects = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            x_boot = x[idx]
            y_boot = y[idx]

            # 단순 회귀 계수
            cov_xy = np.sum((x_boot - np.mean(x_boot)) * (y_boot - np.mean(y_boot)))
            var_x = np.sum((x_boot - np.mean(x_boot)) ** 2)

            if var_x > 0:
                boot_effects.append(cov_xy / var_x)

        if len(boot_effects) > 10:
            return np.std(boot_effects)
        else:
            return abs(observed_effect) * 0.5  # Fallback

    def get_player_effect(self, player_id: int) -> Optional[PlayerEffect]:
        """선수 효과 조회"""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self._player_effects.get(player_id)

    def get_all_effects_df(self) -> pd.DataFrame:
        """전체 선수 효과 DataFrame"""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        records = []
        for pid, effect in self._player_effects.items():
            records.append({
                "player_id": effect.player_id,
                "player_name": effect.player_name,
                "team_id": effect.team_id,
                "effect": effect.effect,
                "std_error": effect.std_error,
                "z_score": effect.z_score,
                "p_value": effect.p_value,
                "ci_lower": effect.ci_lower,
                "ci_upper": effect.ci_upper,
                "is_significant": effect.p_value < 0.05,
            })

        df = pd.DataFrame(records)
        return df.sort_values("effect", ascending=True)  # 음수가 큰 것 = 부재 시 팀에 불리

    def save_effects(self, output_path: Path):
        """효과 저장"""
        df = self.get_all_effects_df()
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(df)} player effects to {output_path}")


# scipy stats import for bootstrap_se
from scipy import stats
