from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from urllib.error import URLError

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/CI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CLASS_NAMES: Tuple[str, str, str] = ("Home Win (1)", "Draw (X)", "Away Win (2)")
TARGET_MAP: Dict[str, int] = {"H": 0, "D": 1, "A": 2}


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for Ligue 1 data ingestion, features, and evaluation."""

    seasons: Tuple[str, ...] = ("2021", "2122", "2223", "2324", "2425", "2526", "2627")
    base_url: str = "https://www.football-data.co.uk/mmz4281/{season}/F1.csv"
    warmup_season: str = "2021"
    train_ratio: float = 0.80
    random_state: int = 42

    # Elo & rolling form parameters
    initial_elo: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 65.0
    rolling_window: int = 5
    prior_goals: float = 1.3
    prior_sot: float = 4.0

    # Feature sets
    base_features: Tuple[str, ...] = ("elo_diff", "elo_home", "elo_away")
    exp_features: Tuple[str, ...] = (
        "elo_diff",
        "elo_home",
        "elo_away",
        "form_goals_diff",
        "form_sot_diff",
    )

    # Output artifacts
    reports_dir: Path = field(default_factory=lambda: Path("reports"))

    @property
    def results_table_path(self) -> Path:
        return self.reports_dir / "results_table.md"

    @property
    def shap_plot_path(self) -> Path:
        return self.reports_dir / "shap_summary.png"

    @property
    def lime_plot_path(self) -> Path:
        return self.reports_dir / "lime_worst_error.png"


@dataclass(frozen=True)
class ExperimentResult:
    """Structured container for model evaluation metrics."""

    model_name: str
    feature_summary: str
    train_size: int
    test_size: int
    test_log_loss: float
    is_selected: bool = False

    def to_markdown_row(self) -> str:
        name = f"**{self.model_name}**" if self.is_selected else self.model_name
        return (
            f"| {name} | {self.feature_summary} | "
            f"{self.train_size} | {self.test_size} | **{self.test_log_loss:.4f}** |"
        )


class Ligue1DataLoader:
    """Downloads, validates, and chronologically sorts Ligue 1 match records."""

    REQUIRED_COLUMNS: Tuple[str, ...] = ("Date", "HomeTeam", "AwayTeam", "FTR")

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def load(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for season in self.config.seasons:
            url = self.config.base_url.format(season=season)
            try:
                df = pd.read_csv(url)
                df["Season"] = season
                frames.append(df)
            except (URLError, pd.errors.EmptyDataError, OSError) as exc:
                logger.warning("Skipping season %s (%s): %s", season, url, exc)

        if not frames:
            raise RuntimeError(
                "Failed to fetch any Ligue 1 CSV datasets from football-data.co.uk."
            )

        raw = pd.concat(frames, ignore_index=True)
        missing_cols = set(self.REQUIRED_COLUMNS) - set(raw.columns)
        if missing_cols:
            raise ValueError(f"Dataset missing required columns: {sorted(missing_cols)}")

        cleaned = raw.dropna(subset=list(self.REQUIRED_COLUMNS)).copy()
        cleaned["Date"] = pd.to_datetime(
            cleaned["Date"], dayfirst=True, format="mixed", errors="coerce"
        )
        cleaned["target"] = cleaned["FTR"].map(TARGET_MAP)

        cleaned = (
            cleaned.dropna(subset=["Date", "target"])
            .sort_values("Date", kind="mergesort")
            .reset_index(drop=True)
        )
        cleaned["target"] = cleaned["target"].astype(np.int64)
        logger.info("Loaded %d completed Ligue 1 matches across %d seasons.", len(cleaned), len(frames))
        return cleaned


class SequentialFeatureEngine:
    """Computes leak-free pre-match Elo ratings and rolling team form metrics."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        elo: Dict[str, float] = {}
        gf_hist: Dict[str, deque[float]] = {}
        ga_hist: Dict[str, deque[float]] = {}
        sot_hist: Dict[str, deque[float]] = {}

        # Vectorized fallback imputation prior to sequential iteration
        fthg_arr = pd.to_numeric(df.get("FTHG", 0.0), errors="coerce").fillna(0.0).to_numpy()
        ftag_arr = pd.to_numeric(df.get("FTAG", 0.0), errors="coerce").fillna(0.0).to_numpy()
        hst_arr = (
            pd.to_numeric(df.get("HST", self.config.prior_sot), errors="coerce")
            .fillna(self.config.prior_sot)
            .to_numpy()
        )
        ast_arr = (
            pd.to_numeric(df.get("AST", self.config.prior_sot), errors="coerce")
            .fillna(self.config.prior_sot)
            .to_numpy()
        )

        records: List[Dict[str, object]] = []
        iterator = zip(
            df["Date"].tolist(),
            df["HomeTeam"].tolist(),
            df["AwayTeam"].tolist(),
            df["Season"].tolist(),
            df["FTR"].tolist(),
            df["target"].tolist(),
            fthg_arr,
            ftag_arr,
            hst_arr,
            ast_arr,
        )

        for date, home, away, season, ftr, target, fthg, ftag, hst, ast in iterator:
            elo_h = elo.get(home, self.config.initial_elo)
            elo_a = elo.get(away, self.config.initial_elo)

            h_gf = self._rolling_mean(gf_hist.get(home), self.config.prior_goals)
            a_gf = self._rolling_mean(gf_hist.get(away), self.config.prior_goals)
            h_ga = self._rolling_mean(ga_hist.get(home), self.config.prior_goals)
            a_ga = self._rolling_mean(ga_hist.get(away), self.config.prior_goals)
            h_sot = self._rolling_mean(sot_hist.get(home), self.config.prior_sot)
            a_sot = self._rolling_mean(sot_hist.get(away), self.config.prior_sot)

            records.append(
                {
                    "Date": date,
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "Season": season,
                    "elo_home": elo_h,
                    "elo_away": elo_a,
                    "elo_diff": (elo_h + self.config.home_advantage) - elo_a,
                    "form_goals_diff": (h_gf - h_ga) - (a_gf - a_ga),
                    "form_sot_diff": h_sot - a_sot,
                    "target": target,
                }
            )

            # Post-match Elo rating update
            expected_h = 1.0 / (
                1.0 + 10.0 ** ((elo_a - (elo_h + self.config.home_advantage)) / 400.0)
            )
            actual_h = 1.0 if ftr == "H" else (0.5 if ftr == "D" else 0.0)
            delta = self.config.k_factor * (actual_h - expected_h)
            elo[home] = elo_h + delta
            elo[away] = elo_a - delta

            # Post-match rolling history update
            self._append_stat(gf_hist, home, float(fthg))
            self._append_stat(ga_hist, home, float(ftag))
            self._append_stat(gf_hist, away, float(ftag))
            self._append_stat(ga_hist, away, float(fthg))
            self._append_stat(sot_hist, home, float(hst))
            self._append_stat(sot_hist, away, float(ast))

        return pd.DataFrame.from_records(records)

    @staticmethod
    def _rolling_mean(history: deque[float] | None, default: float) -> float:
        if not history:
            return default
        return float(sum(history) / len(history))

    def _append_stat(self, store: Dict[str, deque[float]], team: str, value: float) -> None:
        if team not in store:
            store[team] = deque(maxlen=self.config.rolling_window)
        store[team].append(value)


class ModelExplainer:
    """Generates global SHAP feature attribution and local LIME error diagnostics."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def generate_all(
        self,
        model: LogisticRegression,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        test_meta: pd.DataFrame,
        y_test: np.ndarray,
        pred_probs: np.ndarray,
    ) -> None:
        self._save_global_shap(model, X_train, X_test)
        self._save_local_lime_worst_error(
            model, X_train, X_test, test_meta, y_test, pred_probs
        )

    def _save_global_shap(
        self,
        model: LogisticRegression,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> None:
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            home_shap = np.asarray(shap_values[0])
        elif getattr(shap_values, "ndim", 0) == 3:
            home_shap = np.asarray(shap_values[:, :, 0])
        else:
            home_shap = np.asarray(shap_values)

        fig = plt.figure(figsize=(8.0, 4.5))
        shap.summary_plot(
            home_shap,
            X_test,
            feature_names=list(X_train.columns),
            show=False,
        )
        plt.title("SHAP Global Feature Impact on Home Win Probability")
        plt.tight_layout()
        plt.savefig(self.config.shap_plot_path, dpi=150)
        plt.close(fig)
        plt.close("all")
        logger.info("Saved global SHAP plot to %s", self.config.shap_plot_path)

    def _save_local_lime_worst_error(
        self,
        model: LogisticRegression,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        test_meta: pd.DataFrame,
        y_test: np.ndarray,
        pred_probs: np.ndarray,
    ) -> None:
        true_class_probs = pred_probs[np.arange(len(y_test)), y_test]
        worst_idx = int(np.argmin(true_class_probs))
        pred_cls = int(np.argmax(pred_probs[worst_idx]))
        worst_row = test_meta.iloc[worst_idx]

        lime_explainer = LimeTabularExplainer(
            training_data=X_train.to_numpy(),
            feature_names=list(X_train.columns),
            class_names=list(CLASS_NAMES),
            mode="classification",
            random_state=self.config.random_state,
        )
        explanation = lime_explainer.explain_instance(
            data_row=X_test.iloc[worst_idx].to_numpy(),
            predict_fn=model.predict_proba,
            num_features=min(5, X_train.shape[1]),
            top_labels=len(CLASS_NAMES),
        )

        fig = explanation.as_pyplot_figure(label=pred_cls)
        match_date = pd.Timestamp(worst_row["Date"]).strftime("%Y-%m-%d")
        plt.title(
            f"LIME Error Case: {worst_row['HomeTeam']} vs {worst_row['AwayTeam']} ({match_date})\n"
            f"Predicted: {CLASS_NAMES[pred_cls]} | Actual: {CLASS_NAMES[y_test[worst_idx]]}"
        )
        plt.tight_layout()
        plt.savefig(self.config.lime_plot_path, dpi=150)
        plt.close(fig)
        plt.close("all")

        logger.info(
            "Worst prediction: %s vs %s (%s) | probs=%s | actual=%s",
            worst_row["HomeTeam"],
            worst_row["AwayTeam"],
            match_date,
            np.round(pred_probs[worst_idx], 3).tolist(),
            CLASS_NAMES[y_test[worst_idx]],
        )
        logger.info("Saved local LIME error plot to %s", self.config.lime_plot_path)


def render_results_markdown(results: Sequence[ExperimentResult]) -> str:
    header = (
        "| Model | Features | Train Matches | Test Matches | Test Log Loss |\n"
        "| :--- | :--- | :---: | :---: | :---: |\n"
    )
    rows = "\n".join(res.to_markdown_row() for res in results)
    return f"{header}{rows}\n"


def run_pipeline(config: PipelineConfig | None = None) -> None:
    cfg = config or PipelineConfig()
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    raw_df = Ligue1DataLoader(cfg).load()
    feat_df = SequentialFeatureEngine(cfg).transform(raw_df)

    # Exclude warm-up season so Elo ratings converge before model training
    model_df = feat_df[feat_df["Season"] != cfg.warmup_season].reset_index(drop=True)
    split_idx = int(len(model_df) * cfg.train_ratio)

    train_df = model_df.iloc[:split_idx].reset_index(drop=True)
    test_df = model_df.iloc[split_idx:].reset_index(drop=True)
    y_train = train_df["target"].to_numpy()
    y_test = test_df["target"].to_numpy()

    # 1. Naive Baseline (empirical class priors)
    priors = np.bincount(y_train, minlength=len(CLASS_NAMES)) / len(y_train)
    naive_probs = np.tile(priors, (len(y_test), 1))
    naive_ll = float(log_loss(y_test, naive_probs, labels=[0, 1, 2]))

    # 2. Main Baseline: Elo + Multinomial Logistic Regression
    base_cols = list(cfg.base_features)
    X_train_base = train_df[base_cols]
    X_test_base = test_df[base_cols]

    logreg = LogisticRegression(max_iter=1000, random_state=cfg.random_state)
    logreg.fit(X_train_base, y_train)
    base_probs = logreg.predict_proba(X_test_base)
    base_ll = float(log_loss(y_test, base_probs, labels=[0, 1, 2]))

    # 3. Week 3 Experiment: Elo + 5-match Rolling Form + Gradient Boosting
    exp_cols = list(cfg.exp_features)
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=cfg.random_state,
    )
    gb_model.fit(train_df[exp_cols], y_train)
    exp_probs = gb_model.predict_proba(test_df[exp_cols])
    exp_ll = float(log_loss(y_test, exp_probs, labels=[0, 1, 2]))

    results = [
        ExperimentResult(
            model_name="Naive Baseline (Class Priors)",
            feature_summary="Historical H/D/A frequencies",
            train_size=len(train_df),
            test_size=len(test_df),
            test_log_loss=naive_ll,
        ),
        ExperimentResult(
            model_name="Baseline: Elo + Logistic Regression",
            feature_summary="`elo_diff`, `elo_home`, `elo_away`",
            train_size=len(train_df),
            test_size=len(test_df),
            test_log_loss=base_ll,
            is_selected=True,
        ),
        ExperimentResult(
            model_name="Experiment: Elo + Form + Gradient Boosting",
            feature_summary="Elo + 5-match rolling goals & shots on target",
            train_size=len(train_df),
            test_size=len(test_df),
            test_log_loss=exp_ll,
        ),
    ]

    table_md = render_results_markdown(results)
    cfg.results_table_path.write_text(table_md, encoding="utf-8")
    print("\n=== EXPERIMENT RESULTS (LIGUE 1) ===")
    print(table_md)

    # 4. Generate SHAP & LIME explainability artifacts for the selected production model
    ModelExplainer(cfg).generate_all(
        model=logreg,
        X_train=X_train_base,
        X_test=X_test_base,
        test_meta=test_df,
        y_test=y_test,
        pred_probs=base_probs,
    )


if __name__ == "__main__":
    run_pipeline()