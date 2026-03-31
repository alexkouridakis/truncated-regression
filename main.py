# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from collections import OrderedDict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


Interval = Tuple[float, float]
ALL_METHODS: Tuple[str, ...] = ("ols", "wrong_set", "true_set", "full")
METHOD_LABELS: Dict[str, str] = {
    "ols": "OLS (no correction)",
    "wrong_set": "PSGD with wrong S",
    "true_set": "PSGD with true S*",
    "full": "Full algorithm",
}

def normalize_intervals(intervals: Sequence[Interval]) -> List[Interval]:
    # Sorts sequence of intervals, removing any invalid ones
    cleaned = sorted((float(a), float(b)) for a, b in intervals if a <= b)
    if not cleaned:
        return []
    merged: List[Interval] = [cleaned[0]]
    for a, b in cleaned[1:]:
        prev_a, prev_b = merged[-1]
        if a <= prev_b:
            merged[-1] = (prev_a, max(prev_b, b))
        else:
            merged.append((a, b))
    return merged


def shift_intervals(intervals: Sequence[Interval], shift: float) -> List[Interval]:
    return normalize_intervals([(a + shift, b + shift) for a, b in intervals])


def in_intervals(y: np.ndarray | float, intervals: Sequence[Interval]) -> np.ndarray:
    y_arr = np.asarray(y)
    mask = np.zeros_like(y_arr, dtype=bool)
    for a, b in intervals:
        mask |= (a <= y_arr) & (y_arr <= b)
    return mask


def truncated_gaussian_normalization(
    intervals: Sequence[Interval],
    loc: np.ndarray | float,
    scale: float = 1.0,
    eps: float = 1e-15,
) -> np.ndarray | float:
    loc_arr = np.asarray(loc, dtype=float)
    mass = np.zeros_like(loc_arr, dtype=float)
    for a, b in normalize_intervals(intervals):
        mass += norm.cdf((b - loc_arr) / scale) - norm.cdf((a - loc_arr) / scale)
    mass = np.maximum(mass, eps)
    if np.ndim(loc) == 0:
        return float(mass)
    return mass


def truncated_gaussian_mean(
    intervals: Sequence[Interval],
    loc: np.ndarray | float,
    scale: float = 1.0,
    eps: float = 1e-15,
) -> np.ndarray | float:
    """Exact E[Z | Z in S] for Z ~ N(loc, scale^2)."""
    loc_arr = np.asarray(loc, dtype=float)
    denom = truncated_gaussian_normalization(intervals, loc_arr, scale=scale, eps=eps)
    numer = np.zeros_like(loc_arr, dtype=float)
    for a, b in normalize_intervals(intervals):
        alpha = (a - loc_arr) / scale
        beta = (b - loc_arr) / scale
        numer += norm.pdf(alpha) - norm.pdf(beta)
    out = loc_arr + scale * numer / denom
    if np.ndim(loc) == 0:
        return float(out)
    return out


def truncated_gaussian_sampler(
    rng: np.random.Generator,
    intervals: Sequence[Interval],
    loc: float,
    scale: float = 1.0,
) -> float:
    """Correct multi-interval truncated Gaussian sampler."""
    intervals = normalize_intervals(intervals)
    masses = []
    for a, b in intervals:
        mass = norm.cdf((b - loc) / scale) - norm.cdf((a - loc) / scale)
        masses.append(max(mass, 0.0))
    masses = np.asarray(masses, dtype=float)
    total_mass = masses.sum()
    if total_mass <= 0:
        raise ValueError("Truncated Gaussian has essentially zero mass on the supplied intervals.")
    interval_idx = int(rng.choice(len(intervals), p=masses / total_mass))
    a, b = intervals[interval_idx]
    cdf_a = norm.cdf((a - loc) / scale)
    cdf_b = norm.cdf((b - loc) / scale)
    u = np.clip(rng.uniform(cdf_a, cdf_b), 1e-15, 1 - 1e-15)
    return float(loc + scale * norm.ppf(u))


@dataclass
class TruncatedRegressionProblem:
    w_star: np.ndarray
    truncation_intervals: List[Interval]
    feature_weights: np.ndarray
    feature_means: np.ndarray
    feature_covariance: np.ndarray
    noise_std: float = 1.0

    def __post_init__(self) -> None:
        self.w_star = np.asarray(self.w_star, dtype=float)
        self.feature_covariance = np.asarray(self.feature_covariance, dtype=float)
        self.truncation_intervals = normalize_intervals(self.truncation_intervals)
        if any([mean.shape != self.w_star.shape for mean in self.feature_means]):
            raise ValueError("Dimensions of elements of feature_means must match w_star dimension")
        if len(self.feature_weights) != len(self.feature_means):
            print(self.feature_means)
            raise ValueError("Number of elements of feature_weights must match feature_means")
        if self.feature_weights.sum() != 1.0:
            raise ValueError("feature_weights must sum to 1")
        if self.feature_covariance.shape[0] != self.feature_covariance.shape[1]:
            raise ValueError("feature_covariance must be square.")
        if self.feature_covariance.shape[0] != self.w_star.shape[0]:
            raise ValueError("feature_covariance dimension must match w_star dimension.")
        if self.noise_std <= 0:
            raise ValueError("noise_std must be positive.")
        self._cov_chol = np.linalg.cholesky(self.feature_covariance)

    @property
    def d(self) -> int:
        return int(self.w_star.shape[0])

    def make_simulator(self, seed: int) -> "TruncatedRegressionSimulator":
        return TruncatedRegressionSimulator(problem=self, seed=seed)


class TruncatedRegressionSimulator:
    def __init__(self, problem: TruncatedRegressionProblem, seed: int) -> None:
        self.problem = problem
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def sample_features(self, n_samples: int) -> np.ndarray:
        idx_arr = self.rng.choice(
            len(self.problem.feature_means), 
            p=self.problem.feature_weights, 
            size=n_samples
        )
        loc_counter = Counter(idx_arr)
        samples_list = []
        for idx, count in loc_counter.items():
            samples_list.append(
                self.rng.normal(
                    loc=self.problem.feature_means[idx], 
                    size=(count, self.problem.d)
                )
            )
        return np.concatenate(samples_list, axis=0) @ self.problem._cov_chol.T

    def sample_truncated(self, n_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        n_survived = 0
        while n_survived < n_samples:
            remaining = n_samples - n_survived
            n_generated = min(max(10 * remaining, 1_000), 100_000)
            X = self.sample_features(n_generated)
            y = X @ self.problem.w_star + self.problem.noise_std * self.rng.normal(size=n_generated)
            mask = in_intervals(y, self.problem.truncation_intervals)
            if np.any(mask):
                X_list.append(X[mask])
                y_list.append(y[mask])
                n_survived += int(mask.sum())
        X_sampled = np.concatenate(X_list, axis=0)[:n_samples]
        y_sampled = np.concatenate(y_list, axis=0)[:n_samples]
        return X_sampled, y_sampled

    def estimate_survival_probability(self, n_samples: int = 200_000) -> float:
        X = self.sample_features(n_samples)
        y = X @ self.problem.w_star + self.problem.noise_std * self.rng.normal(size=n_samples)
        return float(in_intervals(y, self.problem.truncation_intervals).mean())

    def find_warm_start(self, n_samples: int = 5_000, ridge: float = 1e-8) -> np.ndarray:
        """OLS warm start with solve(...) instead of an explicit inverse."""
        X, y = self.sample_truncated(n_samples)
        gram = X.T @ X
        rhs = X.T @ y
        try:
            return np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), rhs)

    def learn_truncation_set(
        self,
        w_hat: np.ndarray,
        k: int,
        n_samples: int = 5_000,
        r: Optional[int] = None,
        r_scale: float = 1.0,
        max_removed_gaps: Optional[int] = None,
        min_interval_width: float = 1e-8,
    ) -> List[Interval]:
        """Approximate set learning via gap counting."""
        _, y_pos = self.sample_truncated(n_samples)
        y_pos = np.sort(np.asarray(y_pos, dtype=float))
        X_obs, _ = self.sample_truncated(n_samples)
        y_aux = X_obs @ np.asarray(w_hat, dtype=float) + self.problem.noise_std * self.rng.normal(
            size=n_samples
        )
        y_aux = np.sort(np.asarray(y_aux, dtype=float))
        if len(y_pos) == 0:
            raise ValueError("No positive samples were observed while learning the truncation set.")
        if len(y_pos) == 1:
            value = float(y_pos[0])
            return [(value, value)]
        if k <= 1:
            return [(float(y_pos[0]), float(y_pos[-1]))]
        left = np.searchsorted(y_aux, y_pos[:-1], side="right")
        right = np.searchsorted(y_aux, y_pos[1:], side="left")
        gap_counts = right - left
        if r is None:
            r = int(round(r_scale * (k - 1) * math.sqrt(n_samples / max(k, 1))))
            r = max(k - 1, r)
        if max_removed_gaps is not None:
            r = min(r, max_removed_gaps)
        r = min(r, len(gap_counts))
        if r <= 0:
            return [(float(y_pos[0]), float(y_pos[-1]))]
        remove_mask = np.zeros(len(gap_counts), dtype=bool)
        largest = np.argpartition(gap_counts, -r)[-r:]
        remove_mask[largest] = True
        intervals: List[Interval] = []
        start = float(y_pos[0])
        for i, remove_gap in enumerate(remove_mask):
            if remove_gap:
                end = float(y_pos[i])
                if end - start > min_interval_width:
                    intervals.append((start, end))
                start = float(y_pos[i + 1])
        end = float(y_pos[-1])
        if end - start > min_interval_width:
            intervals.append((start, end))
        intervals = normalize_intervals(intervals)
        if not intervals:
            intervals = [(float(y_pos[0]), float(y_pos[-1]))]
        return intervals

    def gradient_sampler(
        self,
        w: np.ndarray,
        intervals: Sequence[Interval],
        batch_size: int = 64,
        use_conditional_mean: bool = True,
    ) -> np.ndarray:
        """Minibatched gradient estimate for the perturbed objective."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        X_hat, y_hat = self.sample_truncated(batch_size)
        X_tilde, _ = self.sample_truncated(batch_size)
        mu_tilde = X_tilde @ np.asarray(w, dtype=float)
        if use_conditional_mean:
            z_tilde = np.asarray(
                truncated_gaussian_mean(intervals, mu_tilde, scale=self.problem.noise_std),
                dtype=float,
            )
        else:
            z_tilde = np.array(
                [
                    truncated_gaussian_sampler(
                        self.rng,
                        intervals,
                        loc=float(mu),
                        scale=self.problem.noise_std,
                    )
                    for mu in mu_tilde
                ],
                dtype=float,
            )
        grads = z_tilde[:, None] * X_tilde - y_hat[:, None] * X_hat
        return grads.mean(axis=0)

    def psgd(
        self,
        w_init: np.ndarray,
        intervals: Sequence[Interval],
        config: "PSGDConfig",
        reference_w: Optional[np.ndarray] = None,
        verbose_label: str = "",
    ) -> "PSGDTrace":
        """Projected SGD returning the **last iterate**."""
        if config.radius <= 0:
            raise ValueError("radius must be positive.")
        if config.T <= 0:
            raise ValueError("T must be positive.")
        w = np.asarray(w_init, dtype=float).copy()
        center = np.asarray(w_init, dtype=float).copy()
        ref = None if reference_w is None else np.asarray(reference_w, dtype=float)
        errors = None if ref is None else np.empty(config.T + 1, dtype=float)
        if errors is not None:
            errors[0] = np.linalg.norm(w - ref)
        step_sizes = np.empty(config.T, dtype=float)
        grad_norms = np.empty(config.T, dtype=float)
        projection_hit_rates = np.empty(config.T, dtype=float)
        projection_hits = 0
        for t in range(1, config.T + 1):
            eta = config.step_size_at(t)
            grad = self.gradient_sampler(
                w,
                intervals,
                batch_size=config.batch_size,
                use_conditional_mean=config.use_conditional_mean,
            )
            grad_norm = float(np.linalg.norm(grad))
            if config.grad_clip is not None and grad_norm > config.grad_clip and grad_norm > 0:
                grad = grad * (config.grad_clip / grad_norm)
            w = w - eta * grad
            diff = w - center
            diff_norm = float(np.linalg.norm(diff))
            if diff_norm > config.radius:
                projection_hits += 1
                w = center + (config.radius / diff_norm) * diff
            step_sizes[t - 1] = eta
            grad_norms[t - 1] = grad_norm
            projection_hit_rates[t - 1] = projection_hits / t
            if errors is not None:
                errors[t] = np.linalg.norm(w - ref)
            if config.verbose_every and t % config.verbose_every == 0:
                prefix = f"[{verbose_label}] " if verbose_label else ""
                status = (
                    f"{prefix}t={t:6d} | eta={eta:.4g} | grad_norm={grad_norm:.4f} | "
                    f"proj_hit_rate={projection_hits / t:.3f}"
                )
                if errors is not None:
                    status += f" | error={errors[t]:.6f}"
                print(status)
        return PSGDTrace(
            w_last=w,
            error_trajectory=errors,
            step_sizes=step_sizes,
            grad_norms=grad_norms,
            projection_hit_rate_trajectory=projection_hit_rates,
        )


@dataclass
class PSGDConfig:
    radius: float = 1.5
    T: int = 1_000
    batch_size: int = 64
    step0: float = 0.05
    step_schedule: str = "inverse_sqrt"
    grad_clip: Optional[float] = 10.0
    use_conditional_mean: bool = True
    verbose_every: int = 0

    def step_size_at(self, t: int) -> float:
        if self.step_schedule == "inverse_time":
            return self.step0 / t
        if self.step_schedule == "inverse_sqrt":
            return self.step0 / math.sqrt(t)
        raise ValueError("step_schedule must be 'inverse_time' or 'inverse_sqrt'.")


@dataclass
class WarmStartConfig:
    n_samples: int = 5_000
    ridge: float = 1e-8


@dataclass
class SetLearningConfig:
    n_samples: int = 5_000
    r: Optional[int] = None
    r_scale: float = 1.0
    max_removed_gaps: Optional[int] = None
    min_interval_width: float = 1e-8


@dataclass
class ExperimentSetup:
    problem: TruncatedRegressionProblem
    wrong_intervals: List[Interval]
    warm_start: WarmStartConfig = field(default_factory=WarmStartConfig)
    set_learning: SetLearningConfig = field(default_factory=SetLearningConfig)
    psgd: PSGDConfig = field(default_factory=PSGDConfig)

    def __post_init__(self) -> None:
        self.wrong_intervals = normalize_intervals(self.wrong_intervals)


@dataclass
class PSGDTrace:
    w_last: np.ndarray
    error_trajectory: Optional[np.ndarray]
    step_sizes: np.ndarray
    grad_norms: np.ndarray
    projection_hit_rate_trajectory: np.ndarray


@dataclass
class MethodRun:
    method: str
    final_w: np.ndarray
    error_trajectory: np.ndarray
    intervals_used: Optional[List[Interval]]
    step_sizes: Optional[np.ndarray] = None
    grad_norms: Optional[np.ndarray] = None
    projection_hit_rate_trajectory: Optional[np.ndarray] = None

    @property
    def final_error(self) -> float:
        return float(self.error_trajectory[-1])


@dataclass
class SingleRunResult:
    seed: int
    w_hat: np.ndarray
    learned_intervals: Optional[List[Interval]]
    method_runs: "OrderedDict[str, MethodRun]"


@dataclass
class TrajectoryStats:
    mean: np.ndarray
    std: np.ndarray

    @property
    def final_mean(self) -> float:
        return float(self.mean[-1])

    @property
    def final_std(self) -> float:
        return float(self.std[-1])


@dataclass
class ComparisonResult:
    setup: ExperimentSetup
    methods: Tuple[str, ...]
    runs: List[SingleRunResult]
    trajectory_stats: Dict[str, TrajectoryStats]


def validate_methods(methods: Iterable[str]) -> Tuple[str, ...]:
    unique = []
    for method in methods:
        if method not in ALL_METHODS:
            raise ValueError(f"Unknown method '{method}'. Available methods: {ALL_METHODS}")
        if method not in unique:
            unique.append(method)
    if not unique:
        raise ValueError("At least one method must be selected.")
    return tuple(unique)


def build_constant_method_run(
    method: str,
    w_hat: np.ndarray,
    reference_w: np.ndarray,
    T: int,
) -> MethodRun:
    """OLS has no iterations, so we plot it as a constant error line."""
    error = float(np.linalg.norm(w_hat - reference_w))
    trajectory = np.full(T + 1, error, dtype=float)
    return MethodRun(
        method=method,
        final_w=np.asarray(w_hat, dtype=float).copy(),
        error_trajectory=trajectory,
        intervals_used=None,
    )


def run_psgd_method(
    setup: ExperimentSetup,
    method: str,
    w_hat: np.ndarray,
    intervals: Sequence[Interval],
    psgd_seed: int,
) -> MethodRun:
    """Run one PSGD-based method from the common warm start."""
    sim = setup.problem.make_simulator(psgd_seed)
    trace = sim.psgd(
        w_init=w_hat,
        intervals=intervals,
        config=setup.psgd,
        reference_w=setup.problem.w_star,
        verbose_label=METHOD_LABELS[method],
    )
    if trace.error_trajectory is None:
        raise RuntimeError("Simulation comparison requires a reference parameter w*.")
    return MethodRun(
        method=method,
        final_w=trace.w_last,
        error_trajectory=trace.error_trajectory,
        intervals_used=list(normalize_intervals(intervals)),
        step_sizes=trace.step_sizes,
        grad_norms=trace.grad_norms,
        projection_hit_rate_trajectory=trace.projection_hit_rate_trajectory,
    )


def run_single_replication(
    setup: ExperimentSetup,
    seed: int,
    methods: Iterable[str] = ALL_METHODS,
) -> SingleRunResult:
    """Run one replicate. All PSGD methods share the same warm start."""
    methods = validate_methods(methods)
    method_runs: "OrderedDict[str, MethodRun]" = OrderedDict()
    warm_seed = int(seed)
    set_seed = int(seed) + 1
    psgd_seed = int(seed) + 2

    warm_sim = setup.problem.make_simulator(warm_seed)
    w_hat = warm_sim.find_warm_start(
        n_samples=setup.warm_start.n_samples,
        ridge=setup.warm_start.ridge,
    )

    learned_intervals: Optional[List[Interval]] = None
    if "full" in methods:
        set_sim = setup.problem.make_simulator(set_seed)
        learned_intervals = set_sim.learn_truncation_set(
            w_hat,
            k=len(setup.problem.truncation_intervals),
            n_samples=setup.set_learning.n_samples,
            r=setup.set_learning.r,
            r_scale=setup.set_learning.r_scale,
            max_removed_gaps=setup.set_learning.max_removed_gaps,
            min_interval_width=setup.set_learning.min_interval_width,
        )

    for method in methods:
        if method == "ols":
            method_runs[method] = build_constant_method_run(
                method=method,
                w_hat=w_hat,
                reference_w=setup.problem.w_star,
                T=setup.psgd.T,
            )
        elif method == "wrong_set":
            method_runs[method] = run_psgd_method(
                setup=setup,
                method=method,
                w_hat=w_hat,
                intervals=setup.wrong_intervals,
                psgd_seed=psgd_seed,
            )
        elif method == "true_set":
            method_runs[method] = run_psgd_method(
                setup=setup,
                method=method,
                w_hat=w_hat,
                intervals=setup.problem.truncation_intervals,
                psgd_seed=psgd_seed,
            )
        elif method == "full":
            if learned_intervals is None:
                raise RuntimeError("learned_intervals should have been computed for the full method.")
            method_runs[method] = run_psgd_method(
                setup=setup,
                method=method,
                w_hat=w_hat,
                intervals=learned_intervals,
                psgd_seed=psgd_seed,
            )
        else:
            raise ValueError(f"Unexpected method '{method}'.")
    return SingleRunResult(
        seed=int(seed),
        w_hat=w_hat,
        learned_intervals=learned_intervals,
        method_runs=method_runs,
    )


def run_repeated_experiment(
    setup: ExperimentSetup,
    R: int = 1,
    methods: Iterable[str] = ALL_METHODS,
    base_seed: int = 0,
) -> ComparisonResult:
    if R <= 0:
        raise ValueError("R must be positive.")
    methods = validate_methods(methods)
    runs: List[SingleRunResult] = []
    for r in range(R):
        seed = base_seed + 10_000 * r
        runs.append(run_single_replication(setup=setup, seed=seed, methods=methods))
    trajectory_stats: Dict[str, TrajectoryStats] = {}
    for method in methods:
        stacked = np.stack([run.method_runs[method].error_trajectory for run in runs], axis=0)
        trajectory_stats[method] = TrajectoryStats(
            mean=stacked.mean(axis=0),
            std=stacked.std(axis=0),
        )
    return ComparisonResult(
        setup=setup,
        methods=methods,
        runs=runs,
        trajectory_stats=trajectory_stats,
    )

def plot_comparison(
    result: ComparisonResult,
    output_path: Optional[str | Path] = None,
    show_std: bool = True,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    palatino_rc = {
        # Tries Palatino first; falls back automatically if unavailable.
        "font.family": "serif",
        "font.serif": [
            "Palatino",
            "Palatino Linotype",
            "Book Antiqua",
            "URW Palladio L",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
    }

    TITLE_SIZE = 22
    LABEL_SIZE = 20
    LEGEND_SIZE = 16
    TICK_SIZE = 16
    LINE_WIDTH = 4

    # Give every line a distinct style so the plot is readable even in grayscale.
    style_by_label = {
        "OLS (no correction)": dict(linestyle=":", linewidth=LINE_WIDTH),
        "PSGD with wrong $S$": dict(linestyle="-.", linewidth=LINE_WIDTH),
        "PSGD with true $S^\star$": dict(linestyle="-", linewidth=LINE_WIDTH), 
        "Full algorithm": dict(linestyle='--', linewidth=LINE_WIDTH),
    }
    fallback_styles = [
        dict(linestyle="-", linewidth=LINE_WIDTH),
        dict(linestyle="--", linewidth=LINE_WIDTH),
        dict(linestyle="-.", linewidth=LINE_WIDTH),
        dict(linestyle=":", linewidth=LINE_WIDTH),
    ]

    with mpl.rc_context(palatino_rc):
        fig, ax = plt.subplots(figsize=(8.5, 5.5))

        # Fully white background.
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_axisbelow(True)

        x = np.arange(result.setup.psgd.T + 1)

        for i, method in enumerate(result.methods):
            stats = result.trajectory_stats[method]
            label = METHOD_LABELS[method]
            style = style_by_label.get(label, fallback_styles[i % len(fallback_styles)])

            line, = ax.plot(
                x,
                stats.mean,
                label=label,
                **style,
                zorder=3 + i,
            )

            if show_std and len(result.runs) >= 2:
                ax.fill_between(
                    x,
                    stats.mean - stats.std,
                    stats.mean + stats.std,
                    color=line.get_color(),
                    alpha=0.12,
                    zorder=1,
                )

        ax.set_xlabel("Iteration", fontsize=LABEL_SIZE)
        ax.set_ylabel(r"$\|w_t - w^\star\|_2$", fontsize=LABEL_SIZE)
        ax.set_title(
            title or "Truncated regression comparison",
            fontsize=TITLE_SIZE,
            pad=14,
        )
        ax.tick_params(axis="both", labelsize=TICK_SIZE)

        ax.legend(
            fontsize=LEGEND_SIZE,
            loc="best",
            frameon=True,
            framealpha=1.0,
            facecolor="white",
            edgecolor="black",
            handlelength=3.0,
            borderpad=0.8,
            labelspacing=0.6,
        )

        ax.grid(True, color="0.85", linewidth=1.0, alpha=1.0)
        fig.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_path,
                dpi=800,
                bbox_inches="tight",
                facecolor="white",
            )

        return fig, ax


def print_experiment_summary(result: ComparisonResult) -> None:
    setup = result.setup
    print("Problem dimension:", setup.problem.d)
    print("True w*:", np.array2string(setup.problem.w_star, precision=4))
    print("True truncation set S*:", setup.problem.truncation_intervals)
    print("Wrong-set baseline uses S:", setup.wrong_intervals)
    print("Number of outer reruns R:", len(result.runs))
    print("PSGD iterations T:", setup.psgd.T)
    surv_sim = setup.problem.make_simulator(seed=987654321)
    print("Estimated survival probability:", surv_sim.estimate_survival_probability(n_samples=100_000))
    first_run = result.runs[0]
    print("Warm-start error ||w_hat - w*|| (run 1):", np.linalg.norm(first_run.w_hat - setup.problem.w_star))
    if first_run.learned_intervals is not None:
        print("Learned truncation intervals (run 1):", first_run.learned_intervals)
    print("\nFinal error summary (mean +/- std across reruns):")
    for method in result.methods:
        stats = result.trajectory_stats[method]
        print(f"  {METHOD_LABELS[method]:25s}: {stats.final_mean:.6f} +/- {stats.final_std:.6f}")


def make_default_demo_setup() -> ExperimentSetup:
    """Default problem with clearly separated baselines."""
    problem = TruncatedRegressionProblem(
        w_star= 20 * np.ones(10),
        truncation_intervals=[(-3.75, -3), (-2.5, -1.5), (-1, 1), (2, 3), (3.25, 4)],
        feature_means=np.asarray([
            np.zeros(10),
            -0.5 * np.ones(10), 
            0.5 * np.ones(10),
            0.5 * np.concatenate([np.ones(5), -np.ones(5)], axis=0),
            0.5 * np.concatenate([-np.ones(5), np.ones(5)], axis=0)
        ]),
        feature_weights=np.asarray([0.25, 0.3, 0.15, 0.1, 0.2]),
        feature_covariance=np.eye(10),
        noise_std=1.0,
    )
    return ExperimentSetup(
        problem=problem,
        wrong_intervals=[(-5.0, 5.0)],
        warm_start=WarmStartConfig(n_samples=5_000, ridge=1e-8),
        set_learning=SetLearningConfig(
            n_samples=5_000,
            r=None,
            r_scale=1.0,
            max_removed_gaps=30,
            min_interval_width=1e-8,
        ),
        psgd=PSGDConfig(
            radius=10,
            T=4_500,
            batch_size=128,
            step0=20,
            step_schedule="inverse_sqrt",
            grad_clip=10.0,
            use_conditional_mean=True,
            verbose_every=500,
        ),
    )


def parse_methods_arg(methods_arg: str) -> Tuple[str, ...]:
    methods = [part.strip() for part in methods_arg.split(",") if part.strip()]
    return validate_methods(methods)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run truncated linear regression baselines and plot ||w_t - w*|| over time."
    )
    parser.add_argument("--R", type=int, default=1)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--step0", type=float, default=None)
    parser.add_argument("--verbose-every", type=int, default=None)
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(ALL_METHODS), 
        help=f"Comma-separated subset of methods. Available: {', '.join(ALL_METHODS)}",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="truncated_regression_comparison.png",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    setup = make_default_demo_setup()
    if args.T is not None:
        setup.psgd.T = int(args.T)
    if args.batch_size is not None:
        setup.psgd.batch_size = int(args.batch_size)
    if args.step0 is not None:
        setup.psgd.step0 = float(args.step0)
    if args.verbose_every is not None:
        setup.psgd.verbose_every = int(args.verbose_every)
    methods = parse_methods_arg(args.methods)
    result = run_repeated_experiment(
        setup=setup,
        R=args.R,
        methods=methods,
        base_seed=0,
    )
    print_experiment_summary(result)
    if not args.no_plot:
        plot_title = f"Truncated regression comparison ($R={args.R}$)"
        plot_comparison(result, output_path=args.output_plot, show_std=True, title=plot_title)
        print(f"\nSaved plot to: {args.output_plot}")


if __name__ == "__main__":
    main()
