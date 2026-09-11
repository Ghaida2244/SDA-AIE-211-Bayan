"""Lab 6 starter: bootstrap confidence intervals."""

import numpy as np


def _validate_values(values, name):
    """Convert a metric sequence into a validated numeric array."""

    array = np.asarray(
        list(values),
        dtype=float,
    )

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")

    if array.size == 0:
        raise ValueError(f"{name} must not be empty")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")

    return array


def _validate_options(n_boot, alpha):
    """Validate shared bootstrap configuration."""

    if not isinstance(n_boot, int) or n_boot <= 0:
        raise ValueError("n_boot must be a positive integer")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")


def bootstrap_ci(
    values,
    *,
    n_boot=2000,
    seed=42,
    alpha=0.05,
):
    """Return the mean and its percentile bootstrap interval."""

    values = _validate_values(values, "values")
    _validate_options(n_boot, alpha)

    random_generator = np.random.default_rng(seed)
    sample_size = values.size
    bootstrap_means = np.empty(
        n_boot,
        dtype=float,
    )

    for bootstrap_index in range(n_boot):
        sampled_indices = random_generator.integers(
            0,
            sample_size,
            size=sample_size,
        )
        bootstrap_means[bootstrap_index] = np.mean(
            values[sampled_indices]
        )

    point_estimate = float(np.mean(values))
    lower = float(
        np.quantile(
            bootstrap_means,
            alpha / 2,
        )
    )
    upper = float(
        np.quantile(
            bootstrap_means,
            1 - alpha / 2,
        )
    )

    # Ensure the reported interval contains its point estimate
    lower = min(lower, point_estimate)
    upper = max(upper, point_estimate)

    return point_estimate, lower, upper


def paired_bootstrap_diff(
    a,
    b,
    *,
    n_boot=2000,
    seed=42,
    alpha=0.05,
):
    """Return a paired mean difference and bootstrap interval."""

    a = _validate_values(a, "a")
    b = _validate_values(b, "b")
    _validate_options(n_boot, alpha)

    if a.size != b.size:
        raise ValueError(
            "Paired inputs must have the same length"
        )

    differences = a - b
    random_generator = np.random.default_rng(seed)
    sample_size = differences.size
    bootstrap_differences = np.empty(
        n_boot,
        dtype=float,
    )

    # Resampling the differences preserves the original pairing
    for bootstrap_index in range(n_boot):
        sampled_indices = random_generator.integers(
            0,
            sample_size,
            size=sample_size,
        )
        bootstrap_differences[bootstrap_index] = np.mean(
            differences[sampled_indices]
        )

    point_estimate = float(np.mean(differences))
    lower = float(
        np.quantile(
            bootstrap_differences,
            alpha / 2,
        )
    )
    upper = float(
        np.quantile(
            bootstrap_differences,
            1 - alpha / 2,
        )
    )

    lower = min(lower, point_estimate)
    upper = max(upper, point_estimate)

    return point_estimate, lower, upper