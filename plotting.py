#!/usr/bin/env python3
"""Generate marketplace experiment plots from CSV output files."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.lines import Line2D

def distinct_color_palette() -> list:
    """Return a high-contrast palette for experiment/model comparison plots."""
    names = ["tab10", "Dark2", "Set1", "Accent"]
    colors = []
    seen = set()
    for name in names:
        cmap = plt.get_cmap(name)
        for color in getattr(cmap, "colors", []):
            rounded = tuple(round(float(channel), 6) for channel in color)
            if rounded not in seen:
                seen.add(rounded)
                colors.append(color)
    return colors

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def plot_welfare(folder: Path, window: int = 5, tick_interval: int = 5) -> None:
    """Save separate customer and business welfare plots for one experiment."""
    rows = _read_csv(folder / "welfare_by_period.csv")
    if not rows:
        print("No welfare data found.")
        return

    rows = sorted(rows, key=lambda row: int(row["period"]))
    periods = np.array([int(row["period"]) for row in rows])

    metrics = [
        ("average_customer_welfare", "Customer", "customer_welfare_by_period.png"),
        ("average_business_welfare", "Business", "business_welfare_by_period.png"),
    ]
    for column, title, filename in metrics:
        values = np.array([float(row[column]) for row in rows])
        averaged = sliding_window_average(values, window)
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.plot(periods, averaged, label=f"{title} welfare (window={window})")
        axis.axhline(0, linewidth=0.8, linestyle="--", color="gray")
        axis.set_xlabel("Period")
        axis.set_ylabel(f"Average {title} Welfare")
        axis.xaxis.set_major_locator(MultipleLocator(tick_interval))
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(folder / filename, dpi=200, bbox_inches="tight")
        plt.close(figure)


def plot_purchases(folder: Path, window: int = 5) -> None:
    rows = _read_csv(folder / "purchases_by_period.csv")

    if not rows:
        print("No purchase data found.")
        return

    rows = sorted(rows, key=lambda row: int(row["period"]))

    periods = np.array([
        int(row["period"]) for row in rows
    ])

    successful = np.array([
        int(row["successful_purchases"]) for row in rows
    ])

    unsuccessful = np.array([
        int(row["unsuccessful_purchases"]) for row in rows
    ])

    total = successful + unsuccessful

    success_rate = np.divide(
        successful,
        total,
        out=np.full(len(total), np.nan),
        where=total > 0,
    )

    # Exclude periods with no customers
    valid = np.isfinite(success_rate)
    periods = periods[valid]
    success_rate = success_rate[valid]

    if len(periods) == 0:
        print("No valid purchase data found.")
        return

    smoothed_rate = sliding_window_average(
        success_rate, window
    )

    figure, axis = plt.subplots(figsize=(9, 5))

    axis.plot(
        periods,
        success_rate,
        color="tab:blue",
        alpha=0.25,
        linewidth=1,
        label="Raw success rate",
    )

    axis.plot(
        periods,
        smoothed_rate,
        color="tab:blue",
        linewidth=2.5,
        label=f"Sliding average (window={window})",
    )

    axis.set_xlabel("Period")
    axis.set_ylabel("Purchase Success Rate")

    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}")
    )
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        folder / "purchases_by_period.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_average_prices(folder: Path, tick_interval: int = 5) -> None:
    price_path = folder / "average_price_by_period.csv"
    rows = _read_csv(price_path)

    if not rows:
        return

    periods = np.array([int(row["period"]) for row in rows])
    businesses = [
        name for name in rows[0].keys()
        if name != "period"
    ]

    baselines_path = folder / "average_price_baselines.csv"
    baselines = {}
    if baselines_path.exists():
        baselines = {
            row["business"]: float(row["base_average_price"])
            for row in _read_csv(baselines_path)
        }

    figure, axis = plt.subplots(figsize=(10, 6))
    final_price_increases = []

    for business in businesses:
        prices = np.array([float(row[business]) for row in rows])
        line = axis.plot(
            periods,
            prices,
            label=business,
        )[0]

        if business in baselines:
            baseline = baselines[business]
            axis.axhline(
                y=baseline,
                linestyle="--",
                alpha=0.5,
                color=line.get_color(),
            )
            if baseline != 0:
                final_price_increases.append(
                    (prices[-1] - baseline) / baseline
                )

    axis.set_xlabel("Period")
    axis.set_ylabel("Average Price")
    axis.xaxis.set_major_locator(MultipleLocator(tick_interval))

    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    if final_price_increases:
        axis.text(
            1.02,
            0.25,
            "Average price increase: "
            f"{np.mean(final_price_increases):.2%}",
            transform=axis.transAxes,
            ha="left",
            va="center",
        )

    figure.tight_layout()
    figure.savefig(
        folder / "average_price_by_period.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

def plot_average_ppe(folder: Path, window: int = 5) -> None:
    rows = _read_csv(folder / "ppe_per_customer.csv")

    ppe_by_period = {}

    for row in rows:
        if row["status"] != "ok" or not row["ppe"]:
            continue

        period = int(row["period"])
        ppe = float(row["ppe"])

        if not np.isfinite(ppe):
            continue

        ppe_by_period.setdefault(period, []).append(ppe)

    if not ppe_by_period:
        print("No valid PPE values found.")
        return

    periods = np.array(sorted(ppe_by_period))

    average_ppe = np.array([
        np.mean(ppe_by_period[period])
        for period in periods
    ])

    window_ppe = sliding_window_average(average_ppe, window)

    figure, axis = plt.subplots(figsize=(9, 5))

    axis.plot(
        periods,
        window_ppe,
        label=f"Average PPE (window={window})",
    )

    axis.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label="Competitive benchmark",
    )

    if len(periods) >= 2:
        slope, intercept = np.polyfit(
            periods, window_ppe, 1
        )
        axis.plot(
            periods,
            slope * periods + intercept,
            linestyle="--",
            label="Best fit",
        )

        correlation = np.corrcoef(periods, window_ppe)[0, 1]

        if np.isfinite(correlation):
            axis.text(
                1.02,
                0.25,
                f"Pearson r = {correlation:.3f}",
                transform=axis.transAxes,
                ha="left",
                va="center",
            )

    axis.set_xlabel("Period")
    axis.set_ylabel("Average PPE")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}")
    )

    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        folder / "average_ppe_by_period.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

def normal_fit(values: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """Return a Gaussian MLE density fit, or None for degenerate samples."""
    if len(values) < 2:
        return None
    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=0))
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    density = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return density, mu, sigma


def plot_business_ppe_distribution(folder: Path) -> None:
    """Plot the distribution of final average item-level PPE across businesses."""
    rows = _read_csv(folder / "average_business_ppe.csv")

    values = []
    for row in rows:
        raw = row["average_ppe_increase"]
        if not raw:
            continue
        value = float(raw)
        if np.isfinite(value):
            values.append(value * 100)  # PPE ratio -> percentage points

    if not values:
        print("No valid business PPE values found.")
        return

    ppe = np.asarray(values, dtype=float)
    mean_ppe = float(np.mean(ppe))
    median_ppe = float(np.median(ppe))
    share_positive = float(np.mean(ppe > 0) * 100)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    spread = float(np.std(ppe))
    padding = max(3 * spread, 1.0)
    x = np.linspace(min(ppe.min(), mean_ppe - padding),
                    max(ppe.max(), mean_ppe + padding), 500)
    fit = normal_fit(ppe, x)
    if fit is not None:
        density, mu, sigma = fit
        axis.plot(x, density, color="tab:blue", linewidth=2.5,
                  label=f"Normal fit (μ={mu:.1f}%, σ={sigma:.1f} pp)")
    else:
        print(f"Skipping normal fit in {folder.name}: need at least 2 non-identical PPE values")
    axis.axvline(
        0, color="black", linestyle="--", linewidth=1.5,
        label="Competitive benchmark (0%)",
    )
    axis.axvline(
        median_ppe, color="tab:orange", linestyle="-", linewidth=1.8,
        label=f"Median ({median_ppe:.1f}%)",
    )

    summary = (
        f"Businesses: {len(ppe)}\n"
        f"Mean PPE: {mean_ppe:.1f}%\n"
        f"Median PPE: {median_ppe:.1f}%\n"
        f"Above competitive: {share_positive:.1f}%"
    )
    axis.text(
        0.98, 0.97, summary,
        transform=axis.transAxes, ha="right", va="top",
        bbox={"facecolor": "white", "edgecolor": "lightgray", "alpha": 0.9},
    )
    axis.set_xlabel("Average business PPE increase (%)")
    axis.set_ylabel("Probability density (per percentage point)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left")

    figure.tight_layout()
    figure.savefig(
        folder / "average_business_ppe_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

def sliding_window_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window < 1:
        raise ValueError("window must be >= 1.")

    return np.array([
        np.mean(values[max(0, i - window + 1):i + 1])
        for i in range(len(values))
    ])



def model_label(folder: Path) -> str:
    """Use an experiment's model.txt for plot legends; fall back to folder name."""
    path = folder / "model.txt"
    try:
        name = path.read_text(encoding="utf-8").strip()
        return name or folder.name
    except (OSError, UnicodeError):
        return folder.name



def save_horizontal_legend(output: Path, experiments: list[Path],
                           experiment_colors: dict, experiment_labels: dict) -> None:
    """Export a single-row model legend, independently of the comparison figures."""
    handles = [Line2D([0], [0], color=experiment_colors[folder], lw=2.8,
                      label=experiment_labels[folder]) for folder in experiments]
    # Grow the canvas with the number and length of model names to limit clipping.
    total_chars = sum(len(experiment_labels[folder]) for folder in experiments)
    width = max(5.0, len(experiments) * 0.85 + total_chars * 0.105)
    fig = plt.figure(figsize=(width, 0.9))
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False,
               fontsize=11, columnspacing=1.6, handlelength=2.5)
    destination = output / "legend_horizontal.png"
    fig.savefig(destination, dpi=200, bbox_inches="tight", pad_inches=0.14,
                transparent=True)
    plt.close(fig)
    print(f"Saved {destination}")


def plot_assets(assets: Path, window: int = 5, tick_interval: int = 5) -> None:
    """Compare immediate experiment subdirectories, skipping unavailable CSVs per plot."""
    experiments = sorted(path for path in assets.iterdir() if path.is_dir()
                         and path.name != "comparison_plots")
    if not experiments:
        print(f"No experiment subfolders found in {assets}")
        return

    output = assets / "comparison_plots"
    output.mkdir(parents=True, exist_ok=True)
    colors = distinct_color_palette()
    experiment_colors = {folder: colors[i % len(colors)]
                         for i, folder in enumerate(experiments)}
    experiment_labels = {folder: model_label(folder) for folder in experiments}
    save_horizontal_legend(output, experiments, experiment_colors, experiment_labels)

    def available(filename: str):
        for folder in experiments:
            path = folder / filename
            if not path.is_file():
                print(f"Skipping {folder.name} for {filename}: file missing")
                continue
            try:
                rows = _read_csv(path)
                if not rows:
                    print(f"Skipping {folder.name} for {filename}: empty CSV")
                    continue
                yield folder, rows
            except (OSError, csv.Error, UnicodeError) as exc:
                print(f"Skipping {folder.name} for {filename}: {exc}")

    def series(filename, transform):
        """Skip experiments with malformed data without losing the other series."""
        for folder, rows in available(filename):
            try:
                data = transform(rows)
            except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
                print(f"Skipping {folder.name} for {filename}: invalid data ({exc})")
                continue
            if data is None:
                print(f"Skipping {folder.name} for {filename}: no valid data")
                continue
            yield folder, data

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / name, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output / name}")

    def per_period(rows, *columns):
        rows = sorted(rows, key=lambda row: int(row["period"]))
        periods = np.array([int(row["period"]) for row in rows])
        values = [np.array([float(row[col]) for row in rows]) for col in columns]
        return (periods, *values)

    # Welfare: separate comparison plots, same experiment colors in each.
    welfare_series = list(series(
        "welfare_by_period.csv",
        lambda rows: per_period(rows, "average_customer_welfare", "average_business_welfare"),
    ))
    for index, title, filename in [
        (1, "Customer", "customer_welfare_by_period.png"),
        (2, "Business", "business_welfare_by_period.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for folder, data in welfare_series:
            periods, values = data[0], data[index]
            ax.plot(periods, sliding_window_average(values, window),
                    color=experiment_colors[folder], label=experiment_labels[folder])
        if welfare_series:
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set(xlabel="Period", ylabel=f"Average {title} Welfare")
            ax.xaxis.set_major_locator(MultipleLocator(tick_interval))
            ax.grid(alpha=0.3)
            save(fig, filename)
        else:
            plt.close(fig)

    # Purchases: success rate (not raw counts) for fair comparison.
    def purchase_data(rows):
        periods, successes, failures = per_period(
            rows, "successful_purchases", "unsuccessful_purchases")
        total = successes + failures
        valid = total > 0
        if not valid.any():
            return None
        return periods[valid], sliding_window_average(successes[valid] / total[valid], window)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    plotted = 0
    for folder, (periods, rate) in series("purchases_by_period.csv", purchase_data):
        ax.plot(periods, rate, color=experiment_colors[folder], label=experiment_labels[folder])
        plotted += 1
    if plotted:
        ax.set(xlabel="Period", ylabel="Purchase Success Rate", ylim=(0, 1))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.xaxis.set_major_locator(MultipleLocator(tick_interval))
        ax.grid(alpha=0.3)
        save(fig, "purchases_by_period.png")
    else:
        plt.close(fig)

    # Prices: each experiment contributes one line (mean across its business columns).
    def mean_prices(rows):
        columns = [key for key in rows[0] if key != "period"]
        if not columns:
            return None
        periods = np.array([int(row["period"]) for row in rows])
        prices = np.array([[float(row[col]) if row[col] else np.nan
                            for col in columns] for row in rows])
        valid = np.isfinite(prices).any(axis=1)
        if not valid.any():
            return None
        # Avoid warnings for rows where every business price is missing.
        means = np.array([np.mean(row[np.isfinite(row)]) if np.isfinite(row).any()
                          else np.nan for row in prices])
        order = np.argsort(periods[valid])
        return periods[valid][order], means[valid][order]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    plotted = 0
    for folder, (periods, values) in series("average_price_by_period.csv", mean_prices):
        ax.plot(periods, values, color=experiment_colors[folder], label=experiment_labels[folder])
        plotted += 1
    if plotted:
        ax.set(xlabel="Period",
               ylabel="Mean of Business Average Prices")
        ax.xaxis.set_major_locator(MultipleLocator(tick_interval))
        ax.grid(alpha=0.3)
        save(fig, "average_price_by_period.png")
    else:
        plt.close(fig)

    # PPE: average across status=ok observations in each period.
    def ppe_data(rows):
        grouped = {}
        for row in rows:
            if row["status"] != "ok" or not row["ppe"]:
                continue
            ppe = float(row["ppe"])
            if np.isfinite(ppe):
                grouped.setdefault(int(row["period"]), []).append(ppe)
        if not grouped:
            return None
        periods = np.array(sorted(grouped))
        means = np.array([np.mean(grouped[period]) for period in periods])
        return periods, sliding_window_average(means, window)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    plotted = 0
    for folder, (periods, values) in series("ppe_per_customer.csv", ppe_data):
        ax.plot(periods, values, color=experiment_colors[folder], label=experiment_labels[folder])
        plotted += 1
    if plotted:
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8,
                   label="Competitive benchmark")
        ax.set(xlabel="Period", ylabel="Average PPE")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.xaxis.set_major_locator(MultipleLocator(tick_interval))
        ax.grid(alpha=0.3)
        save(fig, "average_ppe_by_period.png")
    else:
        plt.close(fig)

    # Business PPE: Gaussian fits only, one per experiment.
    def business_ppe_data(rows):
        values = np.array([float(row["average_ppe_increase"]) * 100
                           for row in rows if row["average_ppe_increase"]], dtype=float)
        values = values[np.isfinite(values)]
        return values if len(values) else None

    distributions = list(series("average_business_ppe.csv", business_ppe_data))
    if distributions:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        all_values = np.concatenate([values for _, values in distributions])
        for folder, values in distributions:
            color = experiment_colors[folder]
            mu = float(np.mean(values))
            sigma = float(np.std(values))
            padding = max(3 * sigma, 1.0)
            x = np.linspace(min(all_values.min(), mu - padding),
                            max(all_values.max(), mu + padding), 500)
            fit = normal_fit(values, x)
            if fit is None:
                print(f"Skipping normal fit for {folder.name}: need at least 2 non-identical PPE values")
                continue
            density, mu, sigma = fit
            ax.plot(x, density, linewidth=2.3, color=color,
                    label=f"{experiment_labels[folder]} (normal: μ={mu:.1f}%, σ={sigma:.1f} pp)")
        ax.axvline(0, color="gray", linestyle="--", linewidth=1,
                   label="Competitive benchmark")
        ax.set(xlabel="Average business PPE increase (%)",
               ylabel="Probability density (per percentage point)")
        ax.grid(axis="y", alpha=0.3)
        save(fig, "average_business_ppe_distribution.png")

    print(f"Comparison plots saved to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate single-experiment or cross-experiment marketplace plots.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-f", "--folder", type=Path, help="Single experiment output folder.")
    mode.add_argument("-a", "--assets", type=Path,
                      help="Assets parent directory containing experiment subfolders.")
    parser.add_argument("-w", "--window", type=int, default=5,
                        help="Sliding average window (default: 5).")
    parser.add_argument("--tick-interval", type=int, default=5,
                        help="Period tick interval in comparison charts (default: 5).")
    args = parser.parse_args()
    if args.window < 1 or args.tick_interval < 1:
        parser.error("--window and --tick-interval must be positive integers")
    folder = (args.assets if args.assets is not None else args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    if args.assets is not None:
        plot_assets(folder, window=args.window, tick_interval=args.tick_interval)
    else:
        # Each plot is independent; missing input skips only that plot.
        jobs = [
            ("welfare_by_period.csv", plot_welfare, {"window": args.window, "tick_interval": args.tick_interval}),
            ("purchases_by_period.csv", plot_purchases, {"window": args.window}),
            ("average_price_by_period.csv", plot_average_prices, {"tick_interval": args.tick_interval}),
            ("ppe_per_customer.csv", plot_average_ppe, {"window": args.window}),
            ("average_business_ppe.csv", plot_business_ppe_distribution, {}),
        ]
        for filename, fn, kwargs in jobs:
            if not (folder / filename).is_file():
                print(f"Skipping {fn.__name__}: {filename} missing")
                continue
            fn(folder, **kwargs)
        print(f"Plots saved to: {folder}")


if __name__ == "__main__":
    main()
