#!/usr/bin/env python3
"""Generate marketplace experiment plots from CSV output files."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def plot_welfare(folder: Path, window: int = 5, tick_interval: int = 5) -> None:
    path = folder / "welfare_by_period.csv"
    rows = _read_csv(path)

    if not rows:
        print("No welfare data found.")
        return

    rows = sorted(rows, key=lambda row: int(row["period"]))

    periods = np.array([int(row["period"]) for row in rows])
    customer = np.array([
        float(row["average_customer_welfare"]) for row in rows
    ])
    business = np.array([
        float(row["average_business_welfare"]) for row in rows
    ])

    customer_avg = sliding_window_average(customer, window)
    business_avg = sliding_window_average(business, window)

    figure, axis = plt.subplots(figsize=(9, 5))

    axis.plot(
        periods,
        customer_avg,
        label=f"Customer welfare (window={window})",
    )
    axis.plot(
        periods,
        business_avg,
        label=f"Business welfare (window={window})",
    )

    correlation_text = []

    if len(periods) >= 2:
        customer_r = np.corrcoef(periods, customer_avg)[0, 1]
        business_r = np.corrcoef(periods, business_avg)[0, 1]

        if np.isfinite(customer_r):
            correlation_text.append(f"Customer r = {customer_r:.3f}")

        if np.isfinite(business_r):
            correlation_text.append(f"Business r = {business_r:.3f}")

    axis.axhline(
        0,
        linewidth=0.8,
        linestyle="--",
        color="gray",
    )

    axis.set_title("Sliding Window Average Welfare by Period")
    axis.set_xlabel("Period")
    axis.set_ylabel("Average Welfare")
    axis.xaxis.set_major_locator(MultipleLocator(tick_interval))

    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    if correlation_text:
        axis.text(
            1.02,
            0.25,
            "\n".join(correlation_text),
            transform=axis.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
        )

    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        folder / "welfare_by_period.png",
        dpi=200,
        bbox_inches="tight",
    )
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

    axis.set_title("Purchase Success Rate by Period")
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

    axis.set_title("Average Business Prices by Period")
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

    axis.set_title("Sliding Window Average PPE by Period")
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
    axis.hist(ppe, bins="auto", edgecolor="white", alpha=0.85)
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
    axis.set_title("Distribution of Average PPE Across Businesses")
    axis.set_xlabel("Average business PPE increase (%)")
    axis.set_ylabel("Number of businesses")
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate marketplace plots from the CSV files in an "
            "experiment output folder."
        )
    )
    parser.add_argument(
        "-f", "--folder",
        type=Path,
        help="Folder containing the marketplace output CSV files.",
    )

    parser.add_argument(
        "-w", "--window",
        type=int,
        default=5,
        help="Sliding window size for welfare and PPE plots (default: 5).",
    )

    args = parser.parse_args()

    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    required = [
        "welfare_by_period.csv",
        "purchases_by_period.csv",
        "average_price_by_period.csv",
        "ppe_per_customer.csv",
        "average_business_ppe.csv",
    ]
    
    missing = [name for name in required if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required CSV file(s): " + ", ".join(missing)
        )

    plot_welfare(folder, window=args.window)
    plot_purchases(folder, window=args.window)
    plot_average_prices(folder)
    plot_average_ppe(folder, window=args.window)
    plot_business_ppe_distribution(folder)

    print(f"Plots saved to: {folder}")


if __name__ == "__main__":
    main()
