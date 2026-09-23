#!/usr/bin/env python3
"""Generate marketplace experiment plots from CSV output files."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def plot_welfare(folder: Path) -> None:
    path = folder / "welfare_by_period.csv"
    rows = _read_csv(path)

    periods = np.array([int(row["period"]) for row in rows])
    customer = np.array(
        [float(row["average_customer_welfare"]) for row in rows]
    )
    business = np.array(
        [float(row["average_business_welfare"]) for row in rows]
    )

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(periods, customer, marker="o", label="Average customer welfare")
    axis.plot(periods, business, marker="o", label="Average business welfare")

    correlation_text = []
    if len(periods) >= 2:
        customer_slope, customer_intercept = np.polyfit(periods, customer, 1)
        business_slope, business_intercept = np.polyfit(periods, business, 1)

        axis.plot(
            periods,
            customer_slope * periods + customer_intercept,
            linestyle="--",
            label="Customer best fit",
        )
        axis.plot(
            periods,
            business_slope * periods + business_intercept,
            linestyle="--",
            label="Business best fit",
        )

        customer_r = np.corrcoef(periods, customer)[0, 1]
        business_r = np.corrcoef(periods, business)[0, 1]
        correlation_text = [
            f"Customer r = {customer_r:.3f}",
            f"Business r = {business_r:.3f}",
        ]

    axis.axhline(0, linewidth=0.8, linestyle="--")
    axis.set_title("Average Customer and Business Welfare by Period")
    axis.set_xlabel("Period")
    axis.set_ylabel("Average welfare")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

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


def plot_purchases(folder: Path) -> None:
    rows = _read_csv(folder / "purchases_by_period.csv")
    periods = np.array([int(row["period"]) for row in rows])
    successful = np.array(
        [int(row["successful_purchases"]) for row in rows]
    )
    unsuccessful = np.array(
        [int(row["unsuccessful_purchases"]) for row in rows]
    )

    width = 0.35
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(
        periods - width / 2,
        successful,
        width,
        label="Successful Purchase",
    )
    axis.bar(
        periods + width / 2,
        unsuccessful,
        width,
        label="No Purchase",
    )
    axis.set_xlabel("Period")
    axis.set_ylabel("Number of Customers")
    axis.set_title("Purchases by Period")
    axis.set_xticks(periods)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    figure.savefig(
        folder / "purchases_by_period.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_average_prices(folder: Path) -> None:
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
            marker="o",
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
    axis.set_xticks(periods)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate marketplace plots from the CSV files in an "
            "experiment output folder."
        )
    )
    parser.add_argument(
        "--folder",
        type=Path,
        help="Folder containing the marketplace output CSV files.",
    )
    args = parser.parse_args()

    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    required = [
        "welfare_by_period.csv",
        "purchases_by_period.csv",
        "average_price_by_period.csv",
    ]
    
    missing = [name for name in required if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required CSV file(s): " + ", ".join(missing)
        )

    plot_welfare(folder)
    plot_purchases(folder)
    plot_average_prices(folder)

    print(f"Plots saved to: {folder}")


if __name__ == "__main__":
    main()
