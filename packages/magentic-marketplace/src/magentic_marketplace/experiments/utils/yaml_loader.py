"""YAML loading functions for experiment data."""

from pathlib import Path

import yaml

from magentic_marketplace.marketplace.shared.models import Business, Customer


def load_businesses_from_yaml(businesses_dir: Path) -> list[Business]:
    """Load business profiles from YAML files in the given directory."""
    businesses: list[Business] = []

    if not businesses_dir.exists():
        raise FileNotFoundError(f"Businesses directory not found: {businesses_dir}")

    yaml_files = list(businesses_dir.glob("*.yaml")) + list(
        businesses_dir.glob("*.yml")
    )

    if not yaml_files:
        raise ValueError(
            f"No YAML files found in businesses directory: {businesses_dir}"
        )

    for yaml_file in sorted(yaml_files):
        with open(yaml_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        business = Business.model_validate(data)
        businesses.append(business)

    return businesses


def load_customers_from_yaml(customers_dir: Path) -> list[Customer]:
    """Load customer profiles from YAML files in the given directory."""
    customers: list[Customer] = []

    if not customers_dir.exists():
        raise FileNotFoundError(f"Customers directory not found: {customers_dir}")

    yaml_files = list(customers_dir.glob("*.yaml")) + list(customers_dir.glob("*.yml"))

    if not yaml_files:
        raise ValueError(f"No YAML files found in customers directory: {customers_dir}")

    for yaml_file in sorted(yaml_files):
        with open(yaml_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        customer = Customer.model_validate(data)
        customers.append(customer)

    return customers


BENCHMARK_FILENAME = "benchmark_prices.yaml"


def load_benchmark_prices(
    data_dir: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Load competitive and monopoly benchmark prices saved with a dataset.

    The file is written by data/data_generation_scripts/pricing.py, either during
    generation or by running ``python pricing.py <data_dir>`` on an existing dataset.
    """
    path = Path(data_dir) / BENCHMARK_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark prices not found at {path}. Generate them with: "
            f"python data/data_generation_scripts/pricing.py {data_dir}"
        )

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data["competitive_prices"], data["monopoly_prices"]
