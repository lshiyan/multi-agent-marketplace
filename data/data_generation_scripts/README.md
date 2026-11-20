# Scripts for synthetic data generation 

This folder contains scripts for synthetic data generation. Generate a synthetic dataset by running the following command:

```bash
python generate_customers_and_businesses.py <path_to_output_directory>
```

See, 
```bash
python generate_customers_and_businesses.py --help
```
for more details on the arguments.


### Prices

After the businesses are generated, `pricing.py` computes competitive (Nash) and
monopoly benchmark prices under a logit demand model and saves them to
`<output_dir>/benchmark_prices.yaml`. Every business then starts at its competitive
prices for the requested items (distractor items keep their sampled prices). Costs
are fixed beforehand, from the sampled prices times `min_price_factor`, and stored as
`minimum_menu_prices`, so resetting the listed prices doesn't move them. Each customer's maximum price for an item is
then sampled as the item's competitive price (averaged over the businesses selling
it) times a multiplier drawn from Normal(mean, std). Set these with
`--max-price-multiplier-mean` (default 1.2) and `--max-price-multiplier-std`
(default 0.1).

`run_experiment` reads the benchmarks from that file. For a dataset generated before
this change, create the file with:

```bash
python pricing.py <path_to_dataset_directory>
```


### Validation
You can validate the generated data by running the following command:

```bash
python validate.py <path_to_input_directory>
```


### Extras

Feature files have already been generated and stored in the `features` folder. If you would like to regenerate them, run the following command:

```bash
python generate_items.py <path_to_output_file: e.g. features/items.yaml>
```

```bash
python generate_people.py <path_to_output_file: e.g. features/people.yaml>
```

### Contractors Dataset

For generating items
```bash
python generate_item_contractors.py <path_to_output_file>
```

For generating business and customer YAML files
```bash
python generate_customers_and_business_contractors.py -f <path_to_features_directory> -c <number_of_customers> <output_dir>
```

