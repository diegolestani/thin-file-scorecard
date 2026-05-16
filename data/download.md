# Getting the Data

This project uses the **Home Credit Default Risk** dataset, released publicly
by Home Credit Group via Kaggle for research and educational purposes.

## Download instructions

1. Create a free account at [kaggle.com](https://www.kaggle.com) if you don't have one
2. Go to: https://www.kaggle.com/competitions/home-credit-default-risk/data
3. Accept the competition rules (required to download)
4. Download **`application_train.csv`** only — this is the only file this project uses
5. Place it at: `data/raw/application_train.csv`

## Why only the main application table?

The competition provides several supplementary tables (bureau data, previous
applications, installment history). This project deliberately uses only the
application table — the information available **at the moment of loan decision**,
before any repayment history exists.

This constraint is intentional: it replicates the thin-file setting where
alternative and proxy variables must substitute for credit history. Using
repayment history from prior loans would defeat the purpose.

## License

The dataset is provided under Kaggle's standard competition terms:
- Personal and educational use: permitted
- Commercial use: not permitted
- Redistribution of the raw data: not permitted

Code in this repository is MIT licensed (see LICENSE).
