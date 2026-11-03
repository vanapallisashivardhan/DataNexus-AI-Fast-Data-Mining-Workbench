# Data Mining & Machine Learning Workbench

A Streamlit application for the Data Mining Assignment-II. It supports dataset upload, inspection, cleaning, association rule mining, decision-tree classification and clustering.

## Algorithms

- Apriori association rule mining
- FP-Growth frequent-pattern mining
- ID3 decision tree using information gain
- CART decision tree using Gini impurity
- C4.5-style decision tree using gain ratio
- K-Means clustering
- Agglomerative (hierarchical) clustering
- DBSCAN clustering

## Requirements

Python 3.9 or newer is recommended.

## Run locally

```bash
python -m venv .venv
```

Activate the environment:

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install packages and launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Dataset guidance

- Apriori and FP-Growth: use a market-basket dataset. Choose either one row per transaction with item columns, or a transaction-ID column plus an item column.
- ID3, CART and C4.5: use a labelled dataset and select the class/target column. Numeric input features are discretized into bins for ID3 and C4.5.
- Clustering: select numeric or categorical input features. The app preprocesses missing values, encodes categories and scales numeric features.

## Notes

The ID3 and C4.5 implementations are educational categorical decision-tree implementations. CART uses scikit-learn's Gini-based decision tree. The C4.5 implementation uses gain ratio, while ID3 uses information gain. For a formal assignment report, state the dataset, selected columns, parameter values and observed results for each algorithm.
