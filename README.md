# DataNexus AI — Fast Data Mining Workbench

A Streamlit application for dataset profiling, cleaning, association-rule mining, decision trees, clustering and exports.

## Run

```bash
python -m venv .venv
```

Windows:
```bash
.venv\\Scripts\\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Launch:
```bash
streamlit run app.py
```

## Performance improvements

- Uploaded CSV and Excel files are cached.
- Cleaning, association mining, decision-tree training and clustering are cached.
- Apriori/FP-Growth itemsets are calculated once and reused for association rules.
- Transaction conversion uses vectorized pandas operations.
- Custom decision-tree prediction avoids slow row-by-row pandas iteration.
- Clustering uses reproducible sampling for large datasets, with a visible notice.
- Silhouette scoring uses a sample to avoid expensive all-pairs calculations.
- Hierarchical clustering is limited to 3,000 rows because its memory and time costs grow rapidly.
- Results remain visible across Streamlit reruns.

## Notes

Apriori and FP-Growth limit generated itemsets to a maximum length of four to keep interactive mining responsive. Increase or remove `max_len=4` in `mine_association_cached` if longer itemsets are required, understanding that runtime can grow substantially.

For best performance, use only relevant columns and avoid unnecessarily high-cardinality categorical features.
