import io
import json
import math
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Mining Workbench", page_icon="📊", layout="wide")

@st.cache_data(show_spinner=False)
def load_data_cached(file_bytes, filename):
    if filename.lower().endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")
    return pd.read_excel(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def clean_dataset_cached(data):
    return clean_dataset(data)

@st.cache_data(show_spinner=False)
def mine_association_cached(basket, algorithm, support, confidence):
    frequent = apriori(basket, min_support=support, use_colnames=True, max_len=4) if algorithm == "Apriori" else fpgrowth(basket, min_support=support, use_colnames=True, max_len=4)
    rules = association_rules(frequent, metric="confidence", min_threshold=confidence) if not frequent.empty and len(frequent) > 1 else pd.DataFrame()
    return frequent, rules

@st.cache_data(show_spinner=False)
def cluster_cached(frame, features, algorithm, clusters, eps, min_samples, max_rows):
    features = list(features)
    data = frame[features].copy()
    sampled = len(data) > max_rows
    if sampled:
        data = data.sample(n=max_rows, random_state=42)
    preprocessor, numeric, categorical = make_preprocessor(data)
    matrix = preprocessor.fit_transform(data)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    if algorithm == "K-Means":
        labels = KMeans(n_clusters=clusters, random_state=42, n_init=10, algorithm="lloyd").fit_predict(matrix)
    elif algorithm == "Agglomerative (Hierarchical)":
        labels = AgglomerativeClustering(n_clusters=clusters, linkage="ward").fit_predict(matrix)
    else:
        labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(matrix)
    score = None
    unique_labels = set(labels)
    if 2 <= len(unique_labels) < len(labels) and len(labels) > 2:
        score = silhouette_score(matrix, labels, sample_size=min(1500, len(labels)), random_state=42)
    return data.index, labels, score, sampled, len(frame)

@st.cache_data(show_spinner=False)
def train_cart_cached(frame, features, target, test_size, max_depth):
    features = list(features)
    model_data = frame[features + [target]].dropna(subset=[target]).copy()
    X = model_data[features]
    y = model_data[target].astype(str)
    if len(model_data) < 5 or y.nunique() < 2:
        return None
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42, stratify=stratify)
    preprocessor, numeric, categorical = make_preprocessor(X)
    model = Pipeline([("preprocess", preprocessor), ("tree", DecisionTreeClassifier(criterion="gini", max_depth=max_depth, random_state=42))])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, X_train, X_test, y_test, predictions

@st.cache_data(show_spinner=False)
def train_custom_tree_cached(frame, features, target, test_size, max_depth, algorithm):
    features = list(features)
    model_data = frame[features + [target]].dropna(subset=[target]).copy()
    for col in features:
        if pd.api.types.is_numeric_dtype(model_data[col]):
            bins = min(5, max(2, model_data[col].nunique()))
            model_data[col] = pd.cut(model_data[col], bins=bins, duplicates="drop").astype(str)
        else:
            model_data[col] = model_data[col].astype(str)
    model_data[target] = model_data[target].astype(str)
    if len(model_data) < 5 or model_data[target].nunique() < 2:
        return None
    stratify = model_data[target] if model_data[target].value_counts().min() >= 2 else None
    train, test = train_test_split(model_data, test_size=test_size / 100, random_state=42, stratify=stratify)
    tree = build_categorical_tree(train, target, features, algorithm, max_depth=max_depth)
    predictions = [predict_categorical(tree, row) for row in test[features].to_dict("records")]
    return tree, test, predictions



def load_data(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")
    return pd.read_excel(uploaded_file)


def clean_dataset(data):
    result = data.copy()
    report = []
    before = len(result)
    result = result.drop_duplicates()
    report.append({"operation": "Removed duplicate rows", "rows_removed": before - len(result)})
    for col in result.columns:
        missing = int(result[col].isna().sum())
        if missing:
            if pd.api.types.is_numeric_dtype(result[col]):
                value = result[col].median()
                result[col] = result[col].fillna(value)
                method = "median"
            else:
                mode = result[col].mode(dropna=True)
                value = mode.iloc[0] if not mode.empty else "Unknown"
                result[col] = result[col].fillna(value)
                method = "mode"
            report.append({"operation": "Filled missing values", "column": str(col), "count": missing, "method": method})
    return result, report


def prepare_transactions(data, mode, item_columns, transaction_column, item_column):
    if mode == "One row per transaction":
        basket = data[item_columns].copy()
        for col in basket.columns:
            values = basket[col]
            if pd.api.types.is_bool_dtype(values):
                basket[col] = values.fillna(False)
            elif pd.api.types.is_numeric_dtype(values):
                basket[col] = values.fillna(0).ne(0)
            else:
                normalized = values.fillna("").astype(str).str.strip().str.lower()
                basket[col] = ~normalized.isin(["", "0", "false", "no", "nan", "none"])
        return basket.astype(bool)
    grouped = data.groupby(transaction_column)[item_column].apply(lambda values: set(values.dropna().astype(str)))
    items = sorted(set().union(*grouped.tolist())) if len(grouped) else []
    rows = [{item: item in transaction_items for item in items} for transaction_items in grouped]
    return pd.DataFrame(rows, index=grouped.index).astype(bool)


def entropy(labels):
    counts = Counter(labels)
    total = sum(counts.values())
    return -sum((count / total) * math.log2(count / total) for count in counts.values() if count)


def choose_split(data, target, features, criterion):
    parent_entropy = entropy(data[target].tolist())
    best_feature = None
    best_score = -1
    best_parts = None
    for feature in features:
        parts = {value: subset for value, subset in data.groupby(feature, dropna=False)}
        total = len(data)
        remainder = sum((len(subset) / total) * entropy(subset[target].tolist()) for subset in parts.values())
        gain = parent_entropy - remainder
        split_info = -sum((len(subset) / total) * math.log2(len(subset) / total) for subset in parts.values() if len(subset))
        score = gain if criterion == "ID3 (Information Gain)" else (gain / split_info if split_info else 0)
        if score > best_score:
            best_feature, best_score, best_parts = feature, score, parts
    return best_feature, best_score, best_parts


def build_categorical_tree(data, target, features, criterion, depth=0, max_depth=6, min_samples=2):
    labels = data[target].astype(str).tolist()
    majority = Counter(labels).most_common(1)[0][0]
    if len(set(labels)) == 1 or not features or depth >= max_depth or len(data) < min_samples:
        return {"leaf": majority, "samples": len(data)}
    feature, score, parts = choose_split(data, target, features, criterion)
    if feature is None or score <= 0:
        return {"leaf": majority, "samples": len(data)}
    branches = {}
    remaining = [item for item in features if item != feature]
    for value, subset in parts.items():
        branches[str(value)] = build_categorical_tree(subset, target, remaining, criterion, depth + 1, max_depth, min_samples)
    return {"feature": str(feature), "score": float(score), "majority": majority, "samples": len(data), "branches": branches}


def predict_categorical(tree, row):
    node = tree
    while "leaf" not in node:
        value = str(row.get(node["feature"], ""))
        if value not in node["branches"]:
            return node["majority"]
        node = node["branches"][value]
    return node["leaf"]


def tree_as_text(tree, prefix=""):
    if "leaf" in tree:
        return prefix + "Predict: " + str(tree["leaf"]) + " (samples=" + str(tree["samples"]) + ")\n"
    output = prefix + "Split: " + tree["feature"] + " | score=" + f"{tree['score']:.4f}" + "\n"
    for value, child in tree["branches"].items():
        output += prefix + "  " + tree["feature"] + " = " + value + "\n"
        output += tree_as_text(child, prefix + "    ")
    return output


def make_preprocessor(frame):
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    categorical = [col for col in frame.columns if col not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return ColumnTransformer(transformers), numeric, categorical


def csv_download(frame):
    return frame.to_csv(index=False).encode("utf-8")

st.title("📊 Data Mining & Machine Learning Workbench")
st.caption("Upload a Kaggle dataset, inspect and prepare it, then run association mining, decision trees and clustering algorithms. Results are cached to make repeated runs faster.")

uploaded = st.file_uploader("Upload a Kaggle dataset (CSV or Excel)", type=["csv", "xlsx"])
if uploaded:
    try:
        raw_df = load_data_cached(uploaded.getvalue(), uploaded.name)
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        st.stop()
    if raw_df.empty:
        st.error("The uploaded dataset is empty.")
        st.stop()
    upload_signature = (uploaded.name, len(uploaded.getvalue()))
    if st.session_state.get("upload_signature") != upload_signature:
        for state_key in list(st.session_state.keys()):
            if state_key.endswith("_done") or state_key.startswith(("assoc_done_", "custom_done_")):
                del st.session_state[state_key]
        st.session_state.pop("cleaned_df", None)
        st.session_state.pop("clean_report", None)
        st.session_state["upload_signature"] = upload_signature
    st.session_state["raw_df"] = raw_df
    st.success(f"Loaded {len(raw_df):,} rows and {len(raw_df.columns):,} columns.")
    overview = st.columns(4)
    overview[0].metric("Rows", f"{len(raw_df):,}")
    overview[1].metric("Columns", len(raw_df.columns))
    overview[2].metric("Missing values", f"{int(raw_df.isna().sum().sum()):,}")
    overview[3].metric("Duplicate rows", f"{int(raw_df.duplicated().sum()):,}")
    with st.expander("Dataset preview", expanded=True):
        st.dataframe(raw_df.head(100), use_container_width=True)
        st.write("Column information")
        st.dataframe(pd.DataFrame({"Data type": raw_df.dtypes.astype(str), "Missing": raw_df.isna().sum(), "Unique": raw_df.nunique(dropna=True)}), use_container_width=True)
    cleaned_df = None
    with st.expander("Data cleaning and preparation"):
        if st.button("Clean dataset", type="primary") or "cleaned_df" in st.session_state:
            cleaned_df, report = clean_dataset_cached(raw_df)
            st.session_state["cleaned_df"] = cleaned_df
            st.session_state["clean_report"] = report
        if "cleaned_df" in st.session_state:
            cleaned_df = st.session_state["cleaned_df"]
            st.success("Dataset cleaned and ready for mining.")
            st.dataframe(cleaned_df.head(100), use_container_width=True)
            st.json(st.session_state.get("clean_report", []))
            st.download_button("Download cleaned dataset", csv_download(cleaned_df), "cleaned_dataset.csv", "text/csv")
    working_df = st.session_state.get("cleaned_df", raw_df)
    tabs = st.tabs(["Apriori", "FP-Growth", "ID3", "CART", "C4.5", "Clustering", "Export & Reports"])

    for tab_index, tab in enumerate(tabs):
        with tab:
            if tab_index in (0, 1):
                st.subheader("Association Rule Mining — " + ("Apriori" if tab_index == 0 else "FP-Growth"))
                mode = st.radio("Dataset format", ["One row per transaction", "Transaction ID and item columns"], key=f"assoc_mode_{tab_index}", horizontal=True)
                if mode == "One row per transaction":
                    default_items = working_df.columns.tolist()
                    selected_items = st.multiselect("Select item columns", working_df.columns.tolist(), default=default_items, key=f"assoc_items_{tab_index}")
                else:
                    transaction_col = st.selectbox("Transaction ID column", working_df.columns, key=f"txn_{tab_index}")
                    item_options = [col for col in working_df.columns if col != transaction_col]
                    item_col = st.selectbox("Item column", item_options, key=f"item_{tab_index}")
                    selected_items = []
                support = st.slider("Minimum support", 0.01, 1.0, 0.05, 0.01, key=f"support_{tab_index}")
                confidence = st.slider("Minimum confidence", 0.01, 1.0, 0.5, 0.01, key=f"confidence_{tab_index}")
                if st.button("Run " + ("Apriori" if tab_index == 0 else "FP-Growth"), key=f"run_assoc_{tab_index}") or f"assoc_done_{tab_index}" in st.session_state:
                    try:
                        basket = prepare_transactions(working_df, mode, selected_items, transaction_col if mode != "One row per transaction" else None, item_col if mode != "One row per transaction" else None)
                        if basket.empty or basket.shape[1] == 0:
                            st.warning("No transaction items were found. Check the selected columns.")
                        elif basket.shape[1] > 500:
                            st.warning("More than 500 item columns were selected. Select a smaller set of relevant item columns to avoid combinatorial mining time.")
                        elif len(basket) > 100000:
                            st.warning("This transaction dataset is very large. Filter it to a representative period or category before mining.")
                        else:
                            with st.spinner("Mining frequent itemsets and generating rules..."):
                                frequent, rules = mine_association_cached(basket, "Apriori" if tab_index == 0 else "FP-Growth", support, confidence)
                            st.session_state[f"assoc_done_{tab_index}"] = True
                            st.write("Frequent itemsets")
                            if frequent.empty:
                                st.info("No frequent itemsets found. Try lowering minimum support.")
                            else:
                                frequent_display = frequent.copy()
                                frequent_display["itemsets"] = frequent_display["itemsets"].apply(lambda values: ", ".join(sorted(map(str, values))))
                                st.dataframe(frequent_display.sort_values("support", ascending=False), use_container_width=True)
                                if not rules.empty:
                                    rules_display = rules.copy()
                                    rules_display["antecedents"] = rules_display["antecedents"].apply(lambda values: ", ".join(sorted(map(str, values))))
                                    rules_display["consequents"] = rules_display["consequents"].apply(lambda values: ", ".join(sorted(map(str, values))))
                                    st.write("Association rules")
                                    st.dataframe(rules_display.sort_values("confidence", ascending=False), use_container_width=True)
                                    st.download_button("Download association rules", csv_download(rules_display), f"{['apriori','fpgrowth'][tab_index]}_rules.csv", "text/csv", key=f"download_rules_{tab_index}")
                                else:
                                    st.info("No rules met the confidence threshold. Lower minimum confidence.")
                    except Exception as exc:
                        st.error(f"Association mining failed: {exc}")
            elif tab_index in (2, 4):
                algorithm = "ID3 (Information Gain)" if tab_index == 2 else "C4.5 (Gain Ratio)"
                st.subheader("Decision Tree — " + algorithm)
                target = st.selectbox("Select target/class column", working_df.columns, key=f"target_{tab_index}")
                available = [col for col in working_df.columns if col != target]
                features = st.multiselect("Select input features", available, default=available, key=f"features_{tab_index}")
                test_size = st.slider("Test data percentage", 10, 40, 20, key=f"test_{tab_index}")
                max_depth = st.slider("Maximum tree depth", 1, 12, 6, key=f"depth_{tab_index}")
                if st.button("Train " + algorithm, key=f"train_custom_{tab_index}") or f"custom_done_{tab_index}" in st.session_state:
                    if not features:
                        st.warning("Select at least one input feature.")
                    else:
                        with st.spinner("Training decision tree..."):
                            trained = train_custom_tree_cached(working_df, tuple(features), target, test_size, max_depth, algorithm)
                            st.session_state[f"custom_done_{tab_index}"] = True
                        if trained is None:
                            st.error("The target needs at least two classes and the dataset needs at least five usable rows.")
                        else:
                            tree, test, predictions = trained
                            st.metric("Test accuracy", f"{accuracy_score(test[target], predictions):.2%}")
                            st.write("Decision tree rules")
                            st.code(tree_as_text(tree))
                            st.write("Classification report")
                            st.code(classification_report(test[target], predictions, zero_division=0))
                            result = test[[target]].copy()
                            result["Predicted"] = predictions
                            st.dataframe(result.head(100), use_container_width=True)
                            st.download_button("Download predictions", csv_download(result), f"{['id3','c45'][tab_index == 4]}_predictions.csv", "text/csv", key=f"pred_custom_{tab_index}")
            elif tab_index == 3:
                st.subheader("Decision Tree — CART (Gini Index)")
                target = st.selectbox("Select target/class column", working_df.columns, key="cart_target")
                features = st.multiselect("Select input features", [col for col in working_df.columns if col != target], default=[col for col in working_df.columns if col != target], key="cart_features")
                test_size = st.slider("Test data percentage", 10, 40, 20, key="cart_test")
                max_depth = st.slider("Maximum tree depth", 1, 12, 5, key="cart_depth")
                if st.button("Train CART") or "cart_done" in st.session_state:
                    if not features:
                        st.warning("Select at least one input feature.")
                    else:
                        with st.spinner("Training CART model..."):
                            trained = train_cart_cached(working_df, tuple(features), target, test_size, max_depth)
                            st.session_state["cart_done"] = True
                        if trained is None:
                            st.error("The target needs at least two classes and the dataset needs at least five usable rows.")
                        else:
                            model, X_train, X_test, y_test, predictions = trained
                            st.metric("Test accuracy", f"{accuracy_score(y_test, predictions):.2%}")
                            st.code(classification_report(y_test, predictions, zero_division=0))
                            result = pd.DataFrame({"Actual": y_test.values, "Predicted": predictions})
                            st.dataframe(result.head(100), use_container_width=True)
                            st.download_button("Download CART predictions", csv_download(result), "cart_predictions.csv", "text/csv")
                            names = model.named_steps["preprocess"].get_feature_names_out()
                            fig, ax = plt.subplots(figsize=(14, 7))
                            plot_tree(model.named_steps["tree"], feature_names=names, class_names=model.named_steps["tree"].classes_, filled=True, rounded=True, max_depth=3, ax=ax)
                            st.pyplot(fig)
            elif tab_index == 5:
                st.subheader("Clustering Algorithms")
                features = st.multiselect("Select features for clustering", working_df.columns.tolist(), default=working_df.select_dtypes(include=np.number).columns.tolist(), key="cluster_features")
                algorithm = st.selectbox("Clustering method", ["K-Means", "Agglomerative (Hierarchical)", "DBSCAN"])
                if algorithm in ("K-Means", "Agglomerative (Hierarchical)"):
                    clusters = st.slider("Number of clusters", 2, 10, 3)
                if algorithm == "DBSCAN":
                    eps = st.slider("Epsilon", 0.1, 5.0, 0.5, 0.1)
                    min_samples = st.slider("Minimum samples", 2, 20, 5)
                max_rows = st.slider("Maximum rows to process", 500, 10000, 3000, 500, help="Large datasets are sampled for clustering to avoid long waits and memory issues.")
                if algorithm == "Agglomerative (Hierarchical)":
                    max_rows = min(max_rows, 3000)
                    st.caption("Hierarchical clustering is memory-intensive. It is limited to at most 3,000 sampled rows.")
                if st.button("Run clustering") or "cluster_done" in st.session_state:
                    if not features:
                        st.warning("Select at least one feature.")
                    elif len(working_df) < 3:
                        st.error("At least three rows are required for clustering.")
                    else:
                        with st.spinner("Preparing features and clustering..."):
                            indices, labels, score, sampled, original_rows = cluster_cached(working_df, tuple(features), algorithm, clusters if algorithm != "DBSCAN" else 2, eps if algorithm == "DBSCAN" else 0.5, min_samples if algorithm == "DBSCAN" else 5, max_rows)
                            st.session_state["cluster_done"] = True
                        if sampled:
                            st.warning(f"To keep processing fast, clustering used a reproducible sample of {len(labels):,} rows from {original_rows:,}.")
                        output = working_df.loc[indices].copy()
                        output["Cluster"] = labels
                        st.metric("Clusters found", len(set(labels) - {-1}) if -1 in labels else len(set(labels)))
                        st.dataframe(output.head(100), use_container_width=True)
                        if score is not None:
                            st.metric("Silhouette score (sampled calculation)", f"{score:.4f}")
                        numeric_features = [col for col in features if pd.api.types.is_numeric_dtype(working_df[col])]
                        if len(numeric_features) >= 2:
                            chart_df = output[numeric_features[:2]].copy()
                            fig = px.scatter(chart_df, x=numeric_features[0], y=numeric_features[1], color=output["Cluster"].astype(str), title="Cluster visualization", render_mode="webgl")
                            st.plotly_chart(fig, use_container_width=True)
                        st.download_button("Download clustered dataset", csv_download(output), "clustered_dataset.csv", "text/csv")
    with tabs[6]:
        st.subheader("Assignment results and downloads")
        st.write("Run each algorithm from its tab. Download the generated itemsets, association rules, predictions or clustered dataset from the corresponding section.")
        st.write("Algorithms included: Apriori, FP-Growth, ID3, CART, C4.5, K-Means, Agglomerative Clustering and DBSCAN.")
        st.download_button("Download current working dataset", csv_download(working_df), "working_dataset.csv", "text/csv")
else:
    st.info("Start by uploading a Kaggle CSV or Excel dataset. You can use a market-basket dataset for Apriori/FP-Growth and a labelled dataset for decision-tree classification.")
    st.markdown("**Suggested Kaggle dataset types**\n- Market basket / grocery transactions for Apriori and FP-Growth\n- Classification datasets with a target column for ID3, CART and C4.5\n- Numeric or mixed-feature datasets for clustering")
