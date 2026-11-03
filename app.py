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
            basket[col] = basket[col].map(lambda x: False if pd.isna(x) or str(x).strip().lower() in {"0", "false", "no", ""} else True)
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
st.caption("Upload a Kaggle dataset, inspect and prepare it, then run association mining, decision trees and clustering algorithms.")

uploaded = st.file_uploader("Upload a Kaggle dataset (CSV or Excel)", type=["csv", "xlsx"])
if uploaded:
    try:
        raw_df = load_data(uploaded)
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        st.stop()
    if raw_df.empty:
        st.error("The uploaded dataset is empty.")
        st.stop()
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
        if st.button("Clean dataset", type="primary"):
            cleaned_df, report = clean_dataset(raw_df)
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
                if st.button("Run " + ("Apriori" if tab_index == 0 else "FP-Growth"), key=f"run_assoc_{tab_index}"):
                    try:
                        basket = prepare_transactions(working_df, mode, selected_items, transaction_col if mode != "One row per transaction" else None, item_col if mode != "One row per transaction" else None)
                        if basket.empty or basket.shape[1] == 0:
                            st.warning("No transaction items were found. Check the selected columns.")
                        else:
                            with st.spinner("Mining frequent itemsets..."):
                                frequent = apriori(basket, min_support=support, use_colnames=True) if tab_index == 0 else fpgrowth(basket, min_support=support, use_colnames=True)
                            st.write("Frequent itemsets")
                            if frequent.empty:
                                st.info("No frequent itemsets found. Try lowering minimum support.")
                            else:
                                frequent["itemsets"] = frequent["itemsets"].apply(lambda values: ", ".join(sorted(map(str, values))))
                                st.dataframe(frequent.sort_values("support", ascending=False), use_container_width=True)
                                try:
                                    rules = association_rules(apriori(basket, min_support=support, use_colnames=True) if tab_index == 0 else fpgrowth(basket, min_support=support, use_colnames=True), metric="confidence", min_threshold=confidence)
                                    if not rules.empty:
                                        rules["antecedents"] = rules["antecedents"].apply(lambda values: ", ".join(sorted(map(str, values))))
                                        rules["consequents"] = rules["consequents"].apply(lambda values: ", ".join(sorted(map(str, values))))
                                        st.write("Association rules")
                                        st.dataframe(rules.sort_values("confidence", ascending=False), use_container_width=True)
                                        st.download_button("Download association rules", csv_download(rules), f"{['apriori','fpgrowth'][tab_index]}_rules.csv", "text/csv", key=f"download_rules_{tab_index}")
                                    else:
                                        st.info("No rules met the confidence threshold. Lower minimum confidence.")
                                except Exception as exc:
                                    st.warning(f"Frequent itemsets were generated, but rules could not be generated: {exc}")
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
                if st.button("Train " + algorithm, key=f"train_custom_{tab_index}"):
                    if not features:
                        st.warning("Select at least one input feature.")
                    else:
                        model_data = working_df[features + [target]].dropna(subset=[target]).copy()
                        for col in features:
                            if pd.api.types.is_numeric_dtype(model_data[col]):
                                bins = min(5, max(2, model_data[col].nunique()))
                                model_data[col] = pd.cut(model_data[col], bins=bins, duplicates="drop").astype(str)
                            else:
                                model_data[col] = model_data[col].astype(str)
                        model_data[target] = model_data[target].astype(str)
                        if len(model_data) < 5 or model_data[target].nunique() < 2:
                            st.error("The target needs at least two classes and the dataset needs at least five usable rows.")
                        else:
                            train, test = train_test_split(model_data, test_size=test_size / 100, random_state=42, stratify=model_data[target] if model_data[target].value_counts().min() >= 2 else None)
                            tree = build_categorical_tree(train, target, features, algorithm, max_depth=max_depth)
                            predictions = [predict_categorical(tree, row) for _, row in test.iterrows()]
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
                if st.button("Train CART"):
                    if not features:
                        st.warning("Select at least one input feature.")
                    else:
                        model_data = working_df[features + [target]].dropna(subset=[target]).copy()
                        X = model_data[features]
                        y = model_data[target].astype(str)
                        if len(model_data) < 5 or y.nunique() < 2:
                            st.error("The target needs at least two classes and the dataset needs at least five usable rows.")
                        else:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42, stratify=y if y.value_counts().min() >= 2 else None)
                            preprocessor, numeric, categorical = make_preprocessor(X)
                            model = Pipeline([("preprocess", preprocessor), ("tree", DecisionTreeClassifier(criterion="gini", max_depth=max_depth, random_state=42))])
                            model.fit(X_train, y_train)
                            predictions = model.predict(X_test)
                            st.metric("Test accuracy", f"{accuracy_score(y_test, predictions):.2%}")
                            st.code(classification_report(y_test, predictions, zero_division=0))
                            result = pd.DataFrame({"Actual": y_test.values, "Predicted": predictions})
                            st.dataframe(result.head(100), use_container_width=True)
                            st.download_button("Download CART predictions", csv_download(result), "cart_predictions.csv", "text/csv")
                            transformed = model.named_steps["preprocess"].transform(X_train)
                            names = model.named_steps["preprocess"].get_feature_names_out()
                            fig, ax = plt.subplots(figsize=(18, 9))
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
                if st.button("Run clustering"):
                    if not features:
                        st.warning("Select at least one feature.")
                    else:
                        X = working_df[features].copy()
                        preprocessor, numeric, categorical = make_preprocessor(X)
                        matrix = preprocessor.fit_transform(X)
                        matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix
                        if len(X) < 3:
                            st.error("At least three rows are required for clustering.")
                        elif algorithm == "K-Means":
                            labels = KMeans(n_clusters=clusters, random_state=42, n_init=10).fit_predict(matrix)
                        elif algorithm == "Agglomerative (Hierarchical)":
                            labels = AgglomerativeClustering(n_clusters=clusters).fit_predict(matrix)
                        else:
                            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(matrix)
                        if len(X) >= 3:
                            output = working_df.copy()
                            output["Cluster"] = labels
                            st.metric("Clusters found", len(set(labels) - {-1}) if -1 in labels else len(set(labels)))
                            st.dataframe(output.head(100), use_container_width=True)
                            valid_labels = set(labels) - {-1}
                            if len(valid_labels) >= 2 and len(valid_labels) < len(labels):
                                try:
                                    st.metric("Silhouette score", f"{silhouette_score(matrix, labels):.4f}")
                                except Exception:
                                    pass
                            numeric_features = [col for col in features if pd.api.types.is_numeric_dtype(working_df[col])]
                            if len(numeric_features) >= 2:
                                chart_df = working_df[numeric_features[:2]].copy()
                                chart_df["Cluster"] = labels.astype(str)
                                fig = px.scatter(chart_df, x=numeric_features[0], y=numeric_features[1], color="Cluster", title="Cluster visualization")
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
