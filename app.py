import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import difflib

st.set_page_config(page_title="Omniscience MLB Universal Analyzer", layout="wide")
st.title("Omniscience MLB Universal Analyzer")

def normalize_cols(df):
    df.columns = [
        str(c).strip().lower().replace("-", "_").replace(" ", "_").replace('.', '_')
        for c in df.columns
    ]
    return df

def try_read(file):
    for enc in ['utf-8-sig', 'utf-8', 'latin1']:
        file.seek(0)
        try:
            df = pd.read_csv(file, encoding=enc)
            df = normalize_cols(df)
            df = df.loc[:, ~df.columns.str.contains('^unnamed')]
            if len(df) > 0: return df
        except Exception:
            file.seek(0)
            try:
                df = pd.read_excel(file, engine='openpyxl')
                df = normalize_cols(df)
                df = df.loc[:, ~df.columns.str.contains('^unnamed')]
                if len(df) > 0: return df
            except Exception:
                continue
    return None

def find_best_key(df1, df2):
    keys1 = set(df1.columns)
    keys2 = set(df2.columns)
    preferred_keys = [
        ["id", "player_id", "playerid"],
        ["name", "player", "player_name", "last_name,_first_name", "pitcher_name"]
    ]
    for group in preferred_keys:
        for k1 in group:
            for k2 in group:
                if k1 in keys1 and k2 in keys2:
                    return k1, k2
    for c1 in keys1:
        close = difflib.get_close_matches(c1, list(keys2), n=1, cutoff=0.9)
        if close:
            return c1, close[0]
    for c in keys1:
        if c in keys2:
            return c, c
    return None, None

def safe_merge(df1, df2):
    k1, k2 = find_best_key(df1, df2)
    if k1 and k2:
        try:
            merged = pd.merge(df1, df2, how="inner", left_on=k1, right_on=k2, suffixes=('', '_file2'))
            return merged, f"Merged on '{k1}' and '{k2}'"
        except Exception as e:
            return None, f"Merge failed on '{k1}/{k2}': {e}"
    else:
        return None, "No common key found for merge."

uploaded_files = st.file_uploader(
    "Upload any number of MLB CSV or Excel files (even unrelated!)",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

file_dfs = []
file_names = []
for file in uploaded_files:
    df = try_read(file)
    if df is not None and not df.empty:
        file_dfs.append(df)
        file_names.append(file.name)
        st.success(f"Loaded {file.name} ({df.shape[0]} rows, {df.shape[1]} cols)")
    else:
        st.warning(f"File {file.name} could not be loaded or is empty.")

if len(file_dfs) == 0:
    st.info("Upload any MLB data files to begin (no need for them to be related).")
else:
    st.header("File Selection & Analysis")
    chosen_file_idx = st.selectbox("Select a file to analyze", list(range(len(file_names))), format_func=lambda i: file_names[i])
    chosen_df = file_dfs[chosen_file_idx]
    st.write(f"**Preview of {file_names[chosen_file_idx]}:**")
    st.dataframe(chosen_df.head(20))
    st.write("**Summary Statistics:**")
    st.write(chosen_df.describe(include='all'))

    if chosen_df.select_dtypes(include=[np.number]).shape[1] > 0:
        st.write("**Numeric Histograms:**")
        for col in chosen_df.select_dtypes(include=[np.number]).columns:
            st.bar_chart(chosen_df[col].dropna())
        st.write("**Correlation Matrix:**")
        st.dataframe(chosen_df.corr(numeric_only=True))

    # Option to try merging files
    if len(file_dfs) > 1:
        st.subheader("Optional: Merge Two Files for Modeling")
        merge1 = st.selectbox("Select first file for merge", list(range(len(file_names))), format_func=lambda i: file_names[i], key="merge1")
        merge2 = st.selectbox("Select second file for merge", list(range(len(file_names))), format_func=lambda i: file_names[i], key="merge2")
        if merge1 != merge2:
            merged_df, merge_log = safe_merge(file_dfs[merge1], file_dfs[merge2])
            st.write(f"**Merge Attempt:** {merge_log}")
            if merged_df is not None and not merged_df.empty:
                st.dataframe(merged_df.head(20))
                chosen_df = merged_df # Allow using this for modeling below
        
    # Modeling on any file
    st.header("Model Training & Prediction")
    numeric_cols = list(chosen_df.select_dtypes(include=[np.number]).columns)
    if not numeric_cols:
        st.error("No numeric columns detected for modeling in selected data.")
    else:
        features = st.multiselect("Select features for model:", numeric_cols, default=numeric_cols[:min(len(numeric_cols), 5)])
        target = st.selectbox("Select target column (label):", ["None"] + numeric_cols)
        if features and target != "None":
            try:
                X = chosen_df[features]
                y = chosen_df[target]
                mask = X.notnull().all(axis=1) & y.notnull()
                X = X[mask]
                y = y[mask]
                if len(X) == 0:
                    st.error("No rows with complete data after cleaning. Please check your selected columns.")
                else:
                    model = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
                    model.fit(X, y)
                    preds = model.predict_proba(X)[:, 1]
                    result_df = chosen_df.loc[X.index].copy()
                    result_df['Prediction'] = preds
                    st.write("### Prediction Results (Top 20 rows)")
                    st.dataframe(result_df[['Prediction'] + features].head(20))
                    st.write("### Feature Importances")
                    st.table(sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]))
                    csv = result_df.to_csv(index=False)
                    st.download_button('Download these results as CSV', csv, file_name='omniscience_predictions.csv')
            except Exception as e:
                st.error(f"Modeling failed: {e}")
        else:
            st.info("Select at least one feature and a target to train the model.")
