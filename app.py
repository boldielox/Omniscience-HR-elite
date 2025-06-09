import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from functools import reduce

st.set_page_config(page_title="Omniscience MLB Advanced Model", layout="wide")
st.title("Omniscience MLB Advanced Model (Robust Integration)")

# Helper to make all column names lower, strip whitespace, and replace - with _
def normalize_cols(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in df.columns]
    return df

# Try reading the file as CSV, then Excel if that fails
def try_read(file):
    try:
        df = pd.read_csv(file)
    except Exception:
        file.seek(0)
        try:
            df = pd.read_excel(file)
        except Exception:
            st.error(f"Could not read {file.name}")
            return None
    return normalize_cols(df)

# Robust join key detection
def find_join_keys(df1, df2):
    # Acceptable synonyms for a "player" key
    possible_keys = [
        ['id', 'player_id', 'playerid'],
        ['name', 'player', 'player_name', 'last_name,_first_name', 'pitcher_name'],
    ]
    for keys in possible_keys:
        for k1 in keys:
            for k2 in keys:
                if k1 in df1.columns and k2 in df2.columns:
                    return k1, k2
    # Try fuzzy: any column exactly matching in both
    for col in df1.columns:
        if col in df2.columns:
            return col, col
    return None, None

def robust_merge(dfs):
    # Start with first DataFrame
    merged = dfs[0]
    merge_log = []
    for i, df in enumerate(dfs[1:], 1):
        k1, k2 = find_join_keys(merged, df)
        if k1 and k2:
            prev_rows = len(merged)
            merged = pd.merge(merged, df, how='left', left_on=k1, right_on=k2, suffixes=('', f'_df{i}'))
            merge_log.append(f"Merged file {i+1} on '{k1}' and '{k2}' ({prev_rows} rows -> {len(merged)} rows)")
        else:
            merge_log.append(f"Could NOT merge file {i+1}: no common key found. Columns: {df.columns}")
    return merged, merge_log

uploaded_files = st.file_uploader(
    "Upload your MLB metrics files (CSV or Excel, multiple allowed)", 
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    dfs = []
    for file in uploaded_files:
        df = try_read(file)
        if df is not None and not df.empty:
            dfs.append(df)
            st.success(f"Loaded {file.name} ({df.shape[0]} rows, {df.shape[1]} cols)")
        else:
            st.error(f"File {file.name} is empty or unreadable.")
    if len(dfs) < 2:
        st.warning("Please upload at least 2 files with a common key for robust merging.")
    else:
        merged_df, merge_log = robust_merge(dfs)
        st.write("**Merge Log:**")
        for line in merge_log:
            st.write("-", line)
        st.write("**Merged Data Preview:**")
        st.dataframe(merged_df.head(20))
        
        # Use only numeric columns for modeling
        numeric_columns = list(merged_df.select_dtypes(include=[np.number]).columns)
        if not numeric_columns:
            st.error("No numeric columns detected in merged data. Try files with numeric stats.")
        else:
            features = st.multiselect("Select features for prediction:", numeric_columns)
            target = st.selectbox("Select target column (label):", ["None"] + numeric_columns)
            
            if features and target != "None":
                X = merged_df[features]
                y = merged_df[target]
                mask = X.notnull().all(axis=1) & y.notnull()
                X = X[mask]
                y = y[mask]
                if len(X) == 0:
                    st.error("No rows with complete data after cleaning. Please check your selected columns.")
                else:
                    model = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
                    model.fit(X, y)
                    preds = model.predict_proba(X)[:, 1]
                    merged_df_clean = merged_df.loc[X.index].copy()
                    merged_df_clean['Prediction'] = preds
                    st.write("### Prediction Results (Top 20)")
                    st.dataframe(merged_df_clean[['Prediction'] + features].head(20))
                    st.write("### Feature Importances")
                    st.table(sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]))
                    csv = merged_df_clean.to_csv(index=False)
                    st.download_button('Download Results as CSV', csv, file_name='omniscience_predictions.csv')
            else:
                st.info("Select at least one feature and a target to train the model.")
else:
    st.info("Upload your MLB metrics files to begin.")
