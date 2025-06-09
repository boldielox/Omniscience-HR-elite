import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import shap

st.set_page_config(page_title="MLB Home Run Probability (Dynamic, Explainable)", layout="wide")
st.title("MLB Home Run Probability – Dynamic, Explainable, Multi-Player Analysis")

def normalize_cols(df):
    df.columns = [
        str(c).strip().lower().replace("-", "_").replace(" ", "_").replace('.', '_')
        for c in df.columns
    ]
    return df

def try_read(uploaded_file):
    """Try reading as CSV, then as Excel. Return None if both fail."""
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        df = normalize_cols(df)
        df = df.loc[:, ~df.columns.str.contains('^unnamed')]
        if not df.empty:
            return df
    except Exception as e_csv:
        pass
    try:
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        df = normalize_cols(df)
        df = df.loc[:, ~df.columns.str.contains('^unnamed')]
        if not df.empty:
            return df
    except Exception as e_xls:
        pass
    return None

def find_hr_target(df):
    hr_targets = [c for c in df.columns if any(k in c.lower() for k in ['hr', 'home_run', 'homers'])]
    hr_targets = [c for c in hr_targets if pd.api.types.is_numeric_dtype(df[c])]
    if not hr_targets:
        hr_targets = [c for c in df.select_dtypes(include=np.number).columns if df[c].nunique() <= 2]
    return hr_targets[0] if hr_targets else None

def encode_categoricals(df, feature_cols):
    categoricals = [c for c in feature_cols if df[c].dtype == 'object']
    if not categoricals:
        return df[feature_cols], []
    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    cats = df[categoricals].fillna('Unknown')
    cat_encoded = enc.fit_transform(cats)
    cat_feature_names = enc.get_feature_names_out(categoricals)
    df_cat = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=df.index)
    numerics = [c for c in feature_cols if df[c].dtype != 'object']
    df_all = pd.concat([df[numerics], df_cat], axis=1)
    return df_all, cat_feature_names

# --- MAIN APP LOGIC ---
uploaded_file = st.file_uploader(
    "Upload a MLB CSV or Excel file (should include a home run indicator column and as many features as you want)", 
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    st.info(f"File {uploaded_file.name} uploaded. Attempting to process...")
    df = try_read(uploaded_file)
    if df is None:
        st.error("❌ File could not be loaded! Please ensure it is a valid CSV or Excel file with a header row and try again.")
    elif df.empty:
        st.error("❌ File was loaded but is empty. Check your file contents.")
    else:
        st.success(f"✅ Loaded {uploaded_file.name}: {df.shape[0]} rows, {df.shape[1]} cols")
        st.dataframe(df.head(20))

        target_col = find_hr_target(df)
        if not target_col:
            st.error("❌ Could not find any suitable home run column. Please include a binary HR indicator like 'hr', 'home_run', or 'homers'.")
        else:
            st.info(f"Using column '{target_col}' as the home run target variable.")
            feature_cols = [c for c in df.columns if c != target_col]
            X, cat_feats = encode_categoricals(df, feature_cols)
            mask = X.notnull().all(axis=1) & df[target_col].notnull()
            X = X[mask]
            y = df.loc[mask, target_col]
            if X.empty or y.empty:
                st.error("❌ No usable data after filtering out missing values. Check your file for missing data.")
            else:
                show_n = st.slider("Show top N predictions", min_value=1, max_value=min(30, X.shape[0]), value=5)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                model = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_test)[:, 1]
                confidences = 1 - 2 * np.abs(probs - 0.5)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                results = pd.DataFrame(X_test, columns=X.columns)
                results['probability'] = probs
                results['confidence'] = confidences
                # Try to include player or batter name column for display, else just index
                id_col = None
                for c in df.columns:
                    if any(k in c.lower() for k in ['batter', 'player', 'name', 'hitter']):
                        id_col = c
                        break
                if id_col and id_col in df.columns:
                    results[id_col] = df.loc[X_test.index, id_col].values
                # Show top N
                top_idx = np.argsort(results['probability'])[-show_n:][::-1]
                st.subheader(f"Top {show_n} Home Run Probabilities")
                st.dataframe(results.iloc[top_idx][[id_col, 'probability', 'confidence']] if id_col else results.iloc[top_idx][['probability', 'confidence']])
                # Detailed pros & cons
                st.subheader("Detailed Analysis (Pros & Cons for Top N)")
                for i, idx in enumerate(top_idx):
                    row = results.iloc[idx]
                    st.markdown(f"**{i+1}. {row.get(id_col, f'Row {idx}')}** — Probability: `{row['probability']:.3f}`, Confidence: `{row['confidence']:.2f}`")
                    sv = shap_values[idx]
                    # Top 2 pros (features boosting), top 2 cons (features lowering)
                    top
