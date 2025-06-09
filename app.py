import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import shap

st.set_page_config(page_title="MLB Flexible Outcome Predictor", layout="wide")
st.title("MLB Flexible Outcome Probability/Regression Predictor & Explainer")

uploaded_file = st.file_uploader("Upload MLB data file (CSV or Excel)", type=["csv", "xlsx"])

@st.cache_data(show_spinner=False)
def load_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    try:
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    return None

def preprocess(df, target_col):
    # Remove rows with missing target
    df = df[df[target_col].notnull()].copy()
    y = df[target_col].values
    Xdf = df.drop(target_col, axis=1)

    # Separate numeric/categorical features
    numeric_cols = Xdf.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in Xdf.columns if c not in numeric_cols]

    if numeric_cols:
        Xdf[numeric_cols] = Xdf[numeric_cols].fillna(Xdf[numeric_cols].median())
        scaler = StandardScaler()
        X_num = scaler.fit_transform(Xdf[numeric_cols])
    else:
        X_num = np.empty((len(df), 0))

    if categorical_cols:
        Xdf[categorical_cols] = Xdf[categorical_cols].fillna("Unknown")
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        X_cat = encoder.fit_transform(Xdf[categorical_cols])
        cat_features = encoder.get_feature_names_out(categorical_cols)
    else:
        X_cat = np.empty((len(df), 0))
        cat_features = []

    X = np.hstack([X_num, X_cat])
    feature_names = numeric_cols + list(cat_features)
    return X, y, feature_names, df

def get_id_col(df):
    for c in df.columns:
        if any(k in c.lower() for k in ['player', 'batter', 'name']):
            return c
    return None

if uploaded_file:
    df = load_file(uploaded_file)
    if df is None or df.shape[1] <= 1:
        st.error("File could not be read. Make sure it's a valid CSV or Excel file with columns.")
    else:
        st.success(f"Loaded file with {df.shape[0]} rows and {df.shape[1]} columns.")
        st.write("First few rows of your file:")
        st.dataframe(df.head(10))

        # Let user pick the target column to predict
        possible_targets = [
            c for c in df.columns 
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1
        ]
        if not possible_targets:
            st.error("No numeric columns found to predict. Please upload a file with outcome columns.")
        else:
            target_col = st.selectbox("Pick outcome column to predict", possible_targets)
            X, y, feature_names, clean_df = preprocess(df, target_col)
            # Detect binary vs regression
            is_binary = np.array_equal(np.unique(y), [0, 1]) or np.array_equal(np.unique(y), [1, 0])
            if X.shape[0] < 10:
                st.error("Not enough rows after cleaning to train a model. Please upload more data.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
                if is_binary:
                    model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.07, use_label_encoder=False, eval_metric='logloss')
                else:
                    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.07)
                model.fit(X_train, y_train)
                # Predict for all
                preds = model.predict(X)
                # SHAP explanations
                explainer = shap.TreeExplainer(model)
                sample_idx = np.arange(min(X.shape[0], 1000))
                shap_values = explainer.shap_values(X[sample_idx])
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                top_features = sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1])[:10]
                result_df = clean_df.copy()
                pred_col = f"{target_col}_prediction"
                result_df[pred_col] = preds
                id_col = get_id_col(result_df)
                show_n = st.slider("Show top N predictions", min_value=1, max_value=min(20, result_df.shape[0]), value=5)
                if is_binary:
                    sort_idx = np.argsort(result_df[pred_col])[-show_n:][::-1]
                else:
                    sort_idx = np.argsort(result_df[pred_col])[-show_n:][::-1]
                st.subheader(f"Top {show_n} Predictions for {target_col}")
                show_cols = [id_col] if id_col else []
                show_cols += [pred_col]
                st.dataframe(result_df.iloc[sort_idx][show_cols])
                st.subheader("Pros & Cons per Prediction (Top N)")
                for rank, idx in enumerate(sort_idx):
                    row = result_df.iloc[idx]
                    row_shap = explainer.shap_values(X[idx:idx+1])[0] if idx in sample_idx else mean_abs_shap
                    top_pos = np.argsort(row_shap)[-2:][::-1]
                    top_neg = np.argsort(row_shap)[:2]
                    st.markdown(
                        f"**{rank+1}. {row.get(id_col, 'Row '+str(idx))}** — Prediction: `{row[pred_col]:.3f}`"
                    )
                    st.write("Pros (features boosting outcome):")
                    for j in top_pos:
                        st.write(f"  + **{feature_names[j]}** (value: `{row.get(feature_names[j], 'NA')}`, impact: `{row_shap[j]:.2f}`)")
                    st.write("Cons (features lowering outcome):")
                    for j in top_neg:
                        st.write(f"  - **{feature_names[j]}** (value: `{row.get(feature_names[j], 'NA')}`, impact: `{row_shap[j]:.2f}`)")
                    st.write("---")
                st.subheader("Top 10 Global Feature Importances")
                st.table(pd.DataFrame(top_features, columns=['Feature', 'Importance']))
                st.download_button(
                    f"Download all {pred_col} predictions",
                    result_df.to_csv(index=False),
                    file_name=f"{pred_col}_predictions.csv"
                )
else:
    st.info("Upload a CSV or Excel file to get started!")
