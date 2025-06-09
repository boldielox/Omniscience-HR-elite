import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import shap
import io

st.set_page_config(page_title="MLB Home Run Predictor", layout="wide")
st.title("MLB Home Run Probability Predictor & Explainer")

st.write(
    """
    Upload a CSV or Excel file with player data (must include a binary home run column, like `hr`, `home_run`, or similar).
    The app will train a model to predict home run probability for each row and explain each prediction.
    """
)

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload MLB data file", type=["csv", "xlsx"])

@st.cache_data(show_spinner=False)
def load_file(uploaded_file):
    # Try CSV
    try:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    # Try Excel
    try:
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    return None

def find_hr_target(df):
    # Accepts hr, home_run, homers, or any binary column
    candidates = [c for c in df.columns if any(k in c.lower() for k in ['hr', 'home_run', 'homers'])]
    for c in candidates:
        if pd.api.types.is_numeric_dtype(df[c]) and set(df[c].dropna().unique()).issubset({0, 1}):
            return c
    # Else pick any binary column
    for c in df.select_dtypes(include=[np.number]).columns:
        unique = set(df[c].dropna().unique())
        if unique.issubset({0, 1}):
            return c
    return None

def preprocess(df, target_col):
    # Remove rows with missing target
    df = df[df[target_col].notnull()].copy()
    y = df[target_col].astype(int).values
    Xdf = df.drop(target_col, axis=1)

    # Separate numeric/categorical features
    numeric_cols = Xdf.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in Xdf.columns if c not in numeric_cols]

    # Impute and scale numerics
    if numeric_cols:
        Xdf[numeric_cols] = Xdf[numeric_cols].fillna(Xdf[numeric_cols].median())
        scaler = StandardScaler()
        X_num = scaler.fit_transform(Xdf[numeric_cols])
    else:
        X_num = np.empty((len(df), 0))

    # Impute and encode categoricals
    if categorical_cols:
        Xdf[categorical_cols] = Xdf[categorical_cols].fillna("Unknown")
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        X_cat = encoder.fit_transform(Xdf[categorical_cols])
        cat_features = encoder.get_feature_names_out(categorical_cols)
    else:
        X_cat = np.empty((len(df), 0))
        cat_features = []

    # Combine
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

        target_col = find_hr_target(df)
        if not target_col:
            st.error("Could not find a binary home run column (e.g. 'hr', 'home_run', 'homers').")
        else:
            st.info(f"Using '{target_col}' as the target (home run indicator) column.")
            X, y, feature_names, clean_df = preprocess(df, target_col)
            if X.shape[0] < 10:
                st.error("Not enough rows after cleaning to train a model. Please upload more data.")
            else:
                # Fit model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
                model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.07, use_label_encoder=False, eval_metric='logloss')
                model.fit(X_train, y_train)

                # Predict on all data (not just test)
                pred_probs = model.predict_proba(X)[:, 1]
                confidence = 1 - 2 * np.abs(pred_probs - 0.5)

                # SHAP explanations (on all data, but restrict to 1000 rows for performance)
                explainer = shap.TreeExplainer(model)
                sample_idx = np.arange(min(X.shape[0], 1000))
                shap_values = explainer.shap_values(X[sample_idx])
                # Save global feature importances
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                top_features = sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1])[:10]

                # Prepare results for display
                result_df = clean_df.copy()
                result_df['hr_probability'] = pred_probs
                result_df['confidence'] = confidence

                id_col = get_id_col(result_df)
                show_n = st.slider("Show top N predictions", min_value=1, max_value=min(20, result_df.shape[0]), value=5)
                top_idx = np.argsort(result_df['hr_probability'])[-show_n:][::-1]

                st.subheader(f"Top {show_n} Home Run Probabilities")
                show_cols = [id_col] if id_col else []
                show_cols += ['hr_probability', 'confidence']
                st.dataframe(result_df.iloc[top_idx][show_cols])

                # Detailed explanations
                st.subheader("Pros & Cons per Prediction (Top N)")

                for rank, idx in enumerate(top_idx):
                    row = result_df.iloc[idx]
                    # Use SHAP for this row if available, otherwise global
                    row_shap = explainer.shap_values(X[idx:idx+1])[0] if idx in sample_idx else mean_abs_shap
                    top_pos = np.argsort(row_shap)[-2:][::-1]
                    top_neg = np.argsort(row_shap)[:2]
                    st.markdown(
                        f"**{rank+1}. {row.get(id_col, 'Row '+str(idx))}** — Probability: `{row['hr_probability']:.3f}`, Confidence: `{row['confidence']:.2f}`"
                    )
                    st.write("Pros (features boosting HR probability):")
                    for j in top_pos:
                        st.write(f"  + **{feature_names[j]}** (value: `{row.get(feature_names[j], 'NA')}`, impact: `{row_shap[j]:.2f}`)")
                    st.write("Cons (features lowering HR probability):")
                    for j in top_neg:
                        st.write(f"  - **{feature_names[j]}** (value: `{row.get(feature_names[j], 'NA')}`, impact: `{row_shap[j]:.2f}`)")
                    st.write("---")

                # Global feature importances
                st.subheader("Top 10 Global Feature Importances (by mean absolute SHAP value)")
                st.table(pd.DataFrame(top_features, columns=['Feature', 'Importance']))

                # Download
                st.download_button(
                    "Download all predictions",
                    result_df.to_csv(index=False),
                    file_name="home_run_predictions.csv"
                )
else:
    st.info("Upload a CSV or Excel file to get started!")
