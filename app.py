import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from io import StringIO

st.set_page_config(page_title="Omniscience MLB Advanced Model", layout="wide")
st.title("Omniscience MLB Advanced Model (Multi-File, Auto-Clean)")

@st.cache_data
def load_and_combine(files):
    dfs = []
    for uploaded_file in files:
        try:
            # Try reading as CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {uploaded_file.name} as CSV.")
        except Exception as e_csv:
            uploaded_file.seek(0)
            try:
                # Try reading as Excel
                df = pd.read_excel(uploaded_file)
                st.success(f"Loaded {uploaded_file.name} as Excel.")
            except Exception as e_excel:
                st.error(f"Could not read {uploaded_file.name}: Not a valid CSV or Excel file.")
                continue
        # Only keep numeric columns (drop empty or all-NaN frames)
        if not df.empty and len(df.select_dtypes(include=[np.number]).columns) > 0:
            dfs.append(df)
        else:
            st.warning(f"{uploaded_file.name} has no numeric columns and was skipped.")
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates().reset_index(drop=True)
        return combined
    else:
        return pd.DataFrame()

class OmniscienceModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def feature_importances(self, features):
        return sorted(zip(features, self.model.feature_importances_), key=lambda x: -x[1])

uploaded_files = st.file_uploader(
    "Upload your MLB metrics files (CSV or Excel, multiple allowed)", 
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    df = load_and_combine(uploaded_files)
    if df.empty:
        st.error("No valid data loaded. Please check your files.")
    else:
        st.success(f"{len(df)} rows loaded from {len(uploaded_files)} file(s)!")
        st.dataframe(df.head())

        # Allow user to select features and (optionally) target
        all_columns = list(df.select_dtypes(include=[np.number]).columns)
        default_features = [col for col in all_columns if col.lower() not in ['is_home_run', 'target', 'home_run']]
        features = st.multiselect("Select features for prediction:", all_columns, default=default_features)
        
        # Try to auto-detect target column
        possible_targets = [col for col in df.columns if col.lower() in ['is_home_run', 'home_run', 'target']]
        target = st.selectbox("Select target column (label):", ["None"] + possible_targets)
        
        if features and target != "None":
            X = df[features]
            y = df[target]
            # Remove NaNs for modeling
            mask = X.notnull().all(axis=1) & y.notnull()
            X = X[mask]
            y = y[mask]
            if len(X) == 0:
                st.error("No rows with complete data after cleaning. Please check your files/columns.")
            else:
                model = OmniscienceModel()
                model.train(X, y)
                st.success("Model trained on your data!")
                preds = model.predict_proba(X)
                df_clean = df.loc[X.index].copy()
                df_clean['Home Run Probability'] = preds

                # Recommendations based on threshold
                df_clean['Recommendation'] = np.where(
                    df_clean['Home Run Probability'] > 0.8, "Strong Bet",
                    np.where(df_clean['Home Run Probability'] > 0.6, "Consider", "Avoid")
                )
                
                st.write("### Detailed Recommendations")
                for i, row in df_clean.iterrows():
                    st.markdown(f"""
                        **Row {i+1}**
                        - Home Run Probability: `{row['Home Run Probability']:.2%}`
                        - Recommendation: `{row['Recommendation']}`
                        - {" | ".join([f"{f}: {row[f]}" for f in features])}
                    """)
                    st.markdown("---")
                
                # Feature importances
                st.write("### Feature Importances")
                st.table(model.feature_importances(features))
                
                # Download results
                csv = df_clean.to_csv(index=False)
                st.download_button('Download Results as CSV', csv, file_name='omniscience_predictions.csv')
        elif features:
            st.info("Please select a target column with binary values (1 for home run, 0 otherwise).")
        else:
            st.info("Please select at least one feature column.")
else:
    st.info("Please upload your MLB metrics files to begin.")
