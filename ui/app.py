# ui/app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

MODEL_PATH = os.path.join("models", "final_model.pkl")  # Path to the trained model 

@st.cache_resource
def load_artifact():
    if not os.path.exists(MODEL_PATH):
        st.warning(f"Model not found at {MODEL_PATH}. Train a model first and place the .pkl file there.")
        return None, None
    loaded = joblib.load(MODEL_PATH)
    pipeline = loaded['pipeline']
    meta = loaded['metadata']
    return pipeline, meta

pipeline, meta = load_artifact()
st.title("Heart Disease Risk Predictor")

if pipeline is None:
    st.info("Run the training script first to generate models/final_model.pkl, then refresh.")
else:
    # --- Mode Selection ---
    mode = st.radio("Choose input mode:", ["Manual Entry", "Upload CSV"])

    # ==========================================================
    # Mode 1: Manual entry (existing form)
    # ==========================================================
    if mode == "Manual Entry":
        st.markdown("Enter patient data below and click **Predict**.")

        def input_widget_for_feature(col, meta):
            if col in meta['numeric_features']:
                return st.number_input(col, value=0.0, format="%.3f")
            elif col in meta['categorical_features']:
                opts = meta['categorical_options'].get(col, [])
                if len(opts) == 0:
                    return st.text_input(col, value="")
                else:
                    opts_str = [str(x) for x in opts]
                    return st.selectbox(col, options=opts_str)
            else:
                return st.text_input(col, value="")

        with st.form(key='predict_form'):
            user_vals = {}
            for feat in meta['feature_order']:
                if feat == 'target':
                    continue
                user_vals[feat] = input_widget_for_feature(feat, meta)
            submit = st.form_submit_button("Predict")

        if submit:
            row = {}
            for k, v in user_vals.items():
                if k in meta['numeric_features']:
                    try:
                        row[k] = float(v)
                    except:
                        row[k] = np.nan
                else:
                    row[k] = v
            X = pd.DataFrame([row])
            proba = pipeline.predict_proba(X)[:, 1][0] if hasattr(pipeline.named_steps['clf'], "predict_proba") else None
            pred = pipeline.predict(X)[0]
            st.write("**Prediction (0 = no disease, 1 = disease)**:", int(pred))
            if proba is not None:
                st.write("**Predicted probability of disease:** {:.2%}".format(proba))

    # ==========================================================
    # Mode 2: CSV Upload
    # ==========================================================
    elif mode == "Upload CSV":
        st.markdown("Upload a CSV file with patient records for bulk prediction.")

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)

                preds = pipeline.predict(input_df)
                if hasattr(pipeline.named_steps['clf'], "predict_proba"):
                    probs = pipeline.predict_proba(input_df)[:, 1]
                    probs = (probs * 100).round(2)

                else:
                    probs = None

                input_df["prediction"] = preds
                if probs is not None:
                    input_df["probability(%)"] = probs

                st.success("✅ Predictions completed!")
                st.dataframe(input_df)

                csv_out = input_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Predictions as CSV",
                    data=csv_out,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Error processing file: {e}")
