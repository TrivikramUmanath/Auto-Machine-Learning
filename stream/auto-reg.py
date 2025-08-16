import streamlit as st
import pandas as pd
import joblib
import os

MODELS_DIR = "models"
TEST_DATA_PATH = "test_data/life_exp_test.csv"

st.title("🔍 AutoML Model Inference Interface")

if st.button("🧠 Run Prediction on Life Expectancy Test Data"):
    try:
        # Load model
        model_path = os.path.join(MODELS_DIR, "autosklearn_model_42372.pkl")
        model = joblib.load(model_path)
        st.success("✅ Model loaded")

        # Load test data
        test_df = pd.read_csv(TEST_DATA_PATH)

        # Predict
        preds = model.predict(test_df)

        # Display predictions
        result_df = pd.DataFrame({"Prediction": preds})
        st.dataframe(result_df)

        # Download option
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error during inference: {e}")

