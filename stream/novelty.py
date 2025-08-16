import streamlit as st
import pandas as pd
import joblib
import os

# Paths
MODELS_DIR = "models"
TEST_DIR = "test_data"

st.title("🔍 AutoML Model Inference Interface")

# Step 1: Model Type Selection
model_type = st.selectbox("Select Model Type", [
    "AutoSklearn MultiOutput Regression",
    "Custom (SMAC variants)"
])

# Step 2: Select model file
selected_model = None

if model_type == "AutoSklearn MultiOutput Regression":
    selected_model = "autosklearn_model_41477.pkl"
else:
    # Filter only custom SMAC models
    custom_models = sorted([
        f for f in os.listdir(MODELS_DIR)
        if f.endswith(".pkl") and (f.startswith("blood") or f.startswith("credit"))
    ])
    selected_model = st.selectbox("Choose a Custom Model", custom_models)

# Step 3: Load corresponding test set
test_file_map = {
    "autosklearn_model_41477.pkl": "edm_test.csv",
    "blood": "blood_transfusion_test.csv",
    "credit": "credit_g_test.csv"
}

test_file = None
for key in ["blood", "credit"]:
    if selected_model.startswith(key):
        test_file = test_file_map[key]
if model_type == "AutoSklearn MultiOutput Regression":
    test_file = test_file_map["autosklearn_model_41477.pkl"]

# Step 4: Inference
if st.button("🧠 Run Prediction"):
    try:
        # Load model
        model = joblib.load(os.path.join(MODELS_DIR, selected_model))
        st.success(f"✅ Model `{selected_model}` loaded.")

        # Load test data
        test_path = os.path.join(TEST_DIR, test_file)
        test_df = pd.read_csv(test_path)
        st.success(f"✅ Loaded test data: `{test_file}`")
        st.write("📄 Sample test data:", test_df.head())

        # Drop target columns if present
        if model_type == "AutoSklearn MultiOutput Regression":
            y_cols = ['target_0', 'target_1']
        elif selected_model.startswith("blood"):
            y_cols = ["Class"]
        elif selected_model.startswith("credit"):
            y_cols = ["class"]
        else:
            y_cols = []

        X_test = test_df.drop(columns=y_cols, errors='ignore')

        # Run prediction
        preds = model.predict(X_test)

        # Format output
        if preds.ndim == 1:
            result_df = pd.DataFrame({"Prediction": preds}, index=X_test.index)
        else:
            result_df = pd.DataFrame(preds, columns=[f"Output_{i}" for i in range(preds.shape[1])], index=X_test.index)

        st.write("🔢 Predictions:")
        st.dataframe(result_df)

        # Download predictions
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error during inference: {e}")
