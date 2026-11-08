
"""
Streamlit application for AIML Assignment 2.

The application loads the five models saved during training.
Users can upload a labelled test dataset, compare model performance,
and inspect the confusion matrix and classification report.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------------
# 1. Project paths and application settings
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
DEFAULT_TEST_FILE = PROJECT_DIR / "test_data.csv"

TARGET_COLUMN = "A16"

FEATURE_COLUMNS = [
    "A15", "A14", "A13", "A12", "A11",
    "A10", "A9", "A8", "A7", "A6",
    "A5", "A4", "A3", "A2", "A1"
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbours": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


# ---------------------------------------------------------
# 2. Streamlit page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Credit Approval Classification",
    page_icon="📊",
    layout="wide"
)

st.title("Credit Approval Classification")
st.markdown(
    """
    **BITS Pilani | M.Tech AIML | Machine Learning Assignment 2**

    This application compares five classification algorithms
    using the UCI Credit Approval dataset.
    """
)

st.divider()


# ---------------------------------------------------------
# 3. Load the saved models
# ---------------------------------------------------------

@st.cache_resource
def load_saved_models():
    """Load all trained models from the models directory."""

    loaded_models = {}

    for model_name, file_name in MODEL_FILES.items():
        model_path = MODEL_DIR / file_name

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        loaded_models[model_name] = joblib.load(model_path)

    return loaded_models


try:
    models = load_saved_models()
    st.success("All five saved models have been loaded successfully.")

except Exception as error:
    st.error(f"Unable to load the saved models: {error}")
    st.stop()


# ---------------------------------------------------------
# 4. Upload or select the test dataset
# ---------------------------------------------------------

st.subheader("1. Select the test dataset")

uploaded_file = st.file_uploader(
    "Upload a labelled test CSV file",
    type=["csv"],
    help="The CSV must contain all 15 input features and the A16 target column."
)

try:
    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)
        st.info("Using the uploaded test dataset.")

    elif DEFAULT_TEST_FILE.exists():
        test_data = pd.read_csv(DEFAULT_TEST_FILE)
        st.info("Using the project's default test_data.csv file.")

    else:
        st.warning("Upload a test CSV file to continue.")
        st.stop()

except Exception as error:
    st.error(f"Unable to read the CSV file: {error}")
    st.stop()


# ---------------------------------------------------------
# 5. Validate the uploaded dataset
# ---------------------------------------------------------

required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

missing_columns = [
    column for column in required_columns
    if column not in test_data.columns
]

if missing_columns:
    st.error(
        "The uploaded dataset is missing these columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

if test_data.empty:
    st.error("The uploaded dataset contains no records.")
    st.stop()

X_test = test_data[FEATURE_COLUMNS].copy()

try:
    y_test = pd.to_numeric(
        test_data[TARGET_COLUMN],
        errors="raise"
    ).astype(int)

except (ValueError, TypeError):
    st.error(
        "The target column A16 must contain numeric labels: "
        "0 for rejected applications and 1 for approved applications."
    )
    st.stop()

if not set(y_test.unique()).issubset({0, 1}):
    st.error("The target column must contain only 0 and 1.")
    st.stop()

if y_test.nunique() < 2:
    st.warning(
        "The test dataset must contain both target classes "
        "to calculate ROC-AUC."
    )


# ---------------------------------------------------------
# 6. Display a preview of the test dataset
# ---------------------------------------------------------

st.subheader("2. Test dataset overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total records", len(test_data))
col2.metric("Input features", len(FEATURE_COLUMNS))
col3.metric("Target classes", y_test.nunique())

with st.expander("View test dataset"):
    st.dataframe(test_data, use_container_width=True)


# ---------------------------------------------------------
# 7. Evaluate all five saved models
# ---------------------------------------------------------

st.subheader("3. Model performance comparison")

evaluation_results = []
predictions = {}

for model_name, model in models.items():

    # Generate predictions using the saved model.
    y_pred = model.predict(X_test)

    # Calculate the probability of the positive class.
    try:
        y_probability = model.predict_proba(X_test)[:, 1]

        if y_test.nunique() == 2:
            auc = roc_auc_score(y_test, y_probability)
        else:
            auc = float("nan")

    except (AttributeError, IndexError, ValueError):
        auc = float("nan")

    # Calculate the six required evaluation metrics.
    result = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    evaluation_results.append(result)
    predictions[model_name] = y_pred


comparison_df = pd.DataFrame(evaluation_results)

st.dataframe(
    comparison_df.style.format({
        "Accuracy": "{:.4f}",
        "AUC": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1 Score": "{:.4f}",
        "MCC": "{:.4f}",
    }),
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# 8. Select a model for detailed evaluation
# ---------------------------------------------------------

st.subheader("4. Detailed model evaluation")

selected_model_name = st.selectbox(
    "Choose a classifier",
    list(models.keys())
)

selected_predictions = predictions[selected_model_name]

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "Accuracy",
    f"{accuracy_score(y_test, selected_predictions):.4f}"
)

metric2.metric(
    "F1 Score",
    f"{f1_score(y_test, selected_predictions, zero_division=0):.4f}"
)

metric3.metric(
    "MCC",
    f"{matthews_corrcoef(y_test, selected_predictions):.4f}"
)


# ---------------------------------------------------------
# 9. Display the confusion matrix
# ---------------------------------------------------------

st.markdown("#### Confusion Matrix")

cm = confusion_matrix(
    y_test,
    selected_predictions,
    labels=[0, 1]
)

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Rejected (0)", "Approved (1)"],
    yticklabels=["Rejected (0)", "Approved (1)"],
    ax=ax
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_title(f"Confusion Matrix - {selected_model_name}")

st.pyplot(fig)
plt.close(fig)


# ---------------------------------------------------------
# 10. Display the classification report
# ---------------------------------------------------------

st.markdown("#### Classification Report")

report = classification_report(
    y_test,
    selected_predictions,
    labels=[0, 1],
    target_names=["Rejected", "Approved"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.style.format("{:.4f}"),
    use_container_width=True
)


# ---------------------------------------------------------
# 11. Application information
# ---------------------------------------------------------

st.divider()

st.caption(
    "The application uses previously trained and saved models. "
    "No model retraining takes place during evaluation."
)
