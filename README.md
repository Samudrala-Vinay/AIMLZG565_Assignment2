# Credit Approval Classification Using ML

**BITS WILP | M. Tech in AIML | AIMLZG565**  
**Course:** Machine Learning S1_26-AIMLZG565  
**Assignment:** Assignment_2  
**Student Name:** Samudrala Vinay  
**BITS ID:** 2026AJ05834

## Live Application

**Streamlit Application:** [Open Credit Approval Classification Dashboard](https://aimlzg565assignment2-5cqla4avy82gq8hndzbihw.streamlit.app/)

**GitHub Repository:** [AIMLZG565_Assignment2](https://github.com/Samudrala-Vinay/AIMLZG565_Assignment2)

## 1. Problem Statement

The objective of this project is to develop and compare multiple supervised machine learning classification algorithms for predicting credit approval decisions.

The project uses the UCI Credit Approval dataset to classify applications into two categories:

- **Approved (+ / 1):** Credit application approved.
- **Rejected (- / 0):** Credit application rejected.

Five classification algorithms are implemented and evaluated using the same dataset and test split. Their performance is compared using six evaluation metrics.

## 2. Dataset Description

**Dataset:** Credit Approval  
**Source:** UCI Machine Learning Repository  
**Dataset ID:** 27  
**Dataset link:** https://archive.ics.uci.edu/dataset/27/credit+approval

| Property | Description |
|---|---|
| Total instances | 690 |
| Input features | 15 |
| Target variable | A16 |
| Classification type | Binary classification |
| Positive class | Approved (1) |
| Negative class | Rejected (0) |

The dataset contains numerical and categorical attributes. Some feature values are missing, so preprocessing is performed before model training.

## 3. Data Preprocessing and Methodology

The following preprocessing operations were performed:

1. The dataset was loaded from the UCI Machine Learning Repository.
2. The target variable A16 was converted into numerical labels:
   - `+` mapped to `1`
   - `-` mapped to `0`
3. Numerical features were processed using median imputation and standardisation.
4. Categorical features were processed using most-frequent-value imputation and one-hot encoding.
5. The dataset was divided into training and testing subsets using an 80:20 split.
6. Stratified sampling was used to preserve the class distribution.
7. The random state was set to 42 for reproducibility.

The resulting dataset contained 552 training instances and 138 testing instances.

The preprocessing operations were incorporated into the model pipelines so that the same transformations are applied during evaluation and application usage.

## 4. Machine Learning Algorithms

The following five classifiers were implemented:

1. **Logistic Regression:** A linear classification algorithm used as a baseline model.
2. **Decision Tree Classifier:** A tree-based model that makes predictions through a sequence of feature-based decisions.
3. **K-Nearest Neighbours (KNN):** A distance-based classifier that predicts a class using nearby training observations.
4. **Gaussian Naive Bayes:** A probabilistic classifier based on Bayes' theorem and the conditional independence assumption.
5. **Support Vector Machine (SVM):** A classification algorithm that uses an RBF kernel to model non-linear decision boundaries.

## 5. Evaluation Metrics

All five classifiers were evaluated using the following metrics:

- **Accuracy:** Proportion of correctly classified observations.
- **AUC:** Area under the ROC curve, measuring the ability to distinguish between the two classes.
- **Precision:** Proportion of positive predictions that are correct.
- **Recall:** Proportion of actual positive observations correctly identified.
- **F1 Score:** Harmonic mean of precision and recall.
- **MCC:** Matthews Correlation Coefficient, which considers all four confusion-matrix outcomes.

## 6. Model Performance Comparison

The following table presents the results obtained on the held-out test dataset.

| Classifier | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8696 | 0.9602 | 0.8525 | 0.8525 | 0.8525 | 0.7356 |
| Decision Tree | 0.8913 | 0.9340 | 0.8382 | 0.9344 | 0.8837 | 0.7863 |
| KNN | 0.8623 | 0.9267 | 0.8750 | 0.8033 | 0.8376 | 0.7205 |
| Naive Bayes | 0.7464 | 0.9030 | 0.9333 | 0.4590 | 0.6154 | 0.5214 |
| SVM | 0.8768 | 0.9557 | 0.8438 | 0.8852 | 0.8640 | 0.7523 |

## 7. Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 86.96% and the highest AUC of 0.9602 among the five classifiers. This indicates strong discrimination between the two classes.

### Decision Tree

The Decision Tree achieved the highest accuracy of 89.13%. It also obtained the highest recall, F1 score and MCC. Its recall of 93.44% indicates that it correctly identified a large proportion of approved applications.

### K-Nearest Neighbours

KNN achieved an accuracy of 86.23%. Its precision was 87.50%, but its recall was comparatively lower at 80.33%. This indicates that the model was more conservative when identifying approved applications.

### Naive Bayes

Naive Bayes achieved the highest precision of 93.33%, but its recall was only 45.90%. Although its positive predictions were relatively reliable, it failed to identify a considerable proportion of actual approved applications.

### Support Vector Machine

SVM achieved an accuracy of 87.68% and an AUC of 0.9557. Its F1 score of 0.8640 indicates a reasonably balanced classification performance.

## 8. Overall Model Selection

Based on the comparison of all six evaluation metrics, the **Decision Tree Classifier** is selected as the overall best-performing model for this experiment.

It achieved:

- Accuracy: 89.13%
- AUC: 0.9340
- Precision: 83.82%
- Recall: 93.44%
- F1 Score: 88.37%
- MCC: 0.7863

Although Logistic Regression achieved the highest AUC, Decision Tree performed better in accuracy, recall, F1 score and MCC.

Therefore, Decision Tree is selected based on its overall performance across the evaluation metrics rather than on a single metric.

## 9. Streamlit Application

A Streamlit application has been developed to demonstrate the trained classification models.

The application provides:

- Uploading a labelled test dataset in CSV format.
- Loading the five previously trained and saved classifiers.
- Comparing all five classifiers using the six evaluation metrics.
- Selecting an individual classifier for detailed evaluation.
- Displaying a confusion matrix.
- Displaying the classification report.

The application loads saved models from the `models` directory. It does not retrain the models during evaluation.

**Live application:** [https://aimlzg565assignment2-5cqla4avy82gq8hndzbihw.streamlit.app/](https://aimlzg565assignment2-5cqla4avy82gq8hndzbihw.streamlit.app/)

## 10. Repository Structure

```text
AIMLZG565_Assignment2/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── test_data.csv
│
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── support_vector_machine.joblib
│
├── notebooks/
│   └── Assignment2_Training.ipynb
│
└── results/
    └── model_comparison.csv
```

## 11. How to Run the Application Locally

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the Streamlit application

```bash
streamlit run app.py
```

The application will open in a browser, normally at `http://localhost:8501`.

## 12. GitHub Repository

**Repository:** https://github.com/Samudrala-Vinay/AIMLZG565_Assignment2

The repository contains the training notebook, saved classification models, evaluation results, test dataset, application source code and dependency list.