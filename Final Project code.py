import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
pima = pd.read_csv("PIMA diabetes dataset_A.csv")

print("PIMA dataset shape",pima.shape)



print("Pima first 5 rows:")
display(pima.head())


print("Pima info:")
pima.info()



print("Missing values in Pima:")
display(pima.isnull().sum())


invalid_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in invalid_zero_cols:
    if col in pima.columns:
        print(f"{col}: {(pima[col] == 0).sum()} zero values")
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

pima[invalid_zero_cols] = imputer.fit_transform(pima[invalid_zero_cols])
print(pima[invalid_zero_cols].isnull().sum())
# For Pima, target is usually called Outcome
print("Pima target distribution:")
display(pima["Outcome"].value_counts())
display(pima["Outcome"].value_counts(normalize=True) * 100)



plt.figure(figsize=(5,4))
sns.countplot(data=pima, x="Outcome")
plt.title("Target Distribution - Pima Dataset")
plt.xlabel("Diabetes Outcome")
plt.ylabel("Count")
plt.show()



pima.hist(figsize=(12, 10), bins=20)
plt.suptitle("Feature Distributions - Pima Dataset")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(pima.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap - Pima Dataset")
plt.show()


summary_table = pd.DataFrame({
    "Dataset": ["Pima Indians Diabetes Dataset"],
    "Samples": [pima.shape[0]],
    "Columns": [pima.shape[1]],
    "Target Variable": ["Outcome"],
    "Feature Type": ["Mostly numerical clinical measurements"]
})

display(summary_table)
# ============================================================
# 15. CREATE DATASET COMPARISON TABLE
# ============================================================

comparison_table = pd.DataFrame({
    "Aspect": [
        "Main purpose",
        "Feature type",
        "Clinical information type",
        "Target variable",
        "Use in this project"
    ],
    "Pima Dataset": [
        "Diabetes prediction",
        "Numerical",
        "Physiological measurements such as glucose, BMI, insulin, age",
        "Outcome",
        "Main training and internal validation dataset"
    ],
   
})

display(comparison_table)


summary_table.to_csv("results/dataset_summary_table.csv", index=False)
comparison_table.to_csv("results/dataset_comparison_table.csv", index=False)

print("Tables saved successfully.")


with open("results/initial_dataset_report.txt", "w") as f:
    f.write("Initial Dataset Exploration Report\n")
    f.write("=================================\n\n")
    
    f.write(f"Pima Dataset Shape: {pima.shape}\n")
    f.write(f"Pima Columns: {pima.columns.tolist()}\n\n")
    

    f.write("Missing Values - Pima:\n")
    f.write(str(pima.isnull().sum()))
    f.write("\n\n")
    
    
    f.write("Invalid Zero Values in Pima:\n")
    for col in invalid_zero_cols:
        if col in pima.columns:
            f.write(f"{col}: {(pima[col] == 0).sum()} zero values\n")
    
    f.write("\nPima Target Distribution:\n")
    f.write(str(pima["Outcome"].value_counts()))
    

print("Initial report saved successfully.")
# Features
X = pima.drop("Outcome", axis=1)

# Target
y = pima["Outcome"]

print("Features:", X.shape)
print("Target:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)

log_reg = LogisticRegression(random_state=42)


log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)

y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]
accuracy = accuracy_score(y_test, y_pred_lr)
precision = precision_score(y_test, y_pred_lr)
recall = recall_score(y_test, y_pred_lr)
f1 = f1_score(y_test, y_pred_lr)
roc_auc = roc_auc_score(y_test, y_prob_lr)
brier = brier_score_loss(y_test, y_prob_lr)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"Brier    : {brier:.4f}")
print(classification_report(y_test, y_pred_lr))
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_lr,
    cmap="Blues"
)

plt.title("Logistic Regression - Confusion Matrix")
plt.show()
logistic_results = {
    "Model": "Logistic Regression",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC-AUC": roc_auc,
    "Brier Score": brier
}

logistic_results
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)


rf.fit(X_train, y_train)
# Predicted labels
y_pred_rf = rf.predict(X_test)

# Predicted probabilities
y_prob_rf = rf.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)
roc_auc = roc_auc_score(y_test, y_prob_rf)
brier = brier_score_loss(y_test, y_prob_rf)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"Brier    : {brier:.4f}")
print(classification_report(y_test, y_pred_rf))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_rf,
    cmap="Greens"
)

plt.title("Random Forest - Confusion Matrix")
plt.show()
random_forest_results = {
    "Model": "Random Forest",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC-AUC": roc_auc,
    "Brier Score": brier
}

random_forest_results

# XGBoost Baseline Model

from xgboost import XGBClassifier


xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)


xgb.fit(X_train, y_train)
# Predictions

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
# Evaluation metrics

accuracy = accuracy_score(y_test, y_pred_xgb)
precision = precision_score(y_test, y_pred_xgb)
recall = recall_score(y_test, y_pred_xgb)
f1 = f1_score(y_test, y_pred_xgb)
roc_auc = roc_auc_score(y_test, y_prob_xgb)
brier = brier_score_loss(y_test, y_prob_xgb)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"Brier    : {brier:.4f}")


print(classification_report(y_test, y_pred_xgb))


ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_xgb,
    cmap="Oranges"
)

plt.title("XGBoost - Confusion Matrix")
plt.show()
# Save results

xgboost_results = {
    "Model": "XGBoost",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC-AUC": roc_auc,
    "Brier Score": brier
}

xgboost_results
from sklearn.model_selection import GridSearchCV, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
log_reg_param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l2"],
    "solver": ["liblinear", "lbfgs"],
    "class_weight": [None, "balanced"]
}

log_reg_grid = GridSearchCV(
    estimator=LogisticRegression(random_state=42, max_iter=1000),
    param_grid=log_reg_param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)

log_reg_grid.fit(X_train_scaled, y_train)

print("Best Parameters:", log_reg_grid.best_params_)
print("Best CV F1 Score:", log_reg_grid.best_score_)
log_reg_param_grid_2 = {
    "C": [0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],
    "class_weight": ["balanced"]
}
log_reg_grid_2 = GridSearchCV(
    estimator=LogisticRegression(random_state=42, max_iter=1000),
    param_grid=log_reg_param_grid_2,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)

log_reg_grid_2.fit(X_train_scaled, y_train)

print("Best Parameters:", log_reg_grid_2.best_params_)
print("Best CV F1 Score:", log_reg_grid_2.best_score_)
best_log_reg = log_reg_grid.best_estimator_

y_pred_lr_tuned = best_log_reg.predict(X_test_scaled)
y_prob_lr_tuned = best_log_reg.predict_proba(X_test_scaled)[:, 1]

print("Tuned Logistic Regression Results")
print("Accuracy :", accuracy_score(y_test, y_pred_lr_tuned))
print("Precision:", precision_score(y_test, y_pred_lr_tuned))
print("Recall   :", recall_score(y_test, y_pred_lr_tuned))
print("F1 Score :", f1_score(y_test, y_pred_lr_tuned))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob_lr_tuned))
print("Brier    :", brier_score_loss(y_test, y_prob_lr_tuned))

rf_param_grid_2 = {
    "n_estimators": [150, 200, 250, 300, 400],
    "max_depth": [4, 5, 6, 7, 8],
    "min_samples_split": [6, 8, 10, 12, 15],
    "min_samples_leaf": [1, 2, 3],
    "max_features": ["sqrt", "log2", None],
    "class_weight": ["balanced"]
}

rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid_2,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print("Best Parameters:", rf_grid.best_params_)
print("Best CV F1 Score:", rf_grid.best_score_)
best_rf = rf_grid.best_estimator_

y_pred_rf_tuned = best_rf.predict(X_test)
y_prob_rf_tuned = best_rf.predict_proba(X_test)[:, 1]

print("Tuned Random Forest Results")
print("Accuracy :", accuracy_score(y_test, y_pred_rf_tuned))
print("Precision:", precision_score(y_test, y_pred_rf_tuned))
print("Recall   :", recall_score(y_test, y_pred_rf_tuned))
print("F1 Score :", f1_score(y_test, y_pred_rf_tuned))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob_rf_tuned))
print("Brier    :", brier_score_loss(y_test, y_prob_rf_tuned))
xgb_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

xgb_grid = GridSearchCV(
    estimator=XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    ),
    param_grid=xgb_param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)

xgb_grid.fit(X_train, y_train)

print("Best Parameters:", xgb_grid.best_params_)
print("Best CV F1 Score:", xgb_grid.best_score_)
best_xgb = xgb_grid.best_estimator_

y_pred_xgb_tuned = best_xgb.predict(X_test)
y_prob_xgb_tuned = best_xgb.predict_proba(X_test)[:, 1]

print("Tuned XGBoost Results")
print("Accuracy :", accuracy_score(y_test, y_pred_xgb_tuned))
print("Precision:", precision_score(y_test, y_pred_xgb_tuned))
print("Recall   :", recall_score(y_test, y_pred_xgb_tuned))
print("F1 Score :", f1_score(y_test, y_pred_xgb_tuned))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob_xgb_tuned))
print("Brier    :", brier_score_loss(y_test, y_prob_xgb_tuned))
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
feature_names = X_train.columns.tolist()


X_train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=feature_names,
    index=X_train.index
)

X_test_scaled_df = pd.DataFrame(
    X_test_scaled,
    columns=feature_names,
    index=X_test.index
)

# Raw/unscaled data used by Random Forest and XGBoost
X_train_df = X_train.copy()
X_test_df = X_test.copy()

print(feature_names)
def get_positive_class_shap_values(shap_output):


    values = (
        shap_output.values
        if hasattr(shap_output, "values")
        else shap_output
    )

   
    # [values_for_class_0, values_for_class_1]
    if isinstance(values, list):
        return np.asarray(values[1])

    values = np.asarray(values)

    
    if values.ndim == 3:
        return values[:, :, 1]

    if values.ndim == 2:
        return values

    raise ValueError(f"Unexpected SHAP output shape: {values.shape}")
# Background data for Logistic Regression
lr_background = shap.sample(
    X_train_scaled_df,
    100,
    random_state=42
)


shap_explainer_lr = shap.LinearExplainer(
    best_log_reg,
    lr_background
)

# Explain all test instances
shap_output_lr = shap_explainer_lr(X_test_scaled_df)

# Extract attribution matrix
shap_values_lr = get_positive_class_shap_values(shap_output_lr)

print("Logistic Regression SHAP shape:", shap_values_lr.shape)
shap.summary_plot(
    shap_values_lr,
    X_test_scaled_df,
    feature_names=feature_names,
    show=True
)
shap.summary_plot(
    shap_values_lr,
    X_test_scaled_df,
    feature_names=feature_names,
    plot_type="bar",
    show=True
)
patient_index = 0

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_lr[patient_index],
        base_values=np.asarray(shap_output_lr.base_values).reshape(-1)[patient_index]
        if np.asarray(shap_output_lr.base_values).size > 1
        else np.asarray(shap_output_lr.base_values).item(),
        data=X_test_scaled_df.iloc[patient_index].values,
        feature_names=feature_names
    ),
    max_display=8
)
patient_index = 0

local_lr = pd.Series(
    shap_values_lr[patient_index],
    index=feature_names
).sort_values(key=np.abs)

local_lr.plot(kind="barh", figsize=(8, 5))
plt.title("Logistic Regression SHAP Explanation — Test Patient 0")
plt.xlabel("SHAP attribution")
plt.tight_layout()
plt.show()
shap_explainer_rf = shap.TreeExplainer(best_rf)

shap_output_rf = shap_explainer_rf(X_test_df)

shap_values_rf = get_positive_class_shap_values(shap_output_rf)

print("Random Forest SHAP shape:", shap_values_rf.shape)
shap.summary_plot(
    shap_values_rf,
    X_test_df,
    feature_names=feature_names,
    show=True
)
patient_index = 0

local_rf = pd.Series(
    shap_values_rf[patient_index],
    index=feature_names
).sort_values(key=np.abs)

local_rf.plot(kind="barh", figsize=(8, 5))
plt.title("Random Forest SHAP Explanation — Test Patient 0")
plt.xlabel("SHAP attribution")
plt.tight_layout()
plt.show()
import xgboost as xgb
import numpy as np
import pandas as pd

# Convert test data into XGBoost's native DMatrix format
xgb_test_matrix = xgb.DMatrix(
    X_test_df,
    feature_names=feature_names
)

# Get SHAP contributions directly from XGBoost
xgb_contributions = best_xgb.get_booster().predict(
    xgb_test_matrix,
    pred_contribs=True
)

print("Raw contribution shape:", xgb_contributions.shape)
shap_values_xgb = xgb_contributions[:, :-1]
xgb_base_values = xgb_contributions[:, -1]

print("XGBoost SHAP values shape:", shap_values_xgb.shape)
print("XGBoost base values shape:", xgb_base_values.shape)
shap.summary_plot(
    shap_values_xgb,
    X_test_df,
    feature_names=feature_names
)
shap.summary_plot(
    shap_values_xgb,
    X_test_df,
    feature_names=feature_names,
    plot_type="bar"
)
patient_index = 0

xgb_local_explanation = shap.Explanation(
    values=shap_values_xgb[patient_index],
    base_values=xgb_base_values[patient_index],
    data=X_test_df.iloc[patient_index].values,
    feature_names=feature_names
)

shap.plots.waterfall(
    xgb_local_explanation,
    max_display=8
)
lime_explainer_lr = LimeTabularExplainer(
    training_data=X_train_scaled_df.values,
    feature_names=feature_names,
    class_names=["Non-diabetic", "Diabetic"],
    mode="classification",
    discretize_continuous=False,
    random_state=42
)
lime_explainer_tree = LimeTabularExplainer(
    training_data=X_train_df.values,
    feature_names=feature_names,
    class_names=["Non-diabetic", "Diabetic"],
    mode="classification",
    discretize_continuous=False,
    random_state=42
)
patient_index = 0

lime_exp_lr = lime_explainer_lr.explain_instance(
    data_row=X_test_scaled_df.iloc[patient_index].values,
    predict_fn=best_log_reg.predict_proba,
    labels=(1,),
    num_features=len(feature_names),
    num_samples=5000
)

print("Logistic Regression LIME explanation:")
print(lime_exp_lr.as_list(label=1))
lime_exp_lr.as_pyplot_figure(label=1)
plt.title("Logistic Regression LIME Explanation — Test Patient 0")
plt.tight_layout()
plt.show()
patient_index = 0

lime_exp_rf = lime_explainer_tree.explain_instance(
    data_row=X_test_df.iloc[patient_index].values,
    predict_fn=best_rf.predict_proba,
    labels=(1,),
    num_features=len(feature_names),
    num_samples=5000
)

print("Random Forest LIME explanation:")
print(lime_exp_rf.as_list(label=1))
lime_exp_rf.as_pyplot_figure(label=1)
plt.title("Random Forest LIME Explanation — Test Patient 0")
plt.tight_layout()
plt.show()
patient_index = 0

lime_exp_xgb = lime_explainer_tree.explain_instance(
    data_row=X_test_df.iloc[patient_index].values,
    predict_fn=best_xgb.predict_proba,
    labels=(1,),
    num_features=len(feature_names),
    num_samples=5000
)

print("XGBoost LIME explanation:")
print(lime_exp_xgb.as_list(label=1))
lime_exp_xgb.as_pyplot_figure(label=1)
plt.title("XGBoost LIME Explanation — Test Patient 0")
plt.tight_layout()
plt.show()
def lime_explanation_to_vector(explanation, n_features, label=1):
   

    vector = np.zeros(n_features, dtype=float)

    explanation_map = explanation.as_map()

    if label not in explanation_map:
        raise KeyError(
            f"Label {label} not found. Available labels: "
            f"{list(explanation_map.keys())}"
        )

    for feature_index, weight in explanation_map[label]:
        vector[feature_index] = weight

    return vector
def generate_lime_matrix(
    explainer,
    model,
    X_data,
    n_features,
    label=1,
    num_samples=5000
):
    

    lime_matrix = np.zeros(
        (len(X_data), n_features),
        dtype=float
    )

    for row_position in range(len(X_data)):
        explanation = explainer.explain_instance(
            data_row=X_data.iloc[row_position].values,
            predict_fn=model.predict_proba,
            labels=(label,),
            num_features=n_features,
            num_samples=num_samples
        )

        lime_matrix[row_position] = lime_explanation_to_vector(
            explanation=explanation,
            n_features=n_features,
            label=label
        )

        if (row_position + 1) % 20 == 0:
            print(
                f"Completed {row_position + 1}/{len(X_data)} explanations"
            )

    return lime_matrix
lime_values_lr = generate_lime_matrix(
    explainer=lime_explainer_lr,
    model=best_log_reg,
    X_data=X_test_scaled_df,
    n_features=len(feature_names),
    label=1,
    num_samples=5000
)

print("LR LIME matrix shape:", lime_values_lr.shape)
lime_values_rf = generate_lime_matrix(
    explainer=lime_explainer_tree,
    model=best_rf,
    X_data=X_test_df,
    n_features=len(feature_names),
    label=1,
    num_samples=5000
)

print("RF LIME matrix shape:", lime_values_rf.shape)
lime_values_xgb = generate_lime_matrix(
    explainer=lime_explainer_tree,
    model=best_xgb,
    X_data=X_test_df,
    n_features=len(feature_names),
    label=1,
    num_samples=5000
)

print("XGB LIME matrix shape:", lime_values_xgb.shape)
explanation_shapes = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "SHAP Shape": [
        str(shap_values_lr.shape),
        str(shap_values_rf.shape),
        str(shap_values_xgb.shape)
    ],
    "LIME Shape": [
        str(lime_values_lr.shape),
        str(lime_values_rf.shape),
        str(lime_values_xgb.shape)
    ]
})

display(explanation_shapes)
explanation_matrices = {
    "Logistic Regression": (shap_values_lr, lime_values_lr),
    "Random Forest": (shap_values_rf, lime_values_rf),
    "XGBoost": (shap_values_xgb, lime_values_xgb)
}

for model_name, (shap_matrix, lime_matrix) in explanation_matrices.items():

    print(model_name)
    print("SHAP shape:", shap_matrix.shape)
    print("LIME shape:", lime_matrix.shape)

    assert shap_matrix.shape == lime_matrix.shape, (
        f"SHAP and LIME shapes do not match for {model_name}"
    )

    assert shap_matrix.shape[1] == len(feature_names), (
        f"Unexpected number of features for {model_name}"
    )

    assert np.isfinite(shap_matrix).all(), (
        f"SHAP contains NaN or infinite values for {model_name}"
    )

    assert np.isfinite(lime_matrix).all(), (
        f"LIME contains NaN or infinite values for {model_name}"
    )

    print("Validation passed.\n")

def calculate_spearman_agreement(shap_vector, lime_vector):
    """
    Calculate SHAP-LIME rank agreement for one patient.

    Absolute attribution values are used because this metric evaluates
    agreement in feature importance ranking.
    """

    shap_importance = np.abs(np.asarray(shap_vector, dtype=float))
    lime_importance = np.abs(np.asarray(lime_vector, dtype=float))

    correlation, p_value = spearmanr(
        shap_importance,
        lime_importance
    )

    # Spearman may be undefined if every value in one vector is identical.
    if np.isnan(correlation):
        correlation = 0.0

    return correlation, p_value
patient_index = 0

spearman_value, p_value = calculate_spearman_agreement(
    shap_values_lr[patient_index],
    lime_values_lr[patient_index]
)

print("Spearman correlation:", spearman_value)
print("p-value:", p_value)
def normalise_spearman(correlation):
    return (correlation + 1.0) / 2.0
normalised_value = normalise_spearman(spearman_value)

print("Original Spearman:", spearman_value)
print("Normalised Spearman:", normalised_value)
def calculate_top_k_overlap(shap_vector, lime_vector, k=3):
    """
    Calculate the proportion of shared features in the SHAP and LIME
    top-k importance sets for one patient.
    """

    shap_importance = np.abs(np.asarray(shap_vector, dtype=float))
    lime_importance = np.abs(np.asarray(lime_vector, dtype=float))

    if not 1 <= k <= len(shap_importance):
        raise ValueError(
            f"k must be between 1 and {len(shap_importance)}"
        )

    shap_top_k = np.argsort(shap_importance)[-k:]
    lime_top_k = np.argsort(lime_importance)[-k:]

    shared_features = set(shap_top_k).intersection(set(lime_top_k))

    overlap_score = len(shared_features) / k

    return overlap_score, shap_top_k, lime_top_k
top_k_score, shap_top_k, lime_top_k = calculate_top_k_overlap(
    shap_values_lr[patient_index],
    lime_values_lr[patient_index],
    k=3
)

print("Top-3 overlap:", top_k_score)

print(
    "SHAP top features:",
    [feature_names[i] for i in shap_top_k[::-1]]
)

print(
    "LIME top features:",
    [feature_names[i] for i in lime_top_k[::-1]]
)
def calculate_patient_agreement(
    shap_vector,
    lime_vector,
    k=3,
    spearman_weight=0.5,
    top_k_weight=0.5
):
    """
    Calculate SHAP-LIME agreement for one patient.

    Returns:
    - original Spearman correlation
    - normalised Spearman score
    - top-k overlap
    - combined agreement score
    """

    if not np.isclose(spearman_weight + top_k_weight, 1.0):
        raise ValueError("The metric weights must sum to 1.")

    spearman_raw, spearman_p = calculate_spearman_agreement(
        shap_vector,
        lime_vector
    )

    spearman_normalised = normalise_spearman(spearman_raw)

    top_k_overlap, shap_top_k, lime_top_k = calculate_top_k_overlap(
        shap_vector,
        lime_vector,
        k=k
    )

    agreement_score = (
        spearman_weight * spearman_normalised
        + top_k_weight * top_k_overlap
    )

    return {
        "spearman_raw": spearman_raw,
        "spearman_p_value": spearman_p,
        "spearman_normalised": spearman_normalised,
        "top_k_overlap": top_k_overlap,
        "agreement_score": agreement_score,
        "shap_top_k": shap_top_k,
        "lime_top_k": lime_top_k
    }
patient_agreement = calculate_patient_agreement(
    shap_values_lr[0],
    lime_values_lr[0],
    k=3
)

patient_agreement
def calculate_agreement_matrix(
    shap_matrix,
    lime_matrix,
    feature_names,
    model_name,
    k=3,
    spearman_weight=0.5,
    top_k_weight=0.5
):
    """
    Calculate SHAP-LIME agreement for every test patient of one model.
    """

    shap_matrix = np.asarray(shap_matrix, dtype=float)
    lime_matrix = np.asarray(lime_matrix, dtype=float)

    if shap_matrix.shape != lime_matrix.shape:
        raise ValueError(
            f"Shape mismatch: SHAP {shap_matrix.shape}, "
            f"LIME {lime_matrix.shape}"
        )

    results = []

    for patient_position in range(shap_matrix.shape[0]):

        patient_result = calculate_patient_agreement(
            shap_vector=shap_matrix[patient_position],
            lime_vector=lime_matrix[patient_position],
            k=k,
            spearman_weight=spearman_weight,
            top_k_weight=top_k_weight
        )

        shap_top_features = [
            feature_names[i]
            for i in patient_result["shap_top_k"][::-1]
        ]

        lime_top_features = [
            feature_names[i]
            for i in patient_result["lime_top_k"][::-1]
        ]

        results.append({
            "Model": model_name,
            "Patient_Position": patient_position,
            "Original_Row_Index": X_test_df.index[patient_position],
            "Spearman_Raw": patient_result["spearman_raw"],
            "Spearman_P_Value": patient_result["spearman_p_value"],
            "Spearman_Normalised": patient_result[
                "spearman_normalised"
            ],
            f"Top_{k}_Overlap": patient_result["top_k_overlap"],
            "Agreement_Score": patient_result["agreement_score"],
            "SHAP_Top_Features": ", ".join(shap_top_features),
            "LIME_Top_Features": ", ".join(lime_top_features)
        })

    return pd.DataFrame(results)
agreement_lr = calculate_agreement_matrix(
    shap_matrix=shap_values_lr,
    lime_matrix=lime_values_lr,
    feature_names=feature_names,
    model_name="Logistic Regression",
    k=3
)

agreement_lr.head()
agreement_rf = calculate_agreement_matrix(
    shap_matrix=shap_values_rf,
    lime_matrix=lime_values_rf,
    feature_names=feature_names,
    model_name="Random Forest",
    k=3
)

agreement_rf.head()
agreement_xgb = calculate_agreement_matrix(
    shap_matrix=shap_values_xgb,
    lime_matrix=lime_values_xgb,
    feature_names=feature_names,
    model_name="XGBoost",
    k=3
)

agreement_xgb.head()
agreement_all = pd.concat(
    [
        agreement_lr,
        agreement_rf,
        agreement_xgb
    ],
    ignore_index=True
)

print(agreement_all.shape)
agreement_all.head()
agreement_summary = (
    agreement_all
    .groupby("Model")
    .agg(
        Mean_Spearman=("Spearman_Raw", "mean"),
        Median_Spearman=("Spearman_Raw", "median"),
        Std_Spearman=("Spearman_Raw", "std"),
        Mean_Top3_Overlap=("Top_3_Overlap", "mean"),
        Median_Top3_Overlap=("Top_3_Overlap", "median"),
        Mean_Agreement=("Agreement_Score", "mean"),
        Median_Agreement=("Agreement_Score", "median"),
        Std_Agreement=("Agreement_Score", "std"),
        Minimum_Agreement=("Agreement_Score", "min"),
        Maximum_Agreement=("Agreement_Score", "max")
    )
    .round(4)
    .reset_index()
)

agreement_summary
def categorise_agreement(score):
    if score >= 0.80:
        return "High"
    elif score >= 0.60:
        return "Moderate"
    else:
        return "Low"
agreement_all["Agreement_Level"] = (
    agreement_all["Agreement_Score"]
    .apply(categorise_agreement)
)
agreement_level_summary = pd.crosstab(
    agreement_all["Model"],
    agreement_all["Agreement_Level"],
    normalize="index"
).mul(100).round(2)

agreement_level_summary
agreement_all.boxplot(
    column="Agreement_Score",
    by="Model",
    figsize=(9, 6)
)

plt.title("SHAP–LIME Agreement by Model")
plt.suptitle("")
plt.xlabel("Model")
plt.ylabel("Agreement Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
mean_agreement = (
    agreement_all
    .groupby("Model")["Agreement_Score"]
    .mean()
    .sort_values(ascending=False)
)

mean_agreement.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Mean SHAP–LIME Agreement by Model")
plt.xlabel("Model")
plt.ylabel("Mean Agreement Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
lowest_agreement_cases = (
    agreement_all
    .sort_values("Agreement_Score")
    .head(10)
)

lowest_agreement_cases[
    [
        "Model",
        "Patient_Position",
        "Original_Row_Index",
        "Spearman_Raw",
        "Top_3_Overlap",
        "Agreement_Score",
        "SHAP_Top_Features",
        "LIME_Top_Features"
    ]
]
highest_agreement_cases = (
    agreement_all
    .sort_values("Agreement_Score", ascending=False)
    .head(10)
)

highest_agreement_cases[
    [
        "Model",
        "Patient_Position",
        "Original_Row_Index",
        "Spearman_Raw",
        "Top_3_Overlap",
        "Agreement_Score",
        "SHAP_Top_Features",
        "LIME_Top_Features"
    ]
]
# Number of perturbed versions generated for each patient
N_PERTURBATIONS = 20

# Noise strength as a proportion of the training-set standard deviation
NOISE_LEVEL = 0.10

# Number of important features used for top-k overlap
TOP_K = 3

# LIME neighbourhood size
LIME_NUM_SAMPLES = 5000

# Reproducibility seed
RANDOM_STATE = 42
non_negative_features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]
integer_features = ["Pregnancies"]
train_feature_std = X_train_df.std(ddof=0)
train_feature_min = X_train_df.min()
train_feature_max = X_train_df.max()

perturbation_statistics = pd.DataFrame({
    "Feature": feature_names,
    "Training_Std": train_feature_std[feature_names].values,
    "Noise_Std": (
        NOISE_LEVEL * train_feature_std[feature_names]
    ).values,
    "Training_Min": train_feature_min[feature_names].values,
    "Training_Max": train_feature_max[feature_names].values
})

display(perturbation_statistics)
def generate_perturbed_patients(
    patient,
    training_std,
    training_min,
    training_max,
    n_perturbations=10,
    noise_level=0.05,
    random_state=42
):
    """
    Generate slightly perturbed versions of one patient.

    Parameters
    ----------
    patient : pandas Series
        Original patient in the unscaled clinical feature space.

    training_std : pandas Series
        Feature standard deviations calculated from the training set.

    training_min : pandas Series
        Minimum values from the training set.

    training_max : pandas Series
        Maximum values from the training set.

    n_perturbations : int
        Number of perturbed patients to generate.

    noise_level : float
        Noise scale as a proportion of each feature's training standard
        deviation.

    random_state : int
        Seed used for reproducibility.

    Returns
    -------
    pandas DataFrame
        Perturbed patients in the original unscaled feature space.
    """

    rng = np.random.default_rng(random_state)

    patient = patient[feature_names].astype(float)

    # Feature-specific noise standard deviations
    noise_std = (
        training_std[feature_names].values
        * noise_level
    )

    # Generate Gaussian noise
    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=(n_perturbations, len(feature_names))
    )

    # Add the noise to the original patient
    perturbed_values = patient.values + noise

    perturbed_df = pd.DataFrame(
        perturbed_values,
        columns=feature_names
    )

    # Restrict perturbed values to ranges observed in training data
    for feature in feature_names:
        perturbed_df[feature] = perturbed_df[feature].clip(
            lower=training_min[feature],
            upper=training_max[feature]
        )

    # Pregnancies represents a count
    perturbed_df["Pregnancies"] = (
        perturbed_df["Pregnancies"]
        .round()
        .astype(int)
    )

    return perturbed_df
patient_index = 0

example_perturbations = generate_perturbed_patients(
    patient=X_test_df.iloc[patient_index],
    training_std=train_feature_std,
    training_min=train_feature_min,
    training_max=train_feature_max,
    n_perturbations=N_PERTURBATIONS,
    noise_level=NOISE_LEVEL,
    random_state=RANDOM_STATE
)

display(example_perturbations)
comparison_example = pd.concat(
    [
        X_test_df.iloc[[patient_index]].assign(
            Version="Original"
        ),
        example_perturbations.assign(
            Version=[
                f"Perturbation_{i + 1}"
                for i in range(len(example_perturbations))
            ]
        )
    ],
    ignore_index=True
)

display(comparison_example)
def calculate_stability_spearman(
    original_explanation,
    perturbed_explanation
):
    """
    Compare the feature-importance rankings of two explanations.
    """

    original_importance = np.abs(
        np.asarray(original_explanation, dtype=float)
    )

    perturbed_importance = np.abs(
        np.asarray(perturbed_explanation, dtype=float)
    )

    correlation, _ = spearmanr(
        original_importance,
        perturbed_importance
    )

    # This can occur when one vector contains identical values
    if np.isnan(correlation):
        correlation = 0.0

    return float(correlation)
def normalise_spearman(correlation):
    """
    Convert Spearman correlation from [-1, 1] to [0, 1].
    """

    return (correlation + 1.0) / 2.0
def calculate_stability_top_k(
    original_explanation,
    perturbed_explanation,
    k=3
):
    """
    Measure overlap between the top-k features in the original and
    perturbed explanations.
    """

    original_importance = np.abs(
        np.asarray(original_explanation, dtype=float)
    )

    perturbed_importance = np.abs(
        np.asarray(perturbed_explanation, dtype=float)
    )

    if not 1 <= k <= len(original_importance):
        raise ValueError(
            f"k must be between 1 and {len(original_importance)}."
        )

    original_top_k = np.argsort(
        original_importance
    )[-k:]

    perturbed_top_k = np.argsort(
        perturbed_importance
    )[-k:]

    shared_features = set(original_top_k).intersection(
        set(perturbed_top_k)
    )

    return len(shared_features) / k
def calculate_explanation_similarity(
    original_explanation,
    perturbed_explanation,
    k=3,
    spearman_weight=0.5,
    top_k_weight=0.5
):
    """
    Calculate the similarity between an original explanation and one
    perturbed explanation.
    """

    if not np.isclose(
        spearman_weight + top_k_weight,
        1.0
    ):
        raise ValueError(
            "Spearman and top-k weights must sum to 1."
        )

    spearman_raw = calculate_stability_spearman(
        original_explanation,
        perturbed_explanation
    )

    spearman_normalised = normalise_spearman(
        spearman_raw
    )

    top_k_overlap = calculate_stability_top_k(
        original_explanation,
        perturbed_explanation,
        k=k
    )

    stability_score = (
        spearman_weight * spearman_normalised
        + top_k_weight * top_k_overlap
    )

    return {
        "spearman_raw": spearman_raw,
        "spearman_normalised": spearman_normalised,
        "top_k_overlap": top_k_overlap,
        "stability_score": stability_score
    }
def get_positive_class_probabilities(
    model,
    X_model_input
):
    """
    Return predicted probabilities for class 1.
    """

    probabilities = model.predict_proba(
        X_model_input
    )[:, 1]

    return np.asarray(probabilities, dtype=float)
def generate_lr_shap_explanations(
    perturbed_raw_df,
    scaler,
    shap_explainer
):
    """
    Scale perturbed clinical values and generate Logistic Regression
    SHAP explanations.
    """

    perturbed_scaled = scaler.transform(
        perturbed_raw_df[feature_names]
    )

    perturbed_scaled_df = pd.DataFrame(
        perturbed_scaled,
        columns=feature_names,
        index=perturbed_raw_df.index
    )

    shap_output = shap_explainer(
        perturbed_scaled_df
    )

    shap_values = get_positive_class_shap_values(
        shap_output
    )

    return shap_values, perturbed_scaled_df
def generate_rf_shap_explanations(
    perturbed_raw_df,
    shap_explainer
):
    """
    Generate Random Forest SHAP explanations.
    """

    shap_output = shap_explainer(
        perturbed_raw_df[feature_names]
    )

    shap_values = get_positive_class_shap_values(
        shap_output
    )

    return shap_values
def generate_xgb_shap_explanations(
    perturbed_raw_df,
    model
):
    """
    Generate XGBoost SHAP values using XGBoost's native
    pred_contribs implementation.
    """

    xgb_matrix = xgb.DMatrix(
        perturbed_raw_df[feature_names],
        feature_names=feature_names
    )

    contributions = model.get_booster().predict(
        xgb_matrix,
        pred_contribs=True
    )

    # Remove the final bias/base-value column
    shap_values = contributions[:, :-1]

    return shap_values
def generate_lime_explanations_for_perturbations(
    explainer,
    model,
    X_model_input,
    n_features,
    label=1,
    num_samples=5000
):
    """
    Generate a LIME attribution vector for every perturbed patient.
    """

    lime_matrix = np.zeros(
        (len(X_model_input), n_features),
        dtype=float
    )

    for row_position in range(len(X_model_input)):

        if isinstance(X_model_input, pd.DataFrame):
            patient_values = (
                X_model_input.iloc[row_position].values
            )
        else:
            patient_values = np.asarray(
                X_model_input[row_position]
            )

        explanation = explainer.explain_instance(
            data_row=patient_values,
            predict_fn=model.predict_proba,
            labels=(label,),
            num_features=n_features,
            num_samples=num_samples
        )

        lime_matrix[row_position] = (
            lime_explanation_to_vector(
                explanation=explanation,
                n_features=n_features,
                label=label
            )
        )

    return lime_matrix
def summarise_patient_stability(
    original_explanation,
    perturbed_explanations,
    k=3
):
    """
    Compare one original explanation with all perturbed explanations
    and return the average stability metrics.
    """

    comparison_results = []

    for perturbation_index in range(
        len(perturbed_explanations)
    ):
        comparison = calculate_explanation_similarity(
            original_explanation=original_explanation,
            perturbed_explanation=(
                perturbed_explanations[
                    perturbation_index
                ]
            ),
            k=k
        )

        comparison_results.append(comparison)

    comparison_df = pd.DataFrame(
        comparison_results
    )

    return {
        "Mean_Stability_Spearman": (
            comparison_df["spearman_raw"].mean()
        ),
        "Std_Stability_Spearman": (
            comparison_df["spearman_raw"].std(ddof=0)
        ),
        "Mean_Stability_TopK": (
            comparison_df["top_k_overlap"].mean()
        ),
        "Std_Stability_TopK": (
            comparison_df["top_k_overlap"].std(ddof=0)
        ),
        "Mean_Stability_Score": (
            comparison_df["stability_score"].mean()
        ),
        "Std_Stability_Score": (
            comparison_df["stability_score"].std(ddof=0)
        ),
        "Minimum_Stability_Score": (
            comparison_df["stability_score"].min()
        ),
        "Maximum_Stability_Score": (
            comparison_df["stability_score"].max()
        )
    }
def calculate_lr_shap_stability(
    X_test_raw,
    original_shap_matrix,
    model,
    scaler,
    shap_explainer,
    n_perturbations=10,
    noise_level=0.05,
    k=3,
    random_state=42
):
    """
    Calculate patient-level SHAP stability for Logistic Regression.
    """

    results = []

    for patient_position in range(len(X_test_raw)):

        original_patient = X_test_raw.iloc[
            patient_position
        ]

        perturbed_raw = generate_perturbed_patients(
            patient=original_patient,
            training_std=train_feature_std,
            training_min=train_feature_min,
            training_max=train_feature_max,
            n_perturbations=n_perturbations,
            noise_level=noise_level,
            random_state=(
                random_state + patient_position
            )
        )

        perturbed_shap, perturbed_scaled = (
            generate_lr_shap_explanations(
                perturbed_raw_df=perturbed_raw,
                scaler=scaler,
                shap_explainer=shap_explainer
            )
        )

        stability = summarise_patient_stability(
            original_explanation=(
                original_shap_matrix[
                    patient_position
                ]
            ),
            perturbed_explanations=perturbed_shap,
            k=k
        )

        original_scaled = (
            X_test_scaled_df.iloc[
                [patient_position]
            ]
        )

        original_probability = (
            model.predict_proba(
                original_scaled
            )[0, 1]
        )

        perturbed_probabilities = (
            model.predict_proba(
                perturbed_scaled
            )[:, 1]
        )

        class_flip_rate = np.mean(
            (perturbed_probabilities >= 0.5)
            != (original_probability >= 0.5)
        )

        results.append({
            "Model": "Logistic Regression",
            "Explainer": "SHAP",
            "Patient_Position": patient_position,
            "Original_Row_Index": (
                X_test_raw.index[
                    patient_position
                ]
            ),
            "Original_Probability": (
                original_probability
            ),
            "Mean_Perturbed_Probability": (
                perturbed_probabilities.mean()
            ),
            "Mean_Absolute_Probability_Change": (
                np.mean(
                    np.abs(
                        perturbed_probabilities
                        - original_probability
                    )
                )
            ),
            "Class_Flip_Rate": class_flip_rate,
            **stability
        })

        if (patient_position + 1) % 20 == 0:
            print(
                "LR SHAP:",
                patient_position + 1,
                "/",
                len(X_test_raw)
            )

    return pd.DataFrame(results)
stability_lr_shap = calculate_lr_shap_stability(
    X_test_raw=X_test_df,
    original_shap_matrix=shap_values_lr,
    model=best_log_reg,
    scaler=scaler,
    shap_explainer=shap_explainer_lr,
    n_perturbations=N_PERTURBATIONS,
    noise_level=NOISE_LEVEL,
    k=TOP_K,
    random_state=RANDOM_STATE
)

stability_lr_shap.head()
def calculate_lr_lime_stability(
    X_test_raw,
    original_lime_matrix,
    model,
    scaler,
    lime_explainer,
    n_perturbations=10,
    noise_level=0.05,
    k=3,
    num_samples=5000,
    random_state=42
):
    """
    Calculate patient-level LIME stability for Logistic Regression.
    """

    results = []

    for patient_position in range(len(X_test_raw)):

        original_patient = X_test_raw.iloc[
            patient_position
        ]

        perturbed_raw = generate_perturbed_patients(
            patient=original_patient,
            training_std=train_feature_std,
            training_min=train_feature_min,
            training_max=train_feature_max,
            n_perturbations=n_perturbations,
            noise_level=noise_level,
            random_state=(
                random_state + patient_position
            )
        )

        perturbed_scaled = scaler.transform(
            perturbed_raw[feature_names]
        )

        perturbed_scaled_df = pd.DataFrame(
            perturbed_scaled,
            columns=feature_names
        )

        perturbed_lime = (
            generate_lime_explanations_for_perturbations(
                explainer=lime_explainer,
                model=model,
                X_model_input=perturbed_scaled_df,
                n_features=len(feature_names),
                label=1,
                num_samples=num_samples
            )
        )

        stability = summarise_patient_stability(
            original_explanation=(
                original_lime_matrix[
                    patient_position
                ]
            ),
            perturbed_explanations=perturbed_lime,
            k=k
        )

        original_probability = (
            model.predict_proba(
                X_test_scaled_df.iloc[
                    [patient_position]
                ]
            )[0, 1]
        )

        perturbed_probabilities = (
            model.predict_proba(
                perturbed_scaled_df
            )[:, 1]
        )

        class_flip_rate = np.mean(
            (perturbed_probabilities >= 0.5)
            != (original_probability >= 0.5)
        )

        results.append({
            "Model": "Logistic Regression",
            "Explainer": "LIME",
            "Patient_Position": patient_position,
            "Original_Row_Index": (
                X_test_raw.index[
                    patient_position
                ]
            ),
            "Original_Probability": (
                original_probability
            ),
            "Mean_Perturbed_Probability": (
                perturbed_probabilities.mean()
            ),
            "Mean_Absolute_Probability_Change": (
                np.mean(
                    np.abs(
                        perturbed_probabilities
                        - original_probability
                    )
                )
            ),
            "Class_Flip_Rate": class_flip_rate,
            **stability
        })

        if (patient_position + 1) % 10 == 0:
            print(
                "LR LIME:",
                patient_position + 1,
                "/",
                len(X_test_raw)
            )

    return pd.DataFrame(results)
stability_lr_lime = calculate_lr_lime_stability(
    X_test_raw=X_test_df,
    original_lime_matrix=lime_values_lr,
    model=best_log_reg,
    scaler=scaler,
    lime_explainer=lime_explainer_lr,
    n_perturbations=N_PERTURBATIONS,
    noise_level=NOISE_LEVEL,
    k=TOP_K,
    num_samples=LIME_NUM_SAMPLES,
    random_state=RANDOM_STATE
)

stability_lr_lime.head()
def calculate_tree_shap_stability(
    X_test_raw,
    original_shap_matrix,
    model,
    model_name,
    shap_generator,
    n_perturbations=10,
    noise_level=0.05,
    k=3,
    random_state=42
):
    """
    Calculate SHAP stability for Random Forest or XGBoost.
    """

    results = []

    for patient_position in range(len(X_test_raw)):

        original_patient = X_test_raw.iloc[
            patient_position
        ]

        perturbed_raw = generate_perturbed_patients(
            patient=original_patient,
            training_std=train_feature_std,
            training_min=train_feature_min,
            training_max=train_feature_max,
            n_perturbations=n_perturbations,
            noise_level=noise_level,
            random_state=(
                random_state + patient_position
            )
        )

        perturbed_shap = shap_generator(
            perturbed_raw
        )

        stability = summarise_patient_stability(
            original_explanation=(
                original_shap_matrix[
                    patient_position
                ]
            ),
            perturbed_explanations=perturbed_shap,
            k=k
        )

        original_probability = (
            model.predict_proba(
                X_test_raw.iloc[
                    [patient_position]
                ]
            )[0, 1]
        )

        perturbed_probabilities = (
            model.predict_proba(
                perturbed_raw
            )[:, 1]
        )

        class_flip_rate = np.mean(
            (perturbed_probabilities >= 0.5)
            != (original_probability >= 0.5)
        )

        results.append({
            "Model": model_name,
            "Explainer": "SHAP",
            "Patient_Position": patient_position,
            "Original_Row_Index": (
                X_test_raw.index[
                    patient_position
                ]
            ),
            "Original_Probability": (
                original_probability
            ),
            "Mean_Perturbed_Probability": (
                perturbed_probabilities.mean()
            ),
            "Mean_Absolute_Probability_Change": (
                np.mean(
                    np.abs(
                        perturbed_probabilities
                        - original_probability
                    )
                )
            ),
            "Class_Flip_Rate": class_flip_rate,
            **stability
        })

        if (patient_position + 1) % 20 == 0:
            print(
                model_name,
                "SHAP:",
                patient_position + 1,
                "/",
                len(X_test_raw)
            )

    return pd.DataFrame(results)
def rf_shap_generator(X_perturbed):
    shap_output = shap_explainer_rf(
        X_perturbed[feature_names]
    )

    return get_positive_class_shap_values(
        shap_output
    )
stability_rf_shap = calculate_tree_shap_stability(
    X_test_raw=X_test_df,
    original_shap_matrix=shap_values_rf,
    model=best_rf,
    model_name="Random Forest",
    shap_generator=rf_shap_generator,
    n_perturbations=N_PERTURBATIONS,
    noise_level=NOISE_LEVEL,
    k=TOP_K,
    random_state=RANDOM_STATE
)
stability_rf_shap.head()
def xgb_shap_generator(X_perturbed):
    return generate_xgb_shap_explanations(
        perturbed_raw_df=X_perturbed,
        model=best_xgb
    )
stability_xgb_shap = calculate_tree_shap_stability(
    X_test_raw=X_test_df,
    original_shap_matrix=shap_values_xgb,
    model=best_xgb,
    model_name="XGBoost",
    shap_generator=xgb_shap_generator,
    n_perturbations=N_PERTURBATIONS,
    noise_level=NOISE_LEVEL,
    k=TOP_K,
    random_state=RANDOM_STATE
)
def calculate_tree_lime_stability(
    X_test_raw,
    original_lime_matrix,
    model,
    model_name,
    lime_explainer,
    n_perturbations=10,
    noise_level=0.05,
    k=3,
    num_samples=5000,
    random_state=42
):
    """
    Calculate LIME stability for Random Forest or XGBoost.
    """

    results = []

    for patient_position in range(len(X_test_raw)):

        original_patient = X_test_raw.iloc[
            patient_position
        ]

        perturbed_raw = generate_perturbed_patients(
            patient=original_patient,
            training_std=train_feature_std,
            training_min=train_feature_min,
            training_max=train_feature_max,
            n_perturbations=n_perturbations,
            noise_level=noise_level,
            random_state=(
                random_state + patient_position
            )
        )

        perturbed_lime = (
            generate_lime_explanations_for_perturbations(
                explainer=lime_explainer,
                model=model,
                X_model_input=perturbed_raw,
                n_features=len(feature_names),
                label=1,
                num_samples=num_samples
            )
        )

        stability = summarise_patient_stability(
            original_explanation=(
                original_lime_matrix[
                    patient_position
                ]
            ),
            perturbed_explanations=perturbed_lime,
            k=k
        )

        original_probability = (
            model.predict_proba(
                X_test_raw.iloc[
                    [patient_position]
                ]
            )[0, 1]
        )

        perturbed_probabilities = (
            model.predict_proba(
                perturbed_raw
            )[:, 1]
        )

        class_flip_rate = np.mean(
            (perturbed_probabilities >= 0.5)
            != (original_probability >= 0.5)
        )

        results.append({
            "Model": model_name,
            "Explainer": "LIME",
            "Patient_Position": patient_position,
            "Original_Row_Index": (
                X_test_raw.index[
                    patient_position
                ]
            ),
            "Original_Probability": (
                original_probability
            ),
            "Mean_Perturbed_Probability": (
                perturbed_probabilities.mean()
            ),
            "Mean_Absolute_Probability_Change": (
                np.mean(
                    np.abs(
                        perturbed_probabilities
                        - original_probability
                    )
                )
            ),
            "Class_Flip_Rate": class_flip_rate,
            **stability
        })

        if (patient_position + 1) % 10 == 0:
            print(
                model_name,
                "LIME:",
                patient_position + 1,
                "/",
                len(X_test_raw)
            )

    return pd.DataFrame(results)
stability_rf_lime = calculate_tree_lime_stability(
    X_test_raw=X_test_df,
    original_lime_matrix=lime_values_rf,
    model=best_rf,
    model_name="Random Forest",
    lime_explainer=lime_explainer_tree,
    n_perturbations=N_PERTURBATIONS,
    noise_level=NOISE_LEVEL,
    k=TOP_K,
    num_samples=LIME_NUM_SAMPLES,
    random_state=RANDOM_STATE
)
stability_xgb_lime = calculate_tree_lime_stability(
    X_test_raw=X_test_df,
    original_lime_matrix=lime_values_xgb,
    model=best_xgb,
    model_name="XGBoost",
    lime_explainer=lime_explainer_tree,
    n_perturbations=N_PERTURBATIONS,
    noise_level=NOISE_LEVEL,
    k=TOP_K,
    num_samples=LIME_NUM_SAMPLES,
    random_state=RANDOM_STATE
)
stability_all = pd.concat(
    [
        stability_lr_shap,
        stability_lr_lime,
        stability_rf_shap,
        stability_rf_lime,
        stability_xgb_shap,
        stability_xgb_lime
    ],
    ignore_index=True
)

print("Combined stability shape:", stability_all.shape)

stability_all.head()
stability_summary = (
    stability_all
    .groupby(["Model", "Explainer"])
    .agg(
        Mean_Spearman=(
            "Mean_Stability_Spearman",
            "mean"
        ),
        Median_Spearman=(
            "Mean_Stability_Spearman",
            "median"
        ),
        Mean_Top3_Overlap=(
            "Mean_Stability_TopK",
            "mean"
        ),
        Median_Top3_Overlap=(
            "Mean_Stability_TopK",
            "median"
        ),
        Mean_Stability=(
            "Mean_Stability_Score",
            "mean"
        ),
        Median_Stability=(
            "Mean_Stability_Score",
            "median"
        ),
        Std_Stability=(
            "Mean_Stability_Score",
            "std"
        ),
        Minimum_Stability=(
            "Mean_Stability_Score",
            "min"
        ),
        Maximum_Stability=(
            "Mean_Stability_Score",
            "max"
        ),
        Mean_Class_Flip_Rate=(
            "Class_Flip_Rate",
            "mean"
        ),
        Mean_Probability_Change=(
            "Mean_Absolute_Probability_Change",
            "mean"
        )
    )
    .round(4)
    .reset_index()
)

display(stability_summary)
print("Agreement table shape:", agreement_all.shape)
print("Stability table shape:", stability_all.shape)

display(agreement_all.head())
display(stability_all.head())
print(
    "Agreement models:",
    agreement_all["Model"].unique()
)

print(
    "Stability models:",
    stability_all["Model"].unique()
)
agreement_duplicates = agreement_all.duplicated(
    subset=["Model", "Patient_Position"],
    keep=False
)

print(
    "Duplicate agreement rows:",
    agreement_duplicates.sum()
)
stability_counts = (
    stability_all
    .groupby(
        [
            "Model",
            "Patient_Position",
            "Explainer"
        ]
    )
    .size()
)

print(
    "Maximum stability rows per combination:",
    stability_counts.max()
)
stability_wide = (
    stability_all
    .pivot_table(
        index=[
            "Model",
            "Patient_Position",
            "Original_Row_Index"
        ],
        columns="Explainer",
        values="Mean_Stability_Score",
        aggfunc="first"
    )
    .reset_index()
)
display(stability_wide.head())
stability_wide = stability_wide.rename(
    columns={
        "SHAP": "SHAP_Stability",
        "LIME": "LIME_Stability"
    }
)
display(stability_wide.head())
required_stability_columns = [
    "SHAP_Stability",
    "LIME_Stability"
]

for column in required_stability_columns:
    if column not in stability_wide.columns:
        raise ValueError(
            f"Missing expected column: {column}"
        )
print(
    stability_wide[
        required_stability_columns
    ].isna().sum()
)
for column in required_stability_columns:

    if not stability_wide[column].between(
        0,
        1
    ).all():

        raise ValueError(
            f"{column} contains values outside [0, 1]."
        )

print("Stability validation passed.")
stability_wide["Average_Stability"] = (
    stability_wide[
        [
            "SHAP_Stability",
            "LIME_Stability"
        ]
    ]
    .mean(axis=1)
)
display(
    stability_wide[
        [
            "Model",
            "Patient_Position",
            "SHAP_Stability",
            "LIME_Stability",
            "Average_Stability"
        ]
    ].head()
)
agreement_for_reliability = agreement_all[
    [
        "Model",
        "Patient_Position",
        "Original_Row_Index",
        "Spearman_Raw",
        "Spearman_Normalised",
        "Top_3_Overlap",
        "Agreement_Score"
    ]
].copy()
print(agreement_all.columns.tolist())
reliability_results = agreement_for_reliability.merge(
    stability_wide[
        [
            "Model",
            "Patient_Position",
            "Original_Row_Index",
            "SHAP_Stability",
            "LIME_Stability",
            "Average_Stability"
        ]
    ],
    on=[
        "Model",
        "Patient_Position",
        "Original_Row_Index"
    ],
    how="inner",
    validate="one_to_one"
)
print(
    "Reliability table shape:",
    reliability_results.shape
)
display(reliability_results.head())
print(
    "Agreement rows:",
    len(agreement_for_reliability)
)

print(
    "Stability rows:",
    len(stability_wide)
)

print(
    "Merged rows:",
    len(reliability_results)
)
AGREEMENT_WEIGHT = 0.50
STABILITY_WEIGHT = 0.50
if not np.isclose(
    AGREEMENT_WEIGHT + STABILITY_WEIGHT,
    1.0
):
    raise ValueError(
        "Reliability weights must sum to 1."
    )
reliability_results["Reliability_Score"] = (
    AGREEMENT_WEIGHT
    * reliability_results["Agreement_Score"]
    +
    STABILITY_WEIGHT
    * reliability_results["Average_Stability"]
)
reliability_results["Reliability_Score"] = (
    reliability_results["Reliability_Score"]
    .clip(0, 1)
)
display(
    reliability_results[
        [
            "Model",
            "Patient_Position",
            "Agreement_Score",
            "SHAP_Stability",
            "LIME_Stability",
            "Average_Stability",
            "Reliability_Score"
        ]
    ].head(10)
)
def categorise_reliability(score):
    if score >= 0.80:
        return "High"
    elif score >= 0.60:
        return "Moderate"
    else:
        return "Low"
reliability_results["Reliability_Level"] = (
    reliability_results["Reliability_Score"]
    .apply(categorise_reliability)
)
reliability_summary = (
    reliability_results
    .groupby("Model")
    .agg(
        Mean_Agreement=(
            "Agreement_Score",
            "mean"
        ),
        Mean_SHAP_Stability=(
            "SHAP_Stability",
            "mean"
        ),
        Mean_LIME_Stability=(
            "LIME_Stability",
            "mean"
        ),
        Mean_Average_Stability=(
            "Average_Stability",
            "mean"
        ),
        Mean_Reliability=(
            "Reliability_Score",
            "mean"
        ),
        Median_Reliability=(
            "Reliability_Score",
            "median"
        ),
        Std_Reliability=(
            "Reliability_Score",
            "std"
        ),
        Minimum_Reliability=(
            "Reliability_Score",
            "min"
        ),
        Maximum_Reliability=(
            "Reliability_Score",
            "max"
        ),
        Number_of_Patients=(
            "Patient_Position",
            "count"
        )
    )
    .round(4)
    .reset_index()
)

display(reliability_summary)
reliability_level_counts = pd.crosstab(
    reliability_results["Model"],
    reliability_results["Reliability_Level"]
)

display(reliability_level_counts)
reliability_level_percentages = pd.crosstab(
    reliability_results["Model"],
    reliability_results["Reliability_Level"],
    normalize="index"
).mul(100).round(2)

display(reliability_level_percentages)
mean_reliability = (
    reliability_results
    .groupby("Model")["Reliability_Score"]
    .mean()
    .sort_values(ascending=False)
)

mean_reliability.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title(
    "Mean Explanation Reliability by Model"
)
plt.xlabel("Model")
plt.ylabel("Mean Reliability Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
reliability_results.boxplot(
    column="Reliability_Score",
    by="Model",
    figsize=(9, 6)
)

plt.title(
    "Distribution of Explanation Reliability by Model"
)
plt.suptitle("")
plt.xlabel("Model")
plt.ylabel("Reliability Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
prediction_results = pd.concat(
    [
        pd.DataFrame({
            "Model": "Logistic Regression",
            "Patient_Position": np.arange(len(y_test)),
            "Original_Row_Index": X_test.index,
            "Actual_Outcome": y_test.values,
            "Predicted_Outcome": y_pred_lr_tuned,
            "Probability_Diabetes": y_prob_lr_tuned
        }),

        pd.DataFrame({
            "Model": "Random Forest",
            "Patient_Position": np.arange(len(y_test)),
            "Original_Row_Index": X_test.index,
            "Actual_Outcome": y_test.values,
            "Predicted_Outcome": y_pred_rf_tuned,
            "Probability_Diabetes": y_prob_rf_tuned
        }),

        pd.DataFrame({
            "Model": "XGBoost",
            "Patient_Position": np.arange(len(y_test)),
            "Original_Row_Index": X_test.index,
            "Actual_Outcome": y_test.values,
            "Predicted_Outcome": y_pred_xgb_tuned,
            "Probability_Diabetes": y_prob_xgb_tuned
        })
    ],
    ignore_index=True
)
prediction_results["Prediction_Correct"] = (
    prediction_results["Actual_Outcome"]
    == prediction_results["Predicted_Outcome"]
)
final_analysis = reliability_results.merge(
    prediction_results,
    on=[
        "Model",
        "Patient_Position",
        "Original_Row_Index"
    ],
    how="inner",
    validate="one_to_one"
)

print(final_analysis.shape)
display(final_analysis.head())
correctness_summary = (
    final_analysis
    .groupby(
        [
            "Model",
            "Prediction_Correct"
        ]
    )
    .agg(
        Mean_Reliability=(
            "Reliability_Score",
            "mean"
        ),
        Median_Reliability=(
            "Reliability_Score",
            "median"
        ),
        Std_Reliability=(
            "Reliability_Score",
            "std"
        ),
        Number_of_Predictions=(
            "Patient_Position",
            "count"
        )
    )
    .round(4)
    .reset_index()
)

display(correctness_summary)
final_analysis.boxplot(
    column="Reliability_Score",
    by=["Model", "Prediction_Correct"],
    figsize=(12, 6),
    rot=25
)

plt.title(
    "Explanation Reliability for Correct and Incorrect Predictions"
)
plt.suptitle("")
plt.xlabel("Model and prediction correctness")
plt.ylabel("Reliability Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
final_model_summary = reliability_summary.merge(
    pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],
        "Accuracy": [
            accuracy_score(
                y_test,
                y_pred_lr_tuned
            ),
            accuracy_score(
                y_test,
                y_pred_rf_tuned
            ),
            accuracy_score(
                y_test,
                y_pred_xgb_tuned
            )
        ],
        "F1_Score": [
            f1_score(
                y_test,
                y_pred_lr_tuned
            ),
            f1_score(
                y_test,
                y_pred_rf_tuned
            ),
            f1_score(
                y_test,
                y_pred_xgb_tuned
            )
        ],
        "ROC_AUC": [
            roc_auc_score(
                y_test,
                y_prob_lr_tuned
            ),
            roc_auc_score(
                y_test,
                y_prob_rf_tuned
            ),
            roc_auc_score(
                y_test,
                y_prob_xgb_tuned
            )
        ]
    }),
    on="Model",
    how="left"
)

display(final_model_summary)