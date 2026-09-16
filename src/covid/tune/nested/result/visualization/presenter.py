from pandas import DataFrame

from covid.tune.nested.result.visualization.data import BoxplotData

_METRIC_PRESENTER = {
    "test_balanced_accuracy": "Balanced Accuracy",
    "test_roc_auc": "ROC AUC",
    "test_average_precision": "Average Precision",
    "test_f1": "F1 Score",
    "test_precision": "Precision",
    "test_recall": "Recall",
    "train_balanced_accuracy": "Balanced Accuracy (Train)",
    "train_roc_auc": "ROC AUC (Train)",
    "train_average_precision": "Average Precision (Train)",
    "train_f1": "F1 Score (Train)",
    "train_precision": "Precision (Train)",
    "train_recall": "Recall (Train)",
}

_MODEL_PRESENTER = {
    "dummy": "Dummy",
    "logistic-regression": "Logistic Regression",
    "svm": "SVM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "light-gbm": "LightGBM",
}


def present_model_name(model_name: str) -> str:
    # fallback to original name if not found
    return _MODEL_PRESENTER.get(model_name, model_name)


def present_metric(metric: str) -> str:
    return _METRIC_PRESENTER.get(metric, metric)


def format_boxplot_dataframe(boxplot_data: BoxplotData) -> DataFrame:
    formatted_data = boxplot_data.to_long_dataframe()
    formatted_data["model"] = formatted_data["model"].map(present_model_name)
    return formatted_data
