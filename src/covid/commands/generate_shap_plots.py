from pathlib import Path

import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split

from covid.cross_val import TrainingConfig
from covid.data import CovidDatasetLoader, DataCleaningConfig
from covid.model import MODEL_ALIASES, MODEL_REGISTRY, ModelFactory


def generate_shap_plots(
    dataset_loader: CovidDatasetLoader,
    training_config: TrainingConfig,
    cleaning_config: DataCleaningConfig,
    output_path: Path,
) -> None:
    dataset = dataset_loader.load_dataset(cleaning_config)
    X, y = dataset.split()
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=training_config.test_size,
        random_state=training_config.random_state,
        stratify=y,
    )

    model = ModelFactory(MODEL_REGISTRY, MODEL_ALIASES).create(
        "xgb", random_state=training_config.random_state
    )
    model.fit(X_train, y_train)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)

    output_path.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, X_train, show=False, feature_names=X_train.columns)

    # TODO: Create a abstraction for saving plots
    plt.savefig(output_path / "shap_summary_plot.png")
    plt.close()
