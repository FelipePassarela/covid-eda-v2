from typing import Mapping

from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from umap import UMAP

PreprocessorRegistry = Mapping[str, Mapping[str, type[TransformerMixin]]]

_IMPUTER_REGISTRY = {"simple": SimpleImputer}
_SELECTOR_REGISTRY = {"rfe": RFE}
_SCALER_REGISTRY = {"standard": StandardScaler, "minmax": MinMaxScaler}
_REDUCER_REGISTRY = {"pca": PCA, "umap": UMAP}

PREPROCESSOR_REGISTRY: PreprocessorRegistry = {
    "imputer": _IMPUTER_REGISTRY,
    "selector": _SELECTOR_REGISTRY,
    "scaler": _SCALER_REGISTRY,
    "reducer": _REDUCER_REGISTRY,
}
