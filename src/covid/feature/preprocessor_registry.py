from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from umap import UMAP

IMPUTER_REGISTRY = {"simple": SimpleImputer}
SCALER_REGISTRY = {"standard": StandardScaler, "minmax": MinMaxScaler}
REDUCER_REGISTRY = {"pca": PCA, "umap": UMAP}
