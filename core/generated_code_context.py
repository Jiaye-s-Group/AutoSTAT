"""Shared helpers for generated-code execution contexts."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd


ExecutionKind = Literal["preprocessing", "visualization", "modeling", "inference"]

_OUTPUT_NAMES: dict[str, str] = {
    "preprocessing": "process_df",
    "visualization": "fig_dict",
    "modeling": "result_dict",
    "inference": "result_dict",
}


def compatible_one_hot_encoder(*args: Any, **kwargs: Any):
    """Support ``sparse`` and ``sparse_output`` across scikit-learn versions."""
    from sklearn.preprocessing import OneHotEncoder

    params = dict(kwargs)
    if "sparse" in params and "sparse_output" not in params:
        params["sparse_output"] = params.pop("sparse")
    try:
        return OneHotEncoder(*args, **params)
    except TypeError:
        if "sparse_output" not in params:
            raise
        params["sparse"] = params.pop("sparse_output")
        return OneHotEncoder(*args, **params)


def execution_output_name(kind: str) -> str:
    try:
        return _OUTPUT_NAMES[str(kind)]
    except KeyError as exc:
        raise ValueError(f"Unsupported execution kind: {kind}") from exc


def code_requires_torch(code: str) -> bool:
    text = str(code or "").lower()
    hints = (
        "import torch",
        "from torch",
        "torch.",
        "torchvision",
        "torch.nn",
        "torch.utils.data",
        "nn.",
        "optim.",
        "tensor(",
        ".backward(",
        "dataloader",
    )
    return any(hint in text for hint in hints)


def build_execution_namespace(
    *,
    kind: str,
    dataframe: pd.DataFrame,
    code: str = "",
    extra_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the public namespace supplied to generated code for one stage."""
    import numpy as np
    import pandas as runtime_pd

    extra_values = dict(extra_values or {})

    if kind == "preprocessing":
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import (
            FunctionTransformer,
            LabelEncoder,
            MinMaxScaler,
            OrdinalEncoder,
            RobustScaler,
            StandardScaler,
        )

        return {
            "df": dataframe,
            "np": np,
            "pd": runtime_pd,
            "SimpleImputer": SimpleImputer,
            "FunctionTransformer": FunctionTransformer,
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "RobustScaler": RobustScaler,
            "OneHotEncoder": compatible_one_hot_encoder,
            "OrdinalEncoder": OrdinalEncoder,
            "LabelEncoder": LabelEncoder,
            "ColumnTransformer": ColumnTransformer,
            "Pipeline": Pipeline,
        }

    if kind == "visualization":
        import plotly.express as px
        import plotly.graph_objects as go

        return {"df": dataframe, "np": np, "pd": runtime_pd, "px": px, "go": go}

    if kind == "modeling":
        import lightgbm
        import xgboost
        from sklearn.ensemble import (
            AdaBoostClassifier,
            AdaBoostRegressor,
            ExtraTreesClassifier,
            ExtraTreesRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.preprocessing import (
            LabelEncoder,
            MinMaxScaler,
            OneHotEncoder,
            OrdinalEncoder,
            RobustScaler,
            StandardScaler,
        )
        from sklearn.svm import SVC, SVR
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        namespace: dict[str, Any] = {
            "df": dataframe,
            "np": np,
            "pd": runtime_pd,
            "train_test_split": train_test_split,
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "RobustScaler": RobustScaler,
            "LabelEncoder": LabelEncoder,
            "OrdinalEncoder": OrdinalEncoder,
            "OneHotEncoder": OneHotEncoder,
            "LinearRegression": LinearRegression,
            "LogisticRegression": LogisticRegression,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "ElasticNet": ElasticNet,
            "RandomForestRegressor": RandomForestRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "RandomForestClassifier": RandomForestClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "ExtraTreesClassifier": ExtraTreesClassifier,
            "ExtraTreesRegressor": ExtraTreesRegressor,
            "AdaBoostClassifier": AdaBoostClassifier,
            "AdaBoostRegressor": AdaBoostRegressor,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "DecisionTreeRegressor": DecisionTreeRegressor,
            "SVC": SVC,
            "SVR": SVR,
            "KNeighborsClassifier": KNeighborsClassifier,
            "KNeighborsRegressor": KNeighborsRegressor,
            "GaussianNB": GaussianNB,
            "xgboost": xgboost,
            "lightgbm": lightgbm,
        }
        if code_requires_torch(code):
            import torch

            namespace["torch"] = torch
            try:
                import torchvision
            except ModuleNotFoundError:
                pass
            else:
                namespace["torchvision"] = torchvision
        return namespace

    if kind == "inference":
        from sklearn.preprocessing import StandardScaler

        return {
            "inference_df": dataframe,
            "model_obj": extra_values.get("model_obj"),
            "np": np,
            "pd": runtime_pd,
            "StandardScaler": StandardScaler,
        }

    raise ValueError(f"Unsupported execution kind: {kind}")
