from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import StudentFeatureEngineer, load_datasets


def build_preprocessor(num_cols, cat_cols):
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def evaluate_classification(pipeline, X_test, y_test):
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, pos_label="Placed"),
        "recall": recall_score(y_test, pred, pos_label="Placed"),
        "f1": f1_score(y_test, pred, pos_label="Placed"),
        "roc_auc": roc_auc_score((y_test == "Placed").astype(int), proba),
    }


def evaluate_regression(pipeline, X_test, y_test):
    pred = pipeline.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, pred),
        "rmse": mean_squared_error(y_test, pred) ** 0.5,
        "r2": r2_score(y_test, pred),
    }


def main():
    parser = argparse.ArgumentParser(description="Train student placement models with MLflow logging.")
    parser.add_argument("--features", type=str, default="A.csv")
    parser.add_argument("--targets", type=str, default="A_targets.csv")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--experiment", type=str, default="student_placement_project")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(args.experiment)

    # Load data and feature engineering
    data = load_datasets(args.features, args.targets)
    data = StudentFeatureEngineer().fit_transform(data)

    # -----------------------
    # Classification task
    # -----------------------
    X_cls = data.drop(columns=["Student_ID", "placement_status", "salary_lpa"])
    y_cls = data["placement_status"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls,
        y_cls,
        test_size=0.2,
        random_state=args.random_state,
        stratify=y_cls,
    )

    num_cols_cls = X_cls.select_dtypes(include="number").columns.tolist()
    cat_cols_cls = X_cls.select_dtypes(exclude="number").columns.tolist()

    preprocessor_cls = build_preprocessor(num_cols_cls, cat_cols_cls)

    cls_candidates = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "rf": RandomForestClassifier(
            n_estimators=300,
            random_state=args.random_state,
            class_weight="balanced_subsample",
        ),
        "gb": GradientBoostingClassifier(random_state=args.random_state),
    }

    best_cls_name = None
    best_cls_score = -1.0
    best_cls_pipeline = None
    cls_summary = {}

    for name, model in cls_candidates.items():
        pipeline = Pipeline(
            steps=[
                ("feature_engineer", StudentFeatureEngineer()),
                ("preprocess", preprocessor_cls),
                ("model", model),
            ]
        )
        pipeline.fit(X_train_c, y_train_c)
        metrics = evaluate_classification(pipeline, X_test_c, y_test_c)
        cls_summary[name] = metrics

        with mlflow.start_run(run_name=f"classification_{name}"):
            mlflow.log_param("task", "classification")
            mlflow.log_param("model_name", name)
            mlflow.log_param("random_state", args.random_state)
            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall", metrics["recall"])
            mlflow.log_metric("f1", metrics["f1"])
            mlflow.log_metric("roc_auc", metrics["roc_auc"])
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

        if metrics["f1"] > best_cls_score:
            best_cls_score = metrics["f1"]
            best_cls_name = name
            best_cls_pipeline = pipeline

    # Refit best classification model on all classification data
    if best_cls_name is None or best_cls_pipeline is None:
        raise RuntimeError("No classification model was selected.")

    best_cls_pipeline.fit(X_cls, y_cls)
    cls_path = output_dir / "best_placement_model.pkl"
    with open(cls_path, "wb") as f:
        pickle.dump(best_cls_pipeline, f)

    # -----------------------
    # Regression task
    # -----------------------
    reg_data = data.loc[data["placement_status"] == "Placed"].copy()
    X_reg = reg_data.drop(columns=["Student_ID", "placement_status", "salary_lpa"])
    y_reg = reg_data["salary_lpa"]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg,
        y_reg,
        test_size=0.2,
        random_state=args.random_state,
    )

    num_cols_reg = X_reg.select_dtypes(include="number").columns.tolist()
    cat_cols_reg = X_reg.select_dtypes(exclude="number").columns.tolist()

    preprocessor_reg = build_preprocessor(num_cols_reg, cat_cols_reg)

    reg_candidates = {
        "linreg": LinearRegression(),
        "rf": RandomForestRegressor(n_estimators=300, random_state=args.random_state),
        "gbr": GradientBoostingRegressor(random_state=args.random_state),
    }

    best_reg_name = None
    best_reg_score = -1.0
    best_reg_pipeline = None
    reg_summary = {}

    for name, model in reg_candidates.items():
        pipeline = Pipeline(
            steps=[
                ("feature_engineer", StudentFeatureEngineer()),
                ("preprocess", preprocessor_reg),
                ("model", model),
            ]
        )
        pipeline.fit(X_train_r, y_train_r)
        metrics = evaluate_regression(pipeline, X_test_r, y_test_r)
        reg_summary[name] = metrics

        with mlflow.start_run(run_name=f"regression_{name}"):
            mlflow.log_param("task", "regression")
            mlflow.log_param("model_name", name)
            mlflow.log_param("random_state", args.random_state)
            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("r2", metrics["r2"])
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

        if metrics["r2"] > best_reg_score:
            best_reg_score = metrics["r2"]
            best_reg_name = name
            best_reg_pipeline = pipeline

    if best_reg_name is None or best_reg_pipeline is None:
        raise RuntimeError("No regression model was selected.")

    # Refit best regression model on all placed data
    best_reg_pipeline.fit(X_reg, y_reg)
    reg_path = output_dir / "best_salary_model.pkl"
    with open(reg_path, "wb") as f:
        pickle.dump(best_reg_pipeline, f)

    # Save metrics summary
    summary = {
        "best_classification_model": best_cls_name,
        "best_regression_model": best_reg_name,
        "classification_metrics": cls_summary,
        "regression_metrics": reg_summary,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Best classification model: {best_cls_name}")
    print(f"Best regression model: {best_reg_name}")
    print(f"Saved: {cls_path}")
    print(f"Saved: {reg_path}")
    print(f"Saved metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
