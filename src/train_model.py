"""
Módulo de Entrenamiento y Evaluación de Modelos (Fases 3 y 4 de CRISP-ML)
Compara algoritmos de regresión, calcula métricas y serializa el mejor modelo.
"""

import json
import os
from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from database.db_manager import ACIDDatabaseManager
from src.data_loader import ensure_data_seeded

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def build_preprocessor() -> ColumnTransformer:
    """Crea el transformador de preprocesamiento de características."""
    numeric_features = [
        "area_m2",
        "habitaciones",
        "banos",
        "parqueaderos",
        "antiguedad_anos",
        "estrato",
        "tiene_piscina",
        "tiene_ascensor",
    ]
    categorical_features = ["zona"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )
    return preprocessor


def train_and_evaluate_all_models() -> Dict[str, Any]:
    """
    Ejecuta el ciclo CRISP-ML de modelado y evaluación:
    1. Carga datos desde la BD ACID
    2. Divide en Train (80%) y Test (20%)
    3. Entrena Random Forest, Gradient Boosting y Ridge
    4. Evalúa con R2, RMSE y MAE
    5. Guarda modelos y métricas en BD ACID
    """
    db = ACIDDatabaseManager()
    df = ensure_data_seeded(db)

    feature_cols = [
        "area_m2",
        "habitaciones",
        "banos",
        "parqueaderos",
        "antiguedad_anos",
        "estrato",
        "zona",
        "tiene_piscina",
        "tiene_ascensor",
    ]
    target_col = "precio_millones"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models_to_train = {
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42),
        "Ridge Regression": Ridge(alpha=1.0),
    }

    results = {}
    best_r2 = -float("inf")
    best_model_name = None
    best_pipeline = None

    for name, regressor in models_to_train.items():
        preprocessor = build_preprocessor()
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])

        # Entrenamiento
        pipeline.fit(X_train, y_train)

        # Predicciones
        y_pred = pipeline.predict(X_test)

        # Métricas CRISP-ML
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

        results[name] = {
            "r2_score": round(r2, 4),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "mape": round(mape, 2),
            "pipeline": pipeline,
            "y_test": y_test.tolist()[:80],
            "y_pred": y_pred.tolist()[:80],
        }

        # Registrar en la base de datos ACID
        db.register_model_metrics({
            "model_name": name,
            "model_version": "v1.0",
            "r2_score": round(r2, 4),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        })

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline

    # Guardar el mejor modelo en disco
    best_model_path = os.path.join(MODELS_DIR, "best_housing_model.joblib")
    joblib.dump(best_pipeline, best_model_path)

    # Guardar metadata para el dashboard
    metadata = {
        "best_model_name": best_model_name,
        "best_r2": round(best_r2, 4),
        "best_rmse": results[best_model_name]["rmse"],
        "best_mae": results[best_model_name]["mae"],
        "features": feature_cols,
        "models_summary": {
            k: {
                "r2_score": v["r2_score"],
                "rmse": v["rmse"],
                "mae": v["mae"],
                "mape": v["mape"],
            }
            for k, v in results.items()
        },
    }

    with open(os.path.join(MODELS_DIR, "models_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "best_model_name": best_model_name,
        "best_model_path": best_model_path,
        "results": results,
        "metadata": metadata,
    }


if __name__ == "__main__":
    print("Iniciando fase de Entrenamiento y Evaluación CRISP-ML...")
    evaluation_output = train_and_evaluate_all_models()
    print("Mejor Modelo Seleccionado:", evaluation_output["best_model_name"])
    print("Métricas del Modelo:")
    for m_name, m_stats in evaluation_output["metadata"]["models_summary"].items():
        print(f" - {m_name}: R2={m_stats['r2_score']}, RMSE={m_stats['rmse']}M, MAE={m_stats['mae']}M")
