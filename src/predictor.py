"""
Módulo de Inferencia y Predicción (Fase 5 de CRISP-ML: Deployment)
Ejecuta la predicción del modelo y almacena el resultado de forma atómica (ACID).
"""

import os
from typing import Any, Dict, Tuple
import joblib
import pandas as pd
from database.db_manager import ACIDDatabaseManager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_housing_model.joblib")


class HousePricePredictor:
    """Clase encargada de ejecutar predicciones y registrarlas en la BD ACID."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.pipeline = self._load_model()
        self.db = ACIDDatabaseManager()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            from src.train_model import train_and_evaluate_all_models
            train_and_evaluate_all_models()
        return joblib.load(self.model_path)

    def predict(
        self,
        features: Dict[str, Any],
        client_name: str = "Cliente Web",
        save_to_db: bool = True,
    ) -> Dict[str, Any]:
        """
        Ejecuta la inferencia sobre las características ingresadas.
        Calcula precio estimado, rango de confianza y desglose.
        Si save_to_db=True, persiste la cotización en la BD con propiedades ACID.
        """
        df_input = pd.DataFrame([{
            "area_m2": float(features["area_m2"]),
            "habitaciones": int(features["habitaciones"]),
            "banos": int(features["banos"]),
            "parqueaderos": int(features["parqueaderos"]),
            "antiguedad_anos": int(features["antiguedad_anos"]),
            "estrato": int(features["estrato"]),
            "zona": str(features["zona"]),
            "tiene_piscina": int(features["tiene_piscina"]),
            "tiene_ascensor": int(features["tiene_ascensor"]),
        }])

        # Inferencia
        predicted_val = float(self.pipeline.predict(df_input)[0])
        predicted_val = max(30.0, round(predicted_val, 2))

        # Rango de mercado estimado (+/- 6% de tolerancia estocástica)
        rango_min = round(predicted_val * 0.94, 2)
        rango_max = round(predicted_val * 1.06, 2)
        precio_m2 = round(predicted_val / float(features["area_m2"]), 2)

        transaction_uuid = None
        db_message = "No guardado en BD"

        if save_to_db:
            payload = {
                "cliente_nombre": client_name,
                "area_m2": features["area_m2"],
                "habitaciones": features["habitaciones"],
                "banos": features["banos"],
                "parqueaderos": features["parqueaderos"],
                "antiguedad_anos": features["antiguedad_anos"],
                "estrato": features["estrato"],
                "zona": features["zona"],
                "tiene_piscina": features["tiene_piscina"],
                "tiene_ascensor": features["tiene_ascensor"],
                "precio_estimado_millones": predicted_val,
                "precio_min_millones": rango_min,
                "precio_max_millones": rango_max,
                "modelo_utilizado": "RandomForest_Production_v1",
            }
            success, db_message, transaction_uuid = self.db.record_prediction_atomic(payload)

        return {
            "precio_estimado_millones": predicted_val,
            "precio_min_millones": rango_min,
            "precio_max_millones": rango_max,
            "precio_m2_millones": precio_m2,
            "transaction_uuid": transaction_uuid,
            "db_status": db_message,
        }
