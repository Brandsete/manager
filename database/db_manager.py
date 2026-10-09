"""
Módulo de Gestión de Base de Datos Transaccional (ACID Compliance)
Garantiza Atomicidad, Consistencia, Aislamiento y Durabilidad en todas las operaciones.
"""

import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "real_estate_acid.db")
SCHEMA_PATH = os.path.join(DB_DIR, "schema.sql")


class ACIDDatabaseManager:
    """
    Gestor de base de datos relacional con garantías ACID completas.
    - Atomicidad: Operaciones en bloques de transacción (BEGIN - COMMIT - ROLLBACK).
    - Consistencia: Validación estricta de esquemas, CHECK constraints y tipos.
    - Aislamiento: SQLite en modo WAL y transacciones serializadas/inmediatas.
    - Durabilidad: Sincronización a disco con journal WAL persistente.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._initialize_database()

    def get_connection(self) -> sqlite3.Connection:
        """Obtiene una conexión configurada para cumplir ACID."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Activar Foreign Keys y modo WAL para aislamiento y durabilidad
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _initialize_database(self) -> None:
        """Aplica el esquema DDL si las tablas no existen."""
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        conn = self.get_connection()
        try:
            conn.executescript(schema_sql)
            conn.commit()
        finally:
            conn.close()

    def insert_property_atomic(self, property_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Inserta una propiedad cumpliendo Atomicidad y Consistencia.
        Si alguna restricción falla (CHECK constraints), hace rollback automático.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE TRANSACTION;")

            cursor.execute(
                """
                INSERT INTO properties (
                    codigo_inmueble, area_m2, habitaciones, banos, parqueaderos,
                    antiguedad_anos, estrato, zona, tiene_piscina, tiene_ascensor, precio_millones
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    property_data["codigo_inmueble"],
                    property_data["area_m2"],
                    property_data["habitaciones"],
                    property_data["banos"],
                    property_data.get("parqueaderos", 0),
                    property_data.get("antiguedad_anos", 0),
                    property_data["estrato"],
                    property_data["zona"],
                    int(property_data.get("tiene_piscina", 0)),
                    int(property_data.get("tiene_ascensor", 0)),
                    property_data["precio_millones"],
                ),
            )
            inserted_id = cursor.lastrowid

            # Registro de auditoría dentro de la misma transacción (Atomicidad)
            cursor.execute(
                """
                INSERT INTO acid_audit_log (operation_type, table_affected, record_id, details, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "INSERT_PROPERTY",
                    "properties",
                    inserted_id,
                    f"Código: {property_data['codigo_inmueble']} - Zona: {property_data['zona']} - Precio: {property_data['precio_millones']}M",
                    "SUCCESS",
                ),
            )

            conn.commit()
            return True, f"Propiedad registrada exitosamente (ID: {inserted_id})"
        except sqlite3.IntegrityError as ie:
            conn.rollback()
            return False, f"Violación de Consistencia ACID (IntegrityError): {str(ie)}"
        except Exception as e:
            conn.rollback()
            return False, f"Error transaccional (Rollback ejecutado): {str(e)}"
        finally:
            conn.close()

    def record_prediction_atomic(self, prediction_payload: Dict[str, Any]) -> Tuple[bool, str, str]:
        """
        Registra una predicción generada por el modelo ML con auditoría transaccional.
        Retorna (exito, mensaje, transaction_uuid).
        """
        trans_uuid = str(uuid.uuid4())
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE TRANSACTION;")

            cursor.execute(
                """
                INSERT INTO prediction_logs (
                    transaction_uuid, cliente_nombre, area_m2, habitaciones, banos,
                    parqueaderos, antiguedad_anos, estrato, zona, tiene_piscina,
                    tiene_ascensor, precio_estimado_millones, precio_min_millones,
                    precio_max_millones, modelo_utilizado, estado_transaccion
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trans_uuid,
                    prediction_payload.get("cliente_nombre", "Anónimo"),
                    prediction_payload["area_m2"],
                    prediction_payload["habitaciones"],
                    prediction_payload["banos"],
                    prediction_payload.get("parqueaderos", 0),
                    prediction_payload.get("antiguedad_anos", 0),
                    prediction_payload["estrato"],
                    prediction_payload["zona"],
                    int(prediction_payload.get("tiene_piscina", 0)),
                    int(prediction_payload.get("tiene_ascensor", 0)),
                    prediction_payload["precio_estimado_millones"],
                    prediction_payload["precio_min_millones"],
                    prediction_payload["precio_max_millones"],
                    prediction_payload.get("modelo_utilizado", "RandomForest_v1"),
                    "COMMITTED",
                ),
            )
            pred_id = cursor.lastrowid

            cursor.execute(
                """
                INSERT INTO acid_audit_log (operation_type, table_affected, record_id, details, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "LOG_PREDICTION",
                    "prediction_logs",
                    pred_id,
                    f"Transacción {trans_uuid} comprometida. Estimado: ${prediction_payload['precio_estimado_millones']:,.2f}M",
                    "COMMITTED",
                ),
            )

            conn.commit()
            return True, f"Predicción guardada en BD ACID (UUID: {trans_uuid[:8]}...)", trans_uuid
        except Exception as e:
            conn.rollback()
            return False, f"Fallo al registrar predicción (Rollback aplicado): {str(e)}", ""
        finally:
            conn.close()

    def get_all_properties(self) -> pd.DataFrame:
        """Carga todas las propiedades de entrenamiento como DataFrame."""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query("SELECT * FROM properties ORDER BY id ASC", conn)
            return df
        finally:
            conn.close()

    def get_prediction_history(self, limit: int = 50) -> pd.DataFrame:
        """Obtiene el historial reciente de cotizaciones/predicciones."""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM prediction_logs ORDER BY fecha_prediccion DESC LIMIT ?",
                conn,
                params=(limit,),
            )
            return df
        finally:
            conn.close()

    def get_audit_logs(self, limit: int = 30) -> pd.DataFrame:
        """Obtiene la traza de auditoría transaccional ACID."""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM acid_audit_log ORDER BY created_at DESC LIMIT ?",
                conn,
                params=(limit,),
            )
            return df
        finally:
            conn.close()

    def register_model_metrics(self, metrics: Dict[str, Any]) -> None:
        """Registra una versión de modelo en el model registry."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE TRANSACTION;")
            cursor.execute("UPDATE model_registry SET is_active = 0 WHERE is_active = 1")
            cursor.execute(
                """
                INSERT INTO model_registry (
                    model_name, model_version, r2_score, rmse, mae, train_samples, test_samples, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    metrics["model_name"],
                    metrics["model_version"],
                    metrics["r2_score"],
                    metrics["rmse"],
                    metrics["mae"],
                    metrics["train_samples"],
                    metrics["test_samples"],
                ),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_model_registry(self) -> pd.DataFrame:
        """Obtiene el registro histórico de modelos evaluados."""
        conn = self.get_connection()
        try:
            return pd.read_sql_query("SELECT * FROM model_registry ORDER BY created_at DESC", conn)
        finally:
            conn.close()
