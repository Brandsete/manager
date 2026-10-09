"""
Módulo de Carga, Generación y Preparación de Datos (Fase 1 y 2 de CRISP-ML)
Se conecta a la base de datos ACID para cargar y persistir datos inmobiliarios.
"""

import os
import random
import numpy as np
import pandas as pd
from database.db_manager import ACIDDatabaseManager


def generate_synthetic_housing_data(n_samples: int = 1200, random_seed: int = 42) -> pd.DataFrame:
    """
    Genera un conjunto de datos realista de propiedades inmobiliarias
    siguiendo patrones del mercado urbano (Colombia / Latinoamérica).
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    zonas = ["Norte", "Sur", "Oriente", "Occidente", "Centro", "Campestre"]
    # Coeficientes base de valor por m2 según zona (en millones de pesos)
    precio_m2_base_zona = {
        "Norte": 4.8,
        "Campestre": 5.2,
        "Occidente": 3.9,
        "Sur": 3.4,
        "Centro": 3.0,
        "Oriente": 2.5,
    }

    # Distribución de estratos por zona
    zona_estrato_dist = {
        "Norte": [4, 5, 6],
        "Campestre": [5, 6],
        "Occidente": [3, 4, 5],
        "Sur": [2, 3, 4],
        "Centro": [3, 4],
        "Oriente": [2, 3],
    }

    records = []
    for i in range(1, n_samples + 1):
        codigo = f"INM-{i:05d}"
        zona = random.choice(zonas)
        estrato = random.choice(zona_estrato_dist[zona])

        # Área según estrato y zona
        if zona == "Campestre":
            area_m2 = round(np.random.uniform(140.0, 480.0), 1)
            habitaciones = random.randint(3, 6)
            banos = random.randint(3, 6)
            parqueaderos = random.randint(2, 4)
            tiene_piscina = 1 if random.random() > 0.35 else 0
            tiene_ascensor = 0  # Casas campestres suelen no tener ascensor
        elif estrato >= 5:
            area_m2 = round(np.random.uniform(75.0, 260.0), 1)
            habitaciones = random.randint(2, 5)
            banos = random.randint(2, 5)
            parqueaderos = random.randint(1, 3)
            tiene_piscina = 1 if random.random() > 0.45 else 0
            tiene_ascensor = 1 if random.random() > 0.25 else 0
        else:
            area_m2 = round(np.random.uniform(42.0, 130.0), 1)
            habitaciones = random.randint(1, 4)
            banos = random.randint(1, 3)
            parqueaderos = random.randint(0, 1)
            tiene_piscina = 1 if random.random() > 0.8 else 0
            tiene_ascensor = 1 if random.random() > 0.6 else 0

        antiguedad_anos = random.randint(0, 35)

        # Cálculo de precio realista con lógica de negocio y ruido estocástico
        precio_m2 = precio_m2_base_zona[zona] * (1.0 + (estrato - 3) * 0.12)
        precio_estimado = area_m2 * precio_m2

        # Incrementos por características
        precio_estimado += banos * 12.0
        precio_estimado += parqueaderos * 28.0
        precio_estimado += tiene_piscina * 45.0
        precio_estimado += tiene_ascensor * 20.0

        # Depreciación por antigüedad (máx 25%)
        factor_depreciacion = max(0.75, 1.0 - (antiguedad_anos * 0.008))
        precio_estimado *= factor_depreciacion

        # Ruido de mercado (+/- 8%)
        ruido = np.random.normal(1.0, 0.06)
        precio_final = round(float(precio_estimado * ruido), 2)
        precio_final = max(35.0, precio_final)  # Mínimo 35 millones

        records.append({
            "codigo_inmueble": codigo,
            "area_m2": area_m2,
            "habitaciones": habitaciones,
            "banos": banos,
            "parqueaderos": parqueaderos,
            "antiguedad_anos": antiguedad_anos,
            "estrato": estrato,
            "zona": zona,
            "tiene_piscina": tiene_piscina,
            "tiene_ascensor": tiene_ascensor,
            "precio_millones": precio_final,
        })

    return pd.DataFrame(records)


def ensure_data_seeded(db_manager: ACIDDatabaseManager, min_records: int = 800) -> pd.DataFrame:
    """
    Verifica si la base de datos ACID ya tiene datos de inmuebles.
    Si está vacía, genera e inserta registros de forma atómica garantizando ACID.
    """
    df_existing = db_manager.get_all_properties()
    if len(df_existing) >= min_records:
        return df_existing

    print(f"Base de datos con {len(df_existing)} registros. Sembrando datos con garantías ACID...")
    synthetic_df = generate_synthetic_housing_data(n_samples=1200)

    # Inserción con transacción masiva para maximizar rendimiento y preservar Atomicidad
    conn = db_manager.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE TRANSACTION;")
        for _, row in synthetic_df.iterrows():
            cursor.execute(
                """
                INSERT OR IGNORE INTO properties (
                    codigo_inmueble, area_m2, habitaciones, banos, parqueaderos,
                    antiguedad_anos, estrato, zona, tiene_piscina, tiene_ascensor, precio_millones
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["codigo_inmueble"],
                    row["area_m2"],
                    row["habitaciones"],
                    row["banos"],
                    row["parqueaderos"],
                    row["antiguedad_anos"],
                    row["estrato"],
                    row["zona"],
                    int(row["tiene_piscina"]),
                    int(row["tiene_ascensor"]),
                    row["precio_millones"],
                ),
            )
        cursor.execute(
            """
            INSERT INTO acid_audit_log (operation_type, table_affected, record_id, details, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("SEED_DATASET", "properties", None, f"Carga inicial de {len(synthetic_df)} propiedades.", "COMMITTED"),
        )
        conn.commit()
        print("Datos sembrados exitosamente en la BD ACID.")
    except Exception as e:
        conn.rollback()
        print(f"Rollback por error al sembrar: {e}")
        raise
    finally:
        conn.close()

    return db_manager.get_all_properties()


if __name__ == "__main__":
    db = ACIDDatabaseManager()
    data = ensure_data_seeded(db)
    print("Muestra de datos cargados de la BD ACID:")
    print(data.head(5))
