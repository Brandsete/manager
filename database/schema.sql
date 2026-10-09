-- =====================================================================
-- ESQUEMA DE BASE DE DATOS TRANSACCIONAL ACID PARA PREDICCIÓN INMOBILIARIA
-- Cumple con:
-- A: Atomicidad (Transacciones completas o rollback)
-- C: Consistencia (Constraints CHECK, Foreign Keys, Not Null)
-- I: Aislamiento (Control de concurrencia y bloqueos transaccionales)
-- D: Durabilidad (Almacenamiento persistente con WAL y sync)
-- =====================================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- 1. Tabla de Propiedades Históricas (Datos de entrenamiento y mercado)
CREATE TABLE IF NOT EXISTS properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    codigo_inmueble TEXT UNIQUE NOT NULL,
    area_m2 REAL NOT NULL CHECK(area_m2 >= 15 AND area_m2 <= 1000),
    habitaciones INTEGER NOT NULL CHECK(habitaciones >= 1 AND habitaciones <= 15),
    banos INTEGER NOT NULL CHECK(banos >= 1 AND banos <= 10),
    parqueaderos INTEGER NOT NULL DEFAULT 0 CHECK(parqueaderos >= 0 AND parqueaderos <= 6),
    antiguedad_anos INTEGER NOT NULL DEFAULT 0 CHECK(antiguedad_anos >= 0 AND antiguedad_anos <= 100),
    estrato INTEGER NOT NULL CHECK(estrato BETWEEN 1 AND 6),
    zona TEXT NOT NULL CHECK(zona IN ('Norte', 'Sur', 'Oriente', 'Occidente', 'Centro', 'Campestre')),
    tiene_piscina INTEGER NOT NULL DEFAULT 0 CHECK(tiene_piscina IN (0, 1)),
    tiene_ascensor INTEGER NOT NULL DEFAULT 0 CHECK(tiene_ascensor IN (0, 1)),
    precio_millones REAL NOT NULL CHECK(precio_millones > 20 AND precio_millones < 20000),
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Tabla de Cotizaciones y Predicciones en Tiempo Real (Log Transaccional)
CREATE TABLE IF NOT EXISTS prediction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_uuid TEXT UNIQUE NOT NULL,
    cliente_nombre TEXT NOT NULL,
    area_m2 REAL NOT NULL CHECK(area_m2 > 0),
    habitaciones INTEGER NOT NULL,
    banos INTEGER NOT NULL,
    parqueaderos INTEGER NOT NULL,
    antiguedad_anos INTEGER NOT NULL,
    estrato INTEGER NOT NULL,
    zona TEXT NOT NULL,
    tiene_piscina INTEGER NOT NULL,
    tiene_ascensor INTEGER NOT NULL,
    precio_estimado_millones REAL NOT NULL CHECK(precio_estimado_millones > 0),
    precio_min_millones REAL NOT NULL,
    precio_max_millones REAL NOT NULL,
    modelo_utilizado TEXT NOT NULL,
    estado_transaccion TEXT NOT NULL CHECK(estado_transaccion IN ('COMMITTED', 'ROLLED_BACK', 'PENDING')),
    fecha_prediccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Tabla de Métricas de Modelos ML (Versionamiento y Monitoreo CRISP-ML)
CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    r2_score REAL NOT NULL,
    rmse REAL NOT NULL,
    mae REAL NOT NULL,
    train_samples INTEGER NOT NULL,
    test_samples INTEGER NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 0 CHECK(is_active IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Tabla de Auditoría ACID (Trazabilidad y Durabilidad de operaciones críticas)
CREATE TABLE IF NOT EXISTS acid_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_type TEXT NOT NULL,
    table_affected TEXT NOT NULL,
    record_id INTEGER,
    details TEXT NOT NULL,
    isolation_level TEXT NOT NULL DEFAULT 'SERIALIZABLE',
    status TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Índices para optimizar consultas de analítica
CREATE INDEX IF NOT EXISTS idx_properties_zona_estrato ON properties(zona, estrato);
CREATE INDEX IF NOT EXISTS idx_properties_precio ON properties(precio_millones);
CREATE INDEX IF NOT EXISTS idx_predictions_fecha ON prediction_logs(fecha_prediccion);
