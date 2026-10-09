# Arquitectura de Base de Datos y Garantías ACID

Este documento describe la fundamentación técnica y la implementación práctica de las propiedades **ACID** (*Atomicity, Consistency, Isolation, Durability*) en el sistema de predicción inmobiliaria.

---

```
             ┌─────────────────────────┐
             │       APLICACIÓN        │
             │     (Streamlit UI)      │
             └────────────┬────────────┘
                          │ Inferencia & Cotización
                          ▼
             ┌─────────────────────────┐
             │     INFERENCE ENGINE    │
             │   (Random Forest ML)    │
             └────────────┬────────────┘
                          │ Transacción Atómica
                          ▼
            ╔═══════════════════════════╗
            ║     MOTOR DE DATOS ACID   ║
            ║         (SQLite)          ║
            ║ ───────────────────────── ║
            ║  • [A] Atomicidad         ║
            ║  • [C] Consistencia       ║
            ║  • [I] Aislamiento (WAL)  ║
            ║  • [D] Durabilidad        ║
            ╚═══════════════════════════╝
```

---

## 1. Desglose de las Propiedades ACID

### 1.1 Atomicidad (Atomicity - Todo o Nada)
- **Principio**: Una transacción se ejecuta completamente o sus efectos son revertidos en su totalidad en caso de falla.
- **Implementación**:
  ```python
  conn = db_manager.get_connection()
  try:
      cursor = conn.cursor()
      cursor.execute("BEGIN IMMEDIATE TRANSACTION;")
      cursor.execute("INSERT INTO prediction_logs (...) VALUES (...);")
      cursor.execute("INSERT INTO acid_audit_log (...) VALUES (...);")
      conn.commit()  # Ambas inserciones se consolidan juntas
  except Exception:
      conn.rollback()  # Si alguna falla, ninguna persiste
  ```

### 1.2 Consistencia (Consistency - Reglas de Integridad)
- **Principio**: La base de datos solo transita de un estado válido a otro, cumpliendo todas las reglas y restricciones del esquema.
- **Implementación en DDL**:
  - Restricciones de Dominio: `CHECK(area_m2 >= 15 AND area_m2 <= 1000)`
  - Restricciones de Rango: `CHECK(estrato BETWEEN 1 AND 6)`
  - Restricciones de Enumeración: `CHECK(zona IN ('Norte', 'Sur', ...))`
  - Restricciones de Unicidad: `transaction_uuid TEXT UNIQUE NOT NULL`

### 1.3 Aislamiento (Isolation - Concurrencia Segura)
- **Principio**: Las operaciones concurrentes no interfieren entre sí.
- **Implementación**:
  - Se activa el modo **WAL (Write-Ahead Logging)**:
    ```sql
    PRAGMA journal_mode = WAL;
    ```
  - Permite que múltiples hilos de consulta lean la base de datos sin bloquear al hilo que está registrando una nueva cotización.

### 1.4 Durabilidad (Durability - Persistencia Confiable)
- **Principio**: Una vez que una transacción ha hecho `COMMIT`, sus efectos son permanentes y no se pierden ante apagados o fallos de software.
- **Implementación**:
  - Se configura la directiva de sincronización a disco:
    ```sql
    PRAGMA synchronous = NORMAL;
    ```
  - Los datos confirmados se escriben físicamente en el archivo de base de datos (`real_estate_acid.db`).

---

## 2. Registro de Auditoría y Verificación

Cada evento crítico genera una entrada en la tabla `acid_audit_log`, garantizando la observabilidad y trazabilidad completa del ciclo de vida del dato.
