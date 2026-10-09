# Metodología Ágil de Desarrollo: Scrum y eXtreme Programming (XP)

Este documento detalla la aplicación de los marcos **Scrum** y **XP** en el ciclo de vida del software para la predicción de precios inmobiliarios.

---

## 1. Justificación de la Combinación Scrum + XP

- **Scrum** proporciona la **estructura de gestión**, gobernanza, roles y ceremonias (Sprints, Backlog, Daily, Retrospectivas).
- **eXtreme Programming (XP)** aporta las **disciplinas técnicas y de ingeniería**, enfocadas en la excelencia del código, pruebas automatizadas, refactorización y diseño simple.

---

## 2. Marco de Trabajo Scrum

### 2.1 Roles del Equipo
- **Product Owner (PO):** Define la visión del producto, prioriza el Product Backlog y asegura que el modelo de predicción responda a las necesidades de negocio del sector inmobiliario.
- **Scrum Master:** Facilita las ceremonias, elimina impedimentos técnicos y metodológicos, y vela por la adherencia al proceso.
- **Developers / ML Engineers:** Equipo multidisciplinario encargado del desarrollo de la arquitectura ACID, pipelines de ML y el frontend en Streamlit.

### 2.2 Historias de Usuario (Product Backlog)

| ID | Historia de Usuario | Criterio de Aceptación | Estimación (Story Points) |
| :--- | :--- | :--- | :---: |
| **US-01** | Como cliente o tasador, quiero ingresar datos de un inmueble para obtener el precio estimado de mercado al instante. | Predicción calculada en < 1 segundo con desglose de márgenes (+/- 6%). | 5 |
| **US-02** | Como oficial de cumplimiento, quiero que cada cotización persista de manera segura sin pérdida de información ante fallas. | Registro con transacción ACID (UUID, estado COMMITTED o ROLLBACK). | 8 |
| **US-03** | Como analista de datos, quiero explorar la distribución geográfica y estratos de las propiedades registradas. | Gráficos interactivos con filtros dinámicos por zona y estrato en Streamlit. | 5 |
| **US-04** | Como ingeniero de ML, quiero comparar múltiples algoritmos y registrar sus métricas en una base de datos. | Métricas R², RMSE, MAE calculadas y guardadas en tabla `model_registry`. | 8 |
| **US-05** | Como administrador, quiero probar que la BD rechace registros corruptos (p. ej. áreas negativas). | Disparo de `ROLLBACK` y mensaje de error de integridad visible en la UI. | 3 |

---

## 3. Prácticas de Ingeniería eXtreme Programming (XP)

1. **Simple Design (Diseño Simple):**
   - El sistema se compone de capas modulares claras:
     - `database/`: Conexión, esquemas y transacciones ACID.
     - `src/`: Lógica de datos, entrenamiento e inferencia.
     - `models/`: Artefactos serializados.
     - `app.py`: Presentación visual e interactiva.

2. **Test-Driven Development / Pruebas Continuas:**
   - Verificación de consistencia relacional mediante constraints SQL.
   - Evaluación continua de residuales y umbrales de métricas antes de promover modelos a producción.

3. **Refactoring Continuo:**
   - Separación estricta entre la capa de persistencia y la de inferencia para permitir cambiar el motor de persistencia (ej. migrar de SQLite a PostgreSQL) sin alterar el modelo de predicción.

4. **Continuous Integration & GitHub Readiness:**
   - Estructura versionada con `.gitignore`, dependencias declarativas en `requirements.txt` y commits semánticos (`feat:`, `fix:`, `docs:`).
