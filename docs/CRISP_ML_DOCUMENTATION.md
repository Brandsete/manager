# Documentación del Ciclo de Vida Machine Learning: CRISP-ML(Q)

Este proyecto implementa el estándar industrial **CRISP-ML(Q)** (*Cross-Industry Standard Process for the development of Machine Learning applications with Quality assurance*).

---

## 1. Fase 1: Business and Data Understanding (Comprensión del Negocio y los Datos)

### 1.1 Objetivo del Negocio
- Desarrollar un sistema automatizado de estimación de avalúos comerciales de inmuebles residenciales.
- Reducir el tiempo de cotización de días a segundos con un margen de error menor al 10%.
- Almacenar las cotizaciones e inmuebles bajo un estándar seguro con trazabilidad transaccional.

### 1.2 Comprensión de los Datos
Las variables seleccionadas para explicar el precio de venta son:
- **`area_m2`**: Área privada construida en metros cuadrados (rango típico: 25 a 600 m²).
- **`habitaciones`**: Cantidad de habitaciones/alcobas.
- **`banos`**: Cantidad de baños en la propiedad.
- **`parqueaderos`**: Número de plazas de estacionamiento privado.
- **`antiguedad_anos`**: Años de construcción del inmueble (genera depreciación gradual).
- **`estrato`**: Estratificación socioeconómica (1 a 6).
- **`zona`**: Ubicación urbana (Norte, Sur, Oriente, Occidente, Centro, Campestre).
- **`tiene_piscina`**: Amenidad de zona húmeda / recreativa (binario: 0 o 1).
- **`tiene_ascensor`**: Acceso vertical mecánico en edificio (binario: 0 o 1).
- **Variable Objetivo (`target`)**: `precio_millones` (Valor comercial en millones de COP).

---

## 2. Fase 2: Data Preparation (Preparación de Datos y Aseguramiento de Calidad)

- **Validación de Integridad**: Implementada a nivel de base de datos relacional mediante `CHECK constraints` de SQL (p. ej. `area_m2 >= 15`, `estrato BETWEEN 1 AND 6`).
- **Pipelines de Preprocesamiento**:
  - `StandardScaler`: Normalización de media 0 y varianza 1 para variables numéricas (`area_m2`, `habitaciones`, `banos`, `antiguedad_anos`, etc.).
  - `OneHotEncoder`: Codificación de variables categóricas nominales (`zona`), omitiendo la primera columna (`drop='first'`) para evitar multicolinealidad.
- **Partición de Datos**: División fija con semilla pseudoaleatoria (`random_state=42`) en 80% entrenamiento y 20% prueba.

---

## 3. Fase 3: Modeling (Modelado y Experimentación)

Se implementaron y compararon tres familias de modelos de regresión:
1. **Random Forest Regressor**: Ensamble no paramétrico de árboles de decisión (Bagging). Altamente efectivo para capturar interacciones no lineales entre estrato, zona y metros cuadrados.
2. **Gradient Boosting Regressor**: Ensamble secuencial (Boosting) enfocado en minimizar residuales iterativamente.
3. **Ridge Regression**: Modelo lineal regularizado con penalidad $L_2$ como benchmark de referencia.

---

## 4. Fase 4: Evaluation (Evaluación del Desempeño)

Métricas evaluadas sobre el conjunto de test independiente:

| Algoritmo | $R^2$ Score | RMSE ($ Millones) | MAE ($ Millones) | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **0.9886** | **$70.72M** | **$44.05M** | **~5.8%** |
| Gradient Boosting Regressor | 0.9881 | $72.07M | $42.22M | ~5.6% |
| Ridge Regression (Lineal) | 0.9749 | $104.82M | $81.03M | ~11.2% |

**Selección final:** El modelo **Random Forest Regressor** fue elegido como modelo productivo por su balance superior de varianza explicada ($R^2 > 0.98$) y estabilidad en rangos extremos de metraje.

---

## 5. Fase 5: Deployment (Despliegue)

- El pipeline de Scikit-Learn completo (incluyendo transformadores y ensamble) se serializa en `models/best_housing_model.joblib`.
- La inferencia se sirve a través de la aplicación interactiva construida en **Streamlit** (`app.py`).
- Cada inferencia se ejecuta con latencia inferior a 50 ms.

---

## 6. Fase 6: Monitoring & Maintenance (Monitoreo y Mantenimiento)

- **Bitácora Transaccional**: Toda inferencia solicitada en la interfaz se guarda atómicamente en la tabla `prediction_logs` con un identificador único `UUID`, fecha y datos de entrada.
- **Registro de Modelos**: La tabla `model_registry` almacena el histórico de versiones entrenadas junto con sus métricas.
- **Detección de Deriva (Drift)**: La bitácora permite analizar si las solicitudes de cotización se desvían de las distribuciones históricas de entrenamiento para disparar reentrenamientos en caliente desde el dashboard.
