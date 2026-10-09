# 🏡 House Price Predictor: Sistema Inteligente de Valuación Inmobiliaria

> **Proyecto de Ingeniería de Software y Machine Learning**  
> Diseñado según la arquitectura integral: **Metodología Ágil (Scrum/XP)**, ciclo de vida **CRISP-ML(Q)**, persistencia relacional con garantías **ACID** y despliegue interactivo en **Streamlit**.

---

## 📌 Tabla de Contenidos
1. [Estructura del Proyecto y Diagrama del Whiteboard](#-estructura-del-proyecto)
2. [Pila Tecnológica](#-pila-tecnológica)
3. [Metodología de Desarrollo: Ágil (Scrum / XP)](#-metodología-ágil-scrum--xp)
4. [Ciclo de Machine Learning: CRISP-ML(Q)](#-ciclo-de-machine-learning-crisp-mlq)
5. [Arquitectura de Datos: Transacciones ACID](#-arquitectura-de-datos-transacciones-acid)
6. [Instalación y Puesta en Marcha](#-instalación-y-puesta-en-marcha)
7. [Uso del Dashboard en Streamlit](#-uso-del-dashboard-en-streamlit)
8. [Publicación en GitHub](#-publicación-en-github)

---

## 📐 Estructura del Proyecto

El desarrollo refleja fielmente la especificación esquematizada en la pizarra:

```text
               ┌────────────────────────────────────────────────────────┐
               │         "Desarrollo de Software" (DOCUMENTAR)         │
               └───────────────────────────┬────────────────────────────┘
                                           │
          ┌────────────────────────────────┼────────────────────────────────┐
          ▼                                ▼                                ▼
  ┌───────────────┐              ┌───────────────────┐             ┌────────────────┐
  │  Aplicación   │              │   Método Ágil     │             │   Machine      │
  │    GitHub     │              │    Scrum / XP     │             │   Learning     │
  └───────────────┘              └─────────┬─────────┘             └───────┬────────┘
                                           │                               │
                                           └───────────► CRISP-ML ◄────────┘
                                                            │
                                  ┌─────────────────────────┴───────────────────────┐
                                  ▼                                                 ▼
                        ┌──────────────────┐                              ┌──────────────────┐
                        │   "Dashboard"    │                              │      DATOS       │
                        │    Streamlit     │                              │       ACID       │
                        └──────────────────┘                              └──────────────────┘
```

### Árbol de Directorios
```
house_price_predictor/
│
├── database/
│   ├── schema.sql                 # Esquema relacional DDL y CHECK constraints
│   ├── db_manager.py              # Gestor transaccional ACID (BEGIN, COMMIT, ROLLBACK)
│   └── real_estate_acid.db        # Base de datos SQLite persistente (WAL mode)
│
├── src/
│   ├── data_loader.py             # Fases 1 y 2 CRISP-ML: Carga y seed de datos
│   ├── train_model.py             # Fases 3 y 4 CRISP-ML: Entrenamiento y evaluación
│   └── predictor.py               # Fase 5 CRISP-ML: Inferencia y registro atómico
│
├── models/
│   ├── best_housing_model.joblib  # Modelo supervisado serializado
│   └── models_metadata.json       # Métricas de validación (R², RMSE, MAE)
│
├── docs/
│   ├── CRISP_ML_DOCUMENTATION.md  # Detalle formal de las 6 fases de CRISP-ML
│   ├── METODOLOGIA_AGIL_SCRUM_XP.md# Historias de usuario y disciplinas técnicas XP
│   └── ARQUITECTURA_ACID.md       # Demostración técnica de garantías ACID
│
├── app.py                         # Aplicación Web y Dashboard interactivo en Streamlit
├── requirements.txt               # Dependencias del entorno
├── .gitignore                     # Configuración de exclusiones Git
└── README.md                      # Documento principal del proyecto
```

---

## 🛠️ Pila Tecnológica

- **Lenguaje:** Python 3.12+
- **Machine Learning:** `scikit-learn`, `numpy`, `pandas`, `joblib`
- **Dashboard & Frontend:** `Streamlit`, `matplotlib`, `seaborn`
- **Base de Datos & Transacciones:** `SQLite3` con soporte transaccional inmediato y modo **WAL (Write-Ahead Logging)**
- **Control de Versiones:** `Git` / `GitHub`

---

## 🏃 Metodología Ágil (Scrum + XP)

Se adopta un enfoque híbrido:
- **Scrum**: Organización iterativa en Sprints de 1 a 2 semanas, Product Backlog priorizado por valor de negocio e Historias de Usuario claras.
- **eXtreme Programming (XP)**: Excelencia técnica mediante **Diseño Simple** (desacoplamiento total entre capas de base de datos, inferencia y presentación), **Refactorización continua** e **Integración continua**.

*(Consulte [`docs/METODOLOGIA_AGIL_SCRUM_XP.md`](docs/METODOLOGIA_AGIL_SCRUM_XP.md) para más detalles).*

---

## 🧠 Ciclo de Machine Learning (CRISP-ML)

El desarrollo del modelo predictivo sigue las 6 fases del estándar **CRISP-ML(Q)**:
1. **Business & Data Understanding**: Identificación de variables críticas (área construida, habitaciones, baños, parqueaderos, estrato, zona, amenidades).
2. **Data Preparation**: Limpieza, imputación nula, escalado con `StandardScaler` y codificación One-Hot de sectores urbanos.
3. **Modeling**: Entrenamiento comparativo de **Random Forest Regressor**, **Gradient Boosting Regressor** y **Ridge Regression**.
4. **Evaluation**: Validación con $R^2 = 0.9886$, $RMSE = \$70.72M$ y $MAE = \$44.05M$.
5. **Deployment**: Servicio de predicción en tiempo real integrado en Streamlit.
6. **Monitoring & Maintenance**: Registro histórico en base de datos para monitoreo y detección de deriva de datos (*Data Drift*).

*(Consulte [`docs/CRISP_ML_DOCUMENTATION.md`](docs/CRISP_ML_DOCUMENTATION.md) para el informe completo).*

---

## 🛡️ Arquitectura de Datos: Transacciones ACID

La base de datos relacional garantiza las 4 propiedades fundamentales:
- **A (Atomicidad)**: Las operaciones de guardado de avalúo y registro de auditoría se ejecutan en un único bloque `BEGIN IMMEDIATE ... COMMIT`. Si ocurre un fallo, se aplica `ROLLBACK` total.
- **C (Consistencia)**: Reglas de validación estrictas a nivel de base de datos (`area_m2 > 15`, `estrato BETWEEN 1 AND 6`, `zona IN (...)`).
- **I (Aislamiento)**: Configuración en modo **WAL** para lecturas sin bloqueos ante transacciones concurrentes.
- **D (Durabilidad)**: Escritura síncrona en disco en archivo relacional persistente.

*(Consulte [`docs/ARQUITECTURA_ACID.md`](docs/ARQUITECTURA_ACID.md) para la explicación técnica).*

---

## 🚀 Instalación y Puesta en Marcha

### 1. Clonar o acceder a la carpeta del proyecto
```bash
cd house_price_predictor
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. (Opcional) Reentrenar modelos y poblar la base de datos ACID
```bash
python -m src.train_model
```

### 4. Iniciar el Dashboard en Streamlit
```bash
streamlit run app.py
```
La aplicación se abrirá automáticamente en su navegador en `http://localhost:8501`.

---

## 📊 Uso del Dashboard en Streamlit

El dashboard ofrece 5 módulos interactivos:
1. **🏠 Cotizador & Predicción**: Formulario de especificaciones de la vivienda, cálculo instantáneo con márgenes, simulador de crédito hipotecario y guardado atómico con UUID.
2. **📊 Explorador de Datos (ACID)**: Filtros por zona, estrato y precio, con gráficos de dispersión y promedios.
3. **🧠 Modelado CRISP-ML**: Tabla de métricas comparativas ($R^2$, RMSE, MAE) y botón de reentrenamiento en vivo.
4. **🛡️ Auditoría y Garantías ACID**: Explicación interactiva, simulador de rechazo de datos corruptos con `ROLLBACK` y visualizador de transacciones.
5. **📖 Documentación de Ingeniería**: Mapa conceptual interactivo que conecta la teoría de software con el código implementado.

---

## 🐙 Publicación en GitHub

Para subir el proyecto a su repositorio en GitHub:

```bash
git init
git add .
git commit -m "feat: implementacion inicial completa de prediccion inmobiliaria con CRISP-ML, ACID y Streamlit"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/house-price-prediction.git
git push -u origin main
```
