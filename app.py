"""
=============================================================================
SISTEMA DE PREDICCIÓN DE PRECIOS DE VIVIENDA (HOUSE PRICE PREDICTOR)
Arquitectura Integral:
- Metodología de Desarrollo: Ágil (Scrum / XP)
- Ciclo de Vida Machine Learning: CRISP-ML(Q)
- Base de Datos Transaccional: SQLite con Garantías ACID
- Dashboard Interactivo: Streamlit
- Control de Versiones: GitHub Ready
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar directorio raíz al path para importaciones limpias
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from database.db_manager import ACIDDatabaseManager
from src.data_loader import ensure_data_seeded
from src.predictor import HousePricePredictor
from src.train_model import train_and_evaluate_all_models

# Configuración de página
st.set_page_config(
    page_title="UniSafe | Valuador Inmobiliario ML & ACID",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS modernos y elegantes
st.markdown(
    """
    <style>
    /* Estilos generales */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-bar {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }
    .tech-badge {
        background-color: #F1F5F9;
        color: #1E293B;
        border: 1px solid #CBD5E1;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .tech-badge-primary {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .tech-badge-success {
        background-color: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .acid-box {
        background: #F8FAFC;
        border-left: 4px solid #10B981;
        border-radius: 0 8px 8px 0;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Inicializar servicios con cache de Streamlit
@st.cache_resource
def get_database_manager():
    db = ACIDDatabaseManager()
    ensure_data_seeded(db)
    return db

@st.cache_resource
def get_predictor():
    return HousePricePredictor()

db_manager = get_database_manager()
predictor = get_predictor()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=400&q=80", caption="Valuación Inmobiliaria Inteligente")
    st.title("⚙️ Panel de Control")
    st.markdown("**Estructura del Proyecto:**")
    st.markdown("""
    - 🔄 **Metodología:** Ágil (Scrum + XP)
    - 🧠 **Ciclo ML:** CRISP-ML(Q)
    - 🛡️ **Persistencia:** SQLite ACID
    - 💻 **Despliegue:** Streamlit
    - 🐙 **Control:** Git / GitHub
    """)
    st.divider()

    st.subheader("Estado de la Base de Datos")
    try:
        conn = db_manager.get_connection()
        total_inmuebles = conn.execute("SELECT COUNT(*) FROM properties").fetchone()[0]
        total_predicciones = conn.execute("SELECT COUNT(*) FROM prediction_logs").fetchone()[0]
        total_auditoria = conn.execute("SELECT COUNT(*) FROM acid_audit_log").fetchone()[0]
        conn.close()

        st.metric("Inmuebles en BD ACID", f"{total_inmuebles:,}")
        st.metric("Predicciones Auditadas", f"{total_predicciones:,}")
        st.metric("Transacciones ACID", f"{total_auditoria:,}")
    except Exception as e:
        st.error(f"Error al consultar BD: {e}")

    st.divider()
    st.info("💡 Este sistema implementa integridad transaccional serializada y monitoreo de inferencia en tiempo real.")

# ----------------- ENCABEZADO PRINCIPAL -----------------
st.markdown('<div class="main-header">🏡 Sistema Inteligente de Predicción de Precios de Casas</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Arquitectura de Software Profesional: Metodología Ágil (Scrum/XP) | CRISP-ML | Persistencia ACID | Streamlit</div>', unsafe_allow_html=True)

st.markdown("""
<div class="badge-bar">
    <span class="tech-badge-primary">📌 Metodología: Scrum + XP</span>
    <span class="tech-badge-success">🧠 ML Framework: CRISP-ML(Q)</span>
    <span class="tech-badge">🛡️ Datos: SQLite ACID Transactions (WAL)</span>
    <span class="tech-badge-primary">📊 Dashboard: Streamlit</span>
    <span class="tech-badge">🐙 Repositorio: GitHub Versioned</span>
</div>
""", unsafe_allow_html=True)

# ----------------- TABS PRINCIPALES -----------------
tab_prediccion, tab_datos, tab_modelos, tab_acid, tab_docs = st.tabs([
    "🏠 1. Cotizador & Predicción",
    "📊 2. Explorador de Datos (ACID)",
    "🧠 3. Modelado CRISP-ML",
    "🛡️ 4. Auditoría y Garantías ACID",
    "📖 5. Documentación de Ingeniería"
])

# =============================================================================
# TAB 1: COTIZADOR & PREDICCIÓN EN TIEMPO REAL
# =============================================================================
with tab_prediccion:
    st.markdown("### 📝 Ingrese las Características del Inmueble")
    st.write("Complete el formulario para calcular el avalúo comercial mediante el modelo optimizado de Machine Learning.")

    col_form1, col_form2 = st.columns([1, 1], gap="large")

    with col_form1:
        st.markdown("#### 📐 Dimensiones y Distribución")
        cliente_nombre = st.text_input("Nombre del Solicitante / Propietario", value="Inversionista Inmobiliario")
        area_m2 = st.slider("Área Construida (m²)", min_value=25.0, max_value=600.0, value=110.0, step=2.5)

        col_h, col_b = st.columns(2)
        with col_h:
            habitaciones = st.number_input("Número de Habitaciones", min_value=1, max_value=10, value=3)
        with col_b:
            banos = st.number_input("Número de Baños", min_value=1, max_value=8, value=2)

        col_p, col_a = st.columns(2)
        with col_p:
            parqueaderos = st.selectbox("Parqueaderos Privados", [0, 1, 2, 3, 4], index=1)
        with col_a:
            antiguedad_anos = st.slider("Antigüedad del Inmueble (Años)", min_value=0, max_value=50, value=5)

    with col_form2:
        st.markdown("#### 📍 Ubicación y Amenidades")
        zona = st.selectbox(
            "Zona Geográfica de la Ciudad",
            ["Norte", "Sur", "Oriente", "Occidente", "Centro", "Campestre"],
            index=0,
            help="El sector incide significativamente en el valor por metro cuadrado."
        )
        estrato = st.select_slider(
            "Estrato Socioeconómico",
            options=[1, 2, 3, 4, 5, 6],
            value=4,
            help="Escala de estratificación socioeconómica (1 a 6)."
        )

        st.markdown("##### Extras y Equipamiento:")
        col_chk1, col_chk2 = st.columns(2)
        with col_chk1:
            tiene_piscina = st.checkbox("🏊‍♂️ Piscina / Zona Húmeda", value=False)
        with col_chk2:
            tiene_ascensor = st.checkbox("🛗 Ascensor en Edificio", value=True)

        st.markdown("---")
        guardar_en_bd = st.checkbox(
            "🔒 Registrar esta cotización en la Base de Datos ACID",
            value=True,
            help="Asegura el registro transaccional con Atomicidad y Durabilidad."
        )

    # Botón de cálculo
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        btn_predecir = st.button("⚡ Calcular Avalúo Comercial", type="primary", use_container_width=True)

    if btn_predecir:
        payload = {
            "area_m2": area_m2,
            "habitaciones": habitaciones,
            "banos": banos,
            "parqueaderos": parqueaderos,
            "antiguedad_anos": antiguedad_anos,
            "estrato": estrato,
            "zona": zona,
            "tiene_piscina": 1 if tiene_piscina else 0,
            "tiene_ascensor": 1 if tiene_ascensor else 0,
        }

        with st.spinner("Ejecutando inferencia ML y verificando transacción ACID..."):
            res = predictor.predict(
                features=payload,
                client_name=cliente_nombre,
                save_to_db=guardar_en_bd,
            )

        st.success("✅ Avalúo calculado y validado exitosamente")

        # Tarjetas de resultados
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                "💰 Precio Estimado",
                f"${res['precio_estimado_millones']:,.2f} M",
                help="Valor central proyectado por el modelo Random Forest Regressor."
            )
        with c2:
            st.metric(
                "📉 Rango Mínimo (-6%)",
                f"${res['precio_min_millones']:,.2f} M",
            )
        with c3:
            st.metric(
                "📈 Rango Máximo (+6%)",
                f"${res['precio_max_millones']:,.2f} M",
            )
        with c4:
            st.metric(
                "📐 Valor por m²",
                f"${res['precio_m2_millones']:,.2f} M/m²",
            )

        if guardar_en_bd and res["transaction_uuid"]:
            st.markdown(
                f"""
                <div class="acid-box">
                    <strong>🛡️ Garantía ACID Confirmada:</strong><br>
                    Transacción comprometida (COMMITTED) en SQLite en modo WAL.<br>
                    <strong>UUID de Transacción:</strong> <code>{res['transaction_uuid']}</code><br>
                    <em>Estado: {res['db_status']}</em>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Simulador Financiero Hipotecario
        st.subheader("🏦 Simulador Financiero de Crédito de Vivienda (Opcional)")
        col_fin1, col_fin2 = st.columns(2)
        with col_fin1:
            cuota_inicial_pct = st.slider("Porcentaje de Cuota Inicial (%)", 10, 50, 30)
            plazo_anos = st.selectbox("Plazo de Financiación", [5, 10, 15, 20, 30], index=3)
        with col_fin2:
            tasa_anual = st.number_input("Tasa de Interés Efectiva Anual (E.A. %)", value=12.5, step=0.5)

            # Cálculo de cuota fija mensual aproximada (Sistema Francés)
            monto_credito = res['precio_estimado_millones'] * (1 - cuota_inicial_pct / 100.0)
            tasa_mensual = ((1 + tasa_anual / 100.0) ** (1/12)) - 1
            n_meses = plazo_anos * 12
            cuota_mensual = monto_credito * (tasa_mensual * ((1 + tasa_mensual)**n_meses)) / (((1 + tasa_mensual)**n_meses) - 1)

            st.metric("Cuota Mensual Estimada", f"${cuota_mensual:,.2f} M / mes")
            st.caption(f"Financiando ${monto_credito:,.2f} M a {plazo_anos} años.")

# =============================================================================
# TAB 2: EXPLORADOR DE DATOS (BD ACID)
# =============================================================================
with tab_datos:
    st.markdown("### 📊 Explorador de Datos de Mercado Inmobiliario")
    st.write("Datos históricos almacenados y administrados bajo el estándar relacional **ACID** (Atomicidad, Consistencia, Aislamiento, Durabilidad).")

    df_properties = db_manager.get_all_properties()

    # Filtros interactivos
    col_fil1, col_fil2, col_fil3 = st.columns(3)
    with col_fil1:
        zonas_seleccionadas = st.multiselect(
            "Filtrar por Zona",
            options=df_properties["zona"].unique().tolist(),
            default=df_properties["zona"].unique().tolist()
        )
    with col_fil2:
        estratos_seleccionados = st.multiselect(
            "Filtrar por Estrato",
            options=sorted(df_properties["estrato"].unique().tolist()),
            default=sorted(df_properties["estrato"].unique().tolist())
        )
    with col_fil3:
        precio_max = float(df_properties["precio_millones"].max())
        rango_precio = st.slider(
            "Rango de Precio (Millones)",
            min_value=float(df_properties["precio_millones"].min()),
            max_value=precio_max,
            value=(float(df_properties["precio_millones"].min()), precio_max)
        )

    # Filtrado
    df_filtrado = df_properties[
        (df_properties["zona"].isin(zonas_seleccionadas)) &
        (df_properties["estrato"].isin(estratos_seleccionados)) &
        (df_properties["precio_millones"] >= rango_precio[0]) &
        (df_properties["precio_millones"] <= rango_precio[1])
    ]

    st.write(f"Mostrando **{len(df_filtrado)}** de **{len(df_properties)}** registros almacenados.")

    # Gráficos de distribución
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("##### 📍 Precio Promedio por Zona")
        precio_zona = df_filtrado.groupby("zona")["precio_millones"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=precio_zona, x="zona", y="precio_millones", palette="Blues_d", ax=ax)
        ax.set_ylabel("Precio Promedio ($ Millones)")
        ax.set_xlabel("Zona")
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        st.pyplot(fig)
        plt.close()

    with col_g2:
        st.markdown("##### 📐 Relación Área (m²) vs Precio")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sns.scatterplot(
            data=df_filtrado,
            x="area_m2",
            y="precio_millones",
            hue="estrato",
            palette="viridis",
            alpha=0.75,
            ax=ax2
        )
        ax2.set_xlabel("Área Construida (m²)")
        ax2.set_ylabel("Precio ($ Millones)")
        ax2.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig2)
        plt.close()

    # Tabla de datos
    st.markdown("##### 📋 Tabla de Propiedades (SQLite ACID)")
    st.dataframe(df_filtrado, use_container_width=True, height=280)

# =============================================================================
# TAB 3: MODELADO CRISP-ML(Q)
# =============================================================================
with tab_modelos:
    st.markdown("### 🧠 Ciclo de Vida del Modelo: CRISP-ML(Q)")
    st.write("Fases 3 y 4 de CRISP-ML: Selección de algoritmos, validación cruzada y métricas de desempeño sobre el conjunto de test.")

    models_df = db_manager.get_model_registry()

    col_m1, col_m2 = st.columns([1.2, 1])

    with col_m1:
        st.markdown("#### 🏆 Comparativa de Algoritmos Evaluados")
        st.dataframe(models_df, use_container_width=True)

        st.markdown("""
        **Interpretación de Métricas:**
        - **$R^2$ (Coeficiente de Determinación):** Porcentaje de la varianza del precio explicada por el modelo (Cercano a 1.0 es óptimo).
        - **RMSE (Raíz del Error Cuadrático Medio):** Magnitud promedio de los errores en Millones de pesos penalizando grandes desviaciones.
        - **MAE (Error Absoluto Medio):** Promedio directo de la desviación en Millones de pesos.
        """)

    with col_m2:
        st.markdown("#### 🎯 Re-entrenar Modelos en Caliente")
        st.write("Ejecuta el pipeline completo de preprocesamiento, escalado, one-hot encoding y entrenamiento con split 80/20.")
        if st.button("🔄 Ejecutar Reentrenamiento CRISP-ML", type="secondary"):
            with st.spinner("Entrenando modelos y calculando métricas..."):
                train_out = train_and_evaluate_all_models()
                st.success(f"¡Entrenamiento completado! Mejor modelo: {train_out['best_model_name']}")
                st.rerun()

    st.divider()

    # Visualización de métricas
    st.markdown("#### 📈 Comparación de Precisión ($R^2$ Score)")
    if not models_df.empty:
        fig_r2, ax_r2 = plt.subplots(figsize=(8, 3.5))
        sns.barplot(data=models_df, x="model_name", y="r2_score", palette="crest", ax=ax_r2)
        ax_r2.set_ylim(0.85, 1.0)
        ax_r2.set_ylabel("R² Score")
        ax_r2.set_xlabel("Modelo")
        for p in ax_r2.patches:
            ax_r2.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
        st.pyplot(fig_r2)
        plt.close()

# =============================================================================
# TAB 4: AUDITORÍA Y GARANTÍAS ACID
# =============================================================================
with tab_acid:
    st.markdown("### 🛡️ Arquitectura de Datos Transaccional (ACID)")
    st.write("Demostración práctica de cómo se garantizan las 4 propiedades fundamentales de confiabilidad de datos:")

    c_a, c_c, c_i, c_d = st.columns(4)
    with c_a:
        st.markdown("""
        **A - Atomicidad**
        O todo se ejecuta o nada se aplica. Cada avalúo y registro de propiedad se ejecuta en una transacción `BEGIN IMMEDIATE` y `COMMIT`. Ante fallo, se aplica `ROLLBACK`.
        """)
    with c_c:
        st.markdown("""
        **C - Consistencia**
        Restricciones de integridad (CHECK constraints) en base de datos: áreas positivas, estrato entre 1 y 6, zonas válidas y tipos estrictos.
        """)
    with c_i:
        st.markdown("""
        **I - Aislamiento**
        SQLite configurado en modo **WAL (Write-Ahead Logging)**, permitiendo lectores concurrentes sin bloquear escrituras transaccionales.
        """)
    with c_d:
        st.markdown("""
        **D - Durabilidad**
        Las operaciones confirmadas persisten en el almacenamiento secundario (`real_estate_acid.db`), sobreviviendo a reinicios y caídas de servidor.
        """)

    st.divider()

    st.markdown("#### 🧪 Simulador de Prueba de Estrés ACID: Intento de Inserción Inválida")
    st.write("Comprueba cómo la base de datos rechaza datos corruptos y ejecuta **ROLLBACK automático** preservando la consistencia:")

    col_test1, col_test2 = st.columns(2)
    with col_test1:
        test_area = st.number_input("Área m² (Prueba inválida con valor negativo)", value=-15.0)
        test_estrato = st.number_input("Estrato (Prueba inválida fuera de rango 1-6)", value=9)

    with col_test2:
        st.write("")
        st.write("")
        if st.button("⚠️ Ejecutar Inserción de Prueba", type="secondary"):
            test_payload = {
                "codigo_inmueble": "TEST-CORRUPT-01",
                "area_m2": test_area,
                "habitaciones": 3,
                "banos": 2,
                "estrato": test_estrato,
                "zona": "Norte",
                "precio_millones": 250.0
            }
            success, msg = db_manager.insert_property_atomic(test_payload)
            if success:
                st.success(f"Insertado: {msg}")
            else:
                st.error(f"🛡️ Rollback activado exitosamente:\n{msg}")

    st.divider()

    st.markdown("#### 📜 Registro de Auditoría Transaccional (`acid_audit_log`)")
    audit_df = db_manager.get_audit_logs(limit=20)
    st.dataframe(audit_df, use_container_width=True)

    st.markdown("#### 🕒 Historial de Predicciones Comprometidas (`prediction_logs`)")
    preds_df = db_manager.get_prediction_history(limit=20)
    st.dataframe(preds_df, use_container_width=True)

# =============================================================================
# TAB 5: DOCUMENTACIÓN DE INGENIERÍA
# =============================================================================
with tab_docs:
    st.markdown("### 📚 Documentación Formal del Proyecto")
    st.markdown("""
    Esta solución fue construida según el diagrama de arquitectura especificado en la pizarra:
    - **Ágil (Scrum / XP):** Iteraciones cortas, refactorización continua, pruebas automáticas e integración.
    - **CRISP-ML(Q):** Metodología estándar de la industria para proyectos de Machine Learning.
    - **Datos ACID:** Motor transaccional relacional con aislamiento WAL y claves foráneas.
    - **Streamlit:** Interfaz interactiva de alta fidelidad para usuarios de negocio e ingenieros.
    - **GitHub:** Listo para control de versiones y CI/CD.
    """)

    st.markdown("#### 🔄 Mapeo del Ciclo CRISP-ML(Q)")
    fases_data = {
        "Fase CRISP-ML": [
            "1. Business & Data Understanding",
            "2. Data Preparation",
            "3. Modeling",
            "4. Evaluation",
            "5. Deployment",
            "6. Monitoring & Maintenance"
        ],
        "Objetivo en la Aplicación": [
            "Identificar factores clave del valor de vivienda (área, zona, estrato, amenidades).",
            "Limpieza de datos, codificación One-Hot de zonas, escalamiento estándar (StandardScaler).",
            "Entrenamiento de modelos supervisados (Random Forest, Gradient Boosting, Ridge).",
            "Cálculo de métricas cuantitativas: R2, RMSE y MAE sobre conjunto de validación.",
            "Despliegue interactivo con Streamlit y registro atómico en SQLite.",
            "Bitácora de auditoría ACID (`prediction_logs` y `acid_audit_log`) para detectar deriva."
        ],
        "Módulo de Código": [
            "src/data_loader.py",
            "src/train_model.py (Pipeline preprocessor)",
            "src/train_model.py (RandomForestRegressor)",
            "src/train_model.py (r2_score, mean_squared_error)",
            "app.py & src/predictor.py",
            "database/db_manager.py (WAL Mode & Transactions)"
        ]
    }
    st.table(pd.DataFrame(fases_data))

    st.markdown("#### 🏃‍♂️ Metodología Ágil: Scrum y Prácticas XP")
    st.markdown("""
    - **Historias de Usuario (Scrum):**
        - *HU-01:* Como comprador o tasador, quiero ingresar las especificaciones de una casa y obtener el precio estimado al instante.
        - *HU-02:* Como analista de riesgos, quiero que cada cotización quede guardada bajo ACID sin pérdida de transacciones.
        - *HU-03:* Como científico de datos, quiero comparar métricas de modelos y reentrenar en caliente desde la plataforma.
    - **Prácticas XP (eXtreme Programming) adoptadas:**
        - **Simple Design:** Arquitectura modular desacoplada (`database/`, `src/`, `models/`, `app.py`).
        - **Refactoring:** Pipeline de Scikit-Learn encapsulado y reutilizable.
        - **Continuous Testing:** Validaciones de consistencia en esquema SQL y asserts de predicción.
    """)

    st.markdown("#### 🐙 Estructura de Repositorio (GitHub Ready)")
    st.code("""
house_price_predictor/
│
├── database/
│   ├── schema.sql           # DDL relacional con CHECK constraints
│   ├── db_manager.py        # Gestor de transacciones ACID (BEGIN/COMMIT/ROLLBACK)
│   └── real_estate_acid.db  # Base de datos SQLite persistente (WAL mode)
│
├── src/
│   ├── data_loader.py       # Fase 1 y 2 CRISP-ML (Carga y semilla de datos)
│   ├── train_model.py       # Fase 3 y 4 CRISP-ML (Entrenamiento y evaluación)
│   └── predictor.py         # Fase 5 CRISP-ML (Inferencia y persistencia)
│
├── models/
│   ├── best_housing_model.joblib  # Pipeline de Machine Learning serializado
│   └── models_metadata.json       # Métricas y configuración del modelo
│
├── docs/
│   ├── CRISP_ML_DOCUMENTATION.md  # Detalle completo de fases CRISP-ML
│   ├── METODOLOGIA_AGIL_SCRUM_XP.md# Historias de usuario y ceremonias ágiles
│   └── ARQUITECTURA_ACID.md       # Demostración matemática y técnica de ACID
│
├── app.py                   # Dashboard interactivo en Streamlit
├── requirements.txt         # Dependencias del proyecto
├── .gitignore               # Configuración para GitHub
└── README.md                # Documentación principal del software
    """, language="text")

st.markdown("---")
st.caption("UniSafe Software Engineering © 2026 | Arquitectura Ágil + CRISP-ML + SQLite ACID + Streamlit")
