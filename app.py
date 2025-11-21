import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time
import joblib
import json
from pathlib import Path
from collections import deque
from datetime import datetime

# ============================================================================
# ARQUITECTURA DEL MODELO (debe coincidir con el entrenamiento)
# ============================================================================

class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim=30, hidden_dims=[20, 10, 5]):
        super(FraudAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dims[0], input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ============================================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ============================================================================

st.set_page_config(
    page_title="Fraud Detection System | Real-Time Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS PERSONALIZADO PARA DISEÑO PROFESIONAL
# ============================================================================

st.markdown("""
<style>
    /* Importar fuentes modernas */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Variables de color - Tema oscuro profesional */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --danger-color: #ef4444;
        --warning-color: #f59e0b;
        --dark-bg: #0f172a;
        --card-bg: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
    }

    /* Fondo general */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Títulos principales */
    h1 {
        font-weight: 700 !important;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
    }

    h2, h3 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    /* Tarjetas de métricas mejoradas */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
    }

    /* Botones modernos */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Sliders modernos */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    }

    /* Tarjetas de información */
    .info-card {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }

    /* Separadores elegantes */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #6366f1, transparent);
        margin: 2rem 0;
    }

    /* Alertas personalizadas */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid var(--primary-color);
        background: rgba(99, 102, 241, 0.1);
    }

    /* DataFrames */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }

    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Animación de pulso para alertas */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    .pulse-animation {
        animation: pulse 2s ease-in-out infinite;
    }

    /* Badges personalizados */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }

    .badge-danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }

    .badge-info {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CARGA DEL MODELO Y DATOS
# ============================================================================

@st.cache_resource
def load_model_and_scaler():
    """Carga el modelo y el scaler una sola vez"""
    import json
    
    try:
        st.info("🔄 Cargando configuración...")
        
        # Buscar archivos en el directorio model/
        model_dir = Path('model')
        if not model_dir.exists():
            st.error("❌ No existe el directorio 'model/'")
            st.stop()
        
        # Buscar archivo .pth
        pth_files = list(model_dir.glob("*.pth"))
        if not pth_files:
            st.error("❌ No se encontró archivo .pth en model/")
            st.stop()
        
        pth_file = pth_files[0]
        st.info(f"📦 Usando modelo: {pth_file.name}")
        
        # Cargar checkpoint
        checkpoint = torch.load(pth_file, 
                               map_location='cpu',
                               weights_only=False)
        
        st.info("✅ Checkpoint cargado")
        
        # Inicializar modelo
        model = FraudAutoencoder()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        st.info("✅ Modelo inicializado")
        
        # Cargar scaler
        scaler_file = model_dir / 'scaler.joblib'
        if not scaler_file.exists():
            st.error(f"❌ No existe {scaler_file}")
            st.stop()
            
        scaler = joblib.load(scaler_file)
        st.info("✅ Scaler cargado")
        
        # Obtener threshold (intentar desde checkpoint, luego config.json)
        threshold = None
        
        if 'threshold' in checkpoint:
            threshold = checkpoint['threshold']
            st.info("📊 Threshold desde checkpoint")
        else:
            # Intentar leer de config.json
            config_file = model_dir / 'config.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    if 'threshold_config' in config:
                        threshold = config['threshold_config']['optimized_threshold']
                        st.info("📊 Threshold desde config.json")
                    elif 'threshold' in config:
                        threshold = config['threshold']
                        st.info("📊 Threshold desde config.json")
        
        if threshold is None:
            st.error("❌ No se encontró threshold en checkpoint ni config.json")
            st.stop()
        
        st.success(f"✅ Todo cargado | Threshold: {threshold:.6f}")
        
        return model, scaler, threshold
        
    except FileNotFoundError as e:
        st.error(f"❌ Archivo no encontrado: {e}")
        st.error("Estructura esperada:")
        st.code("model/\n├── optimized_autoencoder.pth (o similar .pth)\n├── scaler.joblib\n└── config.json (opcional)")
        st.stop()
        
    except KeyError as e:
        st.error(f"❌ Error en estructura del checkpoint: {e}")
        st.error("El archivo .pth no tiene 'model_state_dict'")
        st.stop()
        
    except Exception as e:
        st.error(f"❌ Error inesperado: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

@st.cache_data
def load_transaction_data():
    """Carga datos de transacciones para simular el flujo"""
    try:
        # Priorizar archivo con variaciones si existe
        if Path('creditcard_realistic.csv').exists():
            st.info("🔄 Cargando datos con variaciones (stream)...")
            df = pd.read_csv('creditcard_realistic.csv')
            st.success(f"✅ Dataset stream cargado: {df.shape[0]:,} transacciones con variaciones")
        elif Path('creditcard.csv').exists():
            st.info("🔄 Cargando datos originales...")
            df = pd.read_csv('creditcard.csv')
            st.success(f"✅ Dataset original cargado: {df.shape[0]:,} transacciones")
        else:
            raise FileNotFoundError("No se encontró creditcard.csv ni creditcard_realistic.csv")
        
        # Validar estructura del dataset
        st.info(f"📋 Columnas en dataset: {len(df.columns)}")
        
        # Verificar columnas esenciales
        if 'Class' not in df.columns:
            st.warning("⚠️ No se encontró columna 'Class', se asumirá que todas son normales")
            df['Class'] = 0
        
        # Contar features disponibles (sin Class)
        feature_cols = [col for col in df.columns if col != 'Class']
        
        # Verificar que tengamos al menos las 30 primeras columnas necesarias
        if len(feature_cols) < 30:
            st.error(f"❌ El dataset tiene solo {len(feature_cols)} features, se necesitan 30")
            st.error(f"Columnas encontradas: {list(df.columns)}")
            st.stop()
        
        if len(feature_cols) > 30:
            st.warning(f"⚠️ Dataset tiene {len(feature_cols)} features, se usarán solo las primeras 30")
        
        return df
        
    except FileNotFoundError:
        st.warning("⚠️ No se encontraron datasets, generando datos sintéticos...")
        
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=10000, n_features=30, n_informative=25,
            n_redundant=5, n_clusters_per_class=1, weights=[0.999, 0.001],
            flip_y=0.01, random_state=42
        )
        feature_names = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        df = pd.DataFrame(X, columns=feature_names)
        df['Class'] = y
        
        st.success(f"✅ Dataset sintético generado: {df.shape[0]:,} transacciones")
        return df
        
    except Exception as e:
        st.error(f"❌ Error cargando datos: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

def predict_transaction(model, scaler, threshold, transaction_data):
    """Predice si una transacción es fraude"""

    # Si transaction_data es una Serie de pandas, convertir a numpy
    if isinstance(transaction_data, pd.Series):
        transaction_data = transaction_data.values

    # Asegurar que solo tenemos las 30 features correctas
    # El modelo espera: V1-V28 (28) + Amount (1) + Time (1) = 30 features
    # Pero el orden en el dataset es: Time, V1-V28, Amount, Class
    # Necesitamos extraer solo: Time, V1-V28, Amount (primeras 30 columnas)

    if len(transaction_data) > 30:
        # Si tiene más de 30, tomar solo las primeras 30 (excluye Class)
        transaction_data = transaction_data[:30]

    # Escalar datos
    transaction_scaled = scaler.transform(transaction_data.reshape(1, -1))

    # Convertir a tensor
    transaction_tensor = torch.FloatTensor(transaction_scaled)

    # Predicción
    with torch.no_grad():
        reconstructed = model(transaction_tensor)
        mse_error = torch.mean((transaction_tensor - reconstructed) ** 2).item()

    # Clasificación
    is_fraud = mse_error > threshold
    anomaly_score = mse_error / threshold

    return is_fraud, mse_error, anomaly_score

def initialize_charts(threshold):
    """Inicializa las figuras de Plotly una sola vez (sin parpadeo)"""
    if st.session_state.fig_mse is None:
        # Crear figura MSE
        fig_mse = go.Figure()
        fig_mse.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='lines+markers',
            name='MSE Error',
            line=dict(color='#6366f1', width=3, shape='spline'),
            marker=dict(
                size=10,
                color=[],
                line=dict(color='#1e293b', width=2),
                opacity=0.9
            ),
            hovertemplate='<b>Transaction #%{x}</b><br>MSE Error: %{y:.6f}<br><extra></extra>'
        ))

        # Configurar layout
        fig_mse.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Transaction Sequence",
            yaxis_title="MSE Reconstruction Error",
            yaxis_type="log",
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='#f1f5f9', family='Inter, sans-serif'),
            xaxis=dict(
                gridcolor='rgba(99, 102, 241, 0.1)',
                zerolinecolor='rgba(99, 102, 241, 0.2)',
            ),
            yaxis=dict(
                gridcolor='rgba(99, 102, 241, 0.1)',
                zerolinecolor='rgba(99, 102, 241, 0.2)',
            ),
            hovermode='x unified',
            margin=dict(l=60, r=40, t=40, b=60)
        )

        # Agregar threshold y zonas
        fig_mse.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#ef4444",
            line_width=2,
            annotation_text="Fraud Threshold",
            annotation_position="right",
            annotation=dict(
                font=dict(size=12, color="#ef4444"),
                bgcolor="rgba(239, 68, 68, 0.1)",
                bordercolor="#ef4444",
                borderwidth=1
            )
        )

        st.session_state.fig_mse = fig_mse

    if st.session_state.fig_score is None:
        # Crear figura Anomaly Score
        fig_score = go.Figure()
        fig_score.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='lines+markers',
            name='Anomaly Score',
            line=dict(color='#8b5cf6', width=3, shape='spline'),
            marker=dict(
                size=10,
                color=[],
                line=dict(color='#1e293b', width=2),
                opacity=0.9
            ),
            fill='tozeroy',
            fillcolor='rgba(139, 92, 246, 0.2)',
            hovertemplate='<b>Transaction #%{x}</b><br>Anomaly Score: %{y:.2f}<br><extra></extra>'
        ))

        fig_score.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Transaction Sequence",
            yaxis_title="Anomaly Score (Normalized)",
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='#f1f5f9', family='Inter, sans-serif'),
            xaxis=dict(
                gridcolor='rgba(139, 92, 246, 0.1)',
                zerolinecolor='rgba(139, 92, 246, 0.2)',
            ),
            yaxis=dict(
                gridcolor='rgba(139, 92, 246, 0.1)',
                zerolinecolor='rgba(139, 92, 246, 0.2)',
            ),
            hovermode='x unified',
            margin=dict(l=60, r=40, t=40, b=60)
        )

        # Threshold line
        fig_score.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="#ef4444",
            line_width=2,
            annotation_text="Anomaly Threshold (1.0)",
            annotation_position="right",
            annotation=dict(
                font=dict(size=12, color="#ef4444"),
                bgcolor="rgba(239, 68, 68, 0.1)",
                bordercolor="#ef4444",
                borderwidth=1
            )
        )

        st.session_state.fig_score = fig_score

    st.session_state.initialized_charts = True

def update_chart_data(history_list, threshold):
    """Actualiza SOLO los datos de las figuras (sin recrear, sin parpadeo)"""
    if not history_list:
        return

    # Extraer datos
    x_data = list(range(len(history_list)))
    y_mse = [t['error'] for t in history_list]
    y_score = [t['score'] for t in history_list]
    colors = ['#ef4444' if t['predicted'] == 'FRAUDE' else '#10b981' for t in history_list]

    # Actualizar figura MSE (solo datos)
    st.session_state.fig_mse.data[0].x = x_data
    st.session_state.fig_mse.data[0].y = y_mse
    st.session_state.fig_mse.data[0].marker.color = colors

    # Actualizar zonas dinámicas si es necesario
    if len(history_list) > 0:
        max_error = max(y_mse) * 1.1
        # Actualizar shapes para zonas
        st.session_state.fig_mse.update_shapes(
            dict(type="rect", y0=0, y1=threshold, fillcolor="rgba(16, 185, 129, 0.1)", layer="below", line_width=0),
            selector=0
        )

    # Actualizar figura Score (solo datos)
    st.session_state.fig_score.data[0].x = x_data
    st.session_state.fig_score.data[0].y = y_score
    st.session_state.fig_score.data[0].marker.color = colors

def update_metrics_display(placeholder1, placeholder2, placeholder3, placeholder4, total_placeholder, threshold):
    """Actualiza SOLO el contenido de las métricas (sin recrear placeholders)"""
    # Calcular métricas
    normal_pct = st.session_state.normal_count / max(st.session_state.total_processed, 1) * 100
    fraud_pct = st.session_state.fraud_count / max(st.session_state.total_processed, 1) * 100
    avg_error = np.mean([t['error'] for t in st.session_state.transactions_history]) if st.session_state.transactions_history else 0
    avg_score = np.mean([t['score'] for t in st.session_state.transactions_history]) if st.session_state.transactions_history else 0

    # Actualizar métrica 1 - Normal Transactions
    placeholder1.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.2) 100%);
                border-radius: 16px;
                padding: 1.5rem;
                border: 1px solid rgba(16, 185, 129, 0.2);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <div style="font-size: 1.5rem; margin-right: 0.75rem;">✅</div>
            <h3 style="margin: 0; color: #f1f5f9; font-size: 0.9rem; font-weight: 500;">Normal Transactions</h3>
        </div>
        <p style="margin: 0; color: #10b981; font-size: 2rem; font-weight: 700;">
            {st.session_state.normal_count:,}
        </p>
        <p style="margin: 0.5rem 0 0 0; color: #86efac; font-size: 0.85rem;">
            {normal_pct:.1f}% of total
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Actualizar métrica 2 - Fraud Detected
    placeholder2.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.2) 100%);
                border-radius: 16px;
                padding: 1.5rem;
                border: 1px solid rgba(239, 68, 68, 0.2);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <div style="font-size: 1.5rem; margin-right: 0.75rem;">⚠️</div>
            <h3 style="margin: 0; color: #f1f5f9; font-size: 0.9rem; font-weight: 500;">Fraud Detected</h3>
        </div>
        <p style="margin: 0; color: #ef4444; font-size: 2rem; font-weight: 700;">
            {st.session_state.fraud_count:,}
        </p>
        <p style="margin: 0.5rem 0 0 0; color: #fca5a5; font-size: 0.85rem;">
            {fraud_pct:.1f}% of total
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Actualizar métrica 3 - Avg MSE Error
    placeholder3.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(79, 70, 229, 0.2) 100%);
                border-radius: 16px;
                padding: 1.5rem;
                border: 1px solid rgba(99, 102, 241, 0.2);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <div style="font-size: 1.5rem; margin-right: 0.75rem;">📈</div>
            <h3 style="margin: 0; color: #f1f5f9; font-size: 0.9rem; font-weight: 500;">Avg. MSE Error</h3>
        </div>
        <p style="margin: 0; color: #6366f1; font-size: 2rem; font-weight: 700;">
            {avg_error:.6f}
        </p>
        <p style="margin: 0.5rem 0 0 0; color: #a5b4fc; font-size: 0.85rem;">
            Threshold: {threshold:.6f}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Actualizar métrica 4 - Anomaly Score
    placeholder4.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(124, 58, 237, 0.2) 100%);
                border-radius: 16px;
                padding: 1.5rem;
                border: 1px solid rgba(139, 92, 246, 0.2);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <div style="font-size: 1.5rem; margin-right: 0.75rem;">⚡</div>
            <h3 style="margin: 0; color: #f1f5f9; font-size: 0.9rem; font-weight: 500;">Anomaly Score</h3>
        </div>
        <p style="margin: 0; color: #8b5cf6; font-size: 2rem; font-weight: 700;">
            {avg_score:.2f}
        </p>
        <p style="margin: 0.5rem 0 0 0; color: #c4b5fd; font-size: 0.85rem;">
            Normalized average
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Actualizar total procesadas
    total_placeholder.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
                border-radius: 12px;
                padding: 1rem 2rem;
                text-align: center;
                border: 2px solid rgba(99, 102, 241, 0.3);
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                margin: 1rem 0;">
        <span style="color: #94a3b8; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1px;">
            Total Transactions Processed:
        </span>
        <span style="color: #6366f1; font-size: 2rem; font-weight: 700; margin-left: 1rem;">
            {st.session_state.total_processed:,}
        </span>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# INICIALIZACIÓN DE ESTADO
# ============================================================================

if 'transactions_history' not in st.session_state:
    st.session_state.transactions_history = deque(maxlen=100)
    st.session_state.fraud_count = 0
    st.session_state.normal_count = 0
    st.session_state.total_processed = 0
    st.session_state.streaming = False
    st.session_state.current_idx = 0
    st.session_state.fig_mse = None
    st.session_state.fig_score = None
    st.session_state.initialized_charts = False

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

# Header con logo y título
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.markdown("# 🛡️")
with col_title:
    st.title("Fraud Detection System")
    st.markdown("### Real-Time Transaction Monitoring & Analysis")

st.markdown("---")

# Cargar modelo y datos con debugging
try:
    with st.spinner("🔄 Initializing AI model and loading data..."):
        autoencoder, scaler, threshold = load_model_and_scaler()
        df_transactions = load_transaction_data()

        # Definir feature_cols (todas las columnas excepto Class)
        feature_cols = [col for col in df_transactions.columns if col != 'Class'][:30]

    # Información del sistema en un banner elegante
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
                border-radius: 12px;
                padding: 1rem;
                border-left: 4px solid #6366f1;
                margin-bottom: 2rem;">
        <p style="margin: 0; color: #94a3b8; font-size: 0.9rem;">
            <strong style="color: #f1f5f9;">🤖 System Status:</strong> Online |
            <strong style="color: #f1f5f9;">🎯 Threshold:</strong> {threshold:.6f} |
            <strong style="color: #f1f5f9;">📊 Dataset:</strong> {len(df_transactions):,} transactions loaded
        </p>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"❌ Critical Error: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

# ============================================================================
# SIDEBAR - CONTROLES Y CONFIGURACIÓN
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.markdown("---")

    # Controles principales
    st.markdown("### 🎮 Stream Controls")

    # Botón de inicio/pausa
    btn_text = "▶️ Start Stream" if not st.session_state.streaming else "⏸️ Pause Stream"
    btn_type = "primary" if not st.session_state.streaming else "secondary"

    if st.button(btn_text, use_container_width=True, type=btn_type):
        st.session_state.streaming = not st.session_state.streaming
        st.rerun()

    # Botón de reset
    if st.button("🔄 Reset All Data", use_container_width=True):
        st.session_state.transactions_history.clear()
        st.session_state.fraud_count = 0
        st.session_state.normal_count = 0
        st.session_state.total_processed = 0
        st.session_state.current_idx = 0
        st.rerun()

    st.markdown("---")

    # Configuración de velocidad
    st.markdown("### ⚡ Speed Configuration")
    speed = st.slider(
        "Transactions per second",
        min_value=1,
        max_value=20,
        value=5,
        help="Control the speed of transaction processing"
    )
    delay = 1.0 / speed

    st.markdown(f"""
    <div style="background: rgba(99, 102, 241, 0.1);
                border-radius: 8px;
                padding: 0.5rem;
                text-align: center;
                margin-top: 0.5rem;">
        <span style="color: #94a3b8; font-size: 0.85rem;">
            Processing: <strong style="color: #6366f1;">{speed}</strong> trans/sec
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Información del modelo
    st.markdown("### 🤖 Model Information")
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.5);
                border-radius: 8px;
                padding: 1rem;
                font-size: 0.85rem;">
        <p style="margin: 0.3rem 0; color: #94a3b8;">
            <strong style="color: #f1f5f9;">Architecture:</strong> Autoencoder
        </p>
        <p style="margin: 0.3rem 0; color: #94a3b8;">
            <strong style="color: #f1f5f9;">Framework:</strong> PyTorch
        </p>
        <p style="margin: 0.3rem 0; color: #94a3b8;">
            <strong style="color: #f1f5f9;">Input Features:</strong> 30
        </p>
        <p style="margin: 0.3rem 0; color: #94a3b8;">
            <strong style="color: #f1f5f9;">Threshold:</strong> {threshold:.6f}
        </p>
        <p style="margin: 0.3rem 0; color: #94a3b8;">
            <strong style="color: #f1f5f9;">ROC-AUC:</strong> 95.98%
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Estado del sistema
    st.markdown("### 📊 System Status")
    status_color = "#10b981" if st.session_state.streaming else "#6b7280"
    status_text = "Streaming" if st.session_state.streaming else "Paused"

    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.5);
                border-radius: 8px;
                padding: 1rem;
                text-align: center;">
        <div style="display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    background-color: {status_color};
                    margin-right: 8px;
                    {'animation: pulse 2s ease-in-out infinite;' if st.session_state.streaming else ''}">
        </div>
        <span style="color: #f1f5f9; font-weight: 600;">{status_text}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# MÉTRICAS EN TIEMPO REAL - DISEÑO MEJORADO
# ============================================================================

st.markdown("## 📊 Real-Time Metrics Dashboard")

# Crear columnas para métricas
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

# Crear placeholders persistentes para evitar duplicación
metric_placeholder1 = metric_col1.empty()
metric_placeholder2 = metric_col2.empty()
metric_placeholder3 = metric_col3.empty()
metric_placeholder4 = metric_col4.empty()

st.markdown("<br>", unsafe_allow_html=True)

# Placeholder para métrica de total procesadas
total_processed_placeholder = st.empty()

st.markdown("---")

# ============================================================================
# VISUALIZACIÓN EN TIEMPO REAL - DISEÑO MEJORADO
# ============================================================================

st.markdown("## 📈 Live Analytics & Monitoring")

# Crear contenedores persistentes para evitar parpadeo
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.5);
                border-radius: 12px 12px 0 0;
                padding: 1rem;
                border-left: 4px solid #6366f1;">
        <h3 style="margin: 0; color: #f1f5f9; font-size: 1.2rem;">
            📊 Reconstruction Error (MSE)
        </h3>
        <p style="margin: 0.25rem 0 0 0; color: #94a3b8; font-size: 0.85rem;">
            Real-time error tracking with threshold line
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_chart2:
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.5);
                border-radius: 12px 12px 0 0;
                padding: 1rem;
                border-left: 4px solid #8b5cf6;">
        <h3 style="margin: 0; color: #f1f5f9; font-size: 1.2rem;">
            🎯 Anomaly Score Distribution
        </h3>
        <p style="margin: 0.25rem 0 0 0; color: #94a3b8; font-size: 0.85rem;">
            Normalized anomaly detection scores
        </p>
    </div>
    """, unsafe_allow_html=True)

# Placeholders para los gráficos (contenedores persistentes)
chart_mse_placeholder = col_chart1.empty()
chart_score_placeholder = col_chart2.empty()

# ============================================================================
# TABLA DE ÚLTIMAS TRANSACCIONES
# ============================================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="background: rgba(30, 41, 59, 0.5);
            border-radius: 12px 12px 0 0;
            padding: 1rem;
            border-left: 4px solid #f59e0b;
            margin-top: 2rem;">
    <h3 style="margin: 0; color: #f1f5f9; font-size: 1.2rem;">
        🕐 Recent Transaction History
    </h3>
    <p style="margin: 0.25rem 0 0 0; color: #94a3b8; font-size: 0.85rem;">
        Last 20 transactions with prediction results and accuracy indicators
    </p>
</div>
""", unsafe_allow_html=True)
table_placeholder = st.empty()

# Inicializar figuras una sola vez
initialize_charts(threshold)

# ============================================================================
# STREAMING DE TRANSACCIONES (NUEVO ENFOQUE SIN PARPADEO)
# ============================================================================

if st.session_state.streaming:
    # LOOP CONTINUO SIN st.rerun() - Actualiza solo los placeholders
    while st.session_state.streaming and st.session_state.current_idx < len(df_transactions):
        # Obtener transacción actual
        row = df_transactions.iloc[st.session_state.current_idx]

        # Preparar features para el modelo
        features = row[feature_cols].values.reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled)

        # Predecir con el modelo
        with torch.no_grad():
            reconstructed = autoencoder(features_tensor)
            mse_error = torch.mean((features_tensor - reconstructed) ** 2, dim=1).item()

        # Determinar si es fraude
        is_fraud = mse_error > threshold
        predicted_class = 'FRAUDE' if is_fraud else 'NORMAL'
        actual_class = 'FRAUDE' if row['Class'] == 1 else 'NORMAL'

        # Calcular anomaly score normalizado
        anomaly_score = (mse_error / threshold) if threshold > 0 else 0

        # Actualizar contadores
        if is_fraud:
            st.session_state.fraud_count += 1
        else:
            st.session_state.normal_count += 1

        st.session_state.total_processed += 1

        # Añadir a historial
        transaction_info = {
            'id': st.session_state.current_idx,
            'error': mse_error,
            'score': anomaly_score,
            'predicted': predicted_class,
            'actual': actual_class,
            'match': '✓' if predicted_class == actual_class else '✗'
        }
        st.session_state.transactions_history.append(transaction_info)

        # Avanzar al siguiente índice
        st.session_state.current_idx += 1

        # Obtener historial para actualizar visualizaciones
        history_list = list(st.session_state.transactions_history)

        # ACTUALIZAR MÉTRICAS SIN RECREAR (SIN PARPADEO)
        update_metrics_display(metric_placeholder1, metric_placeholder2, metric_placeholder3, metric_placeholder4, total_processed_placeholder, threshold)

        # ACTUALIZAR GRÁFICOS SIN RECREAR (SIN PARPADEO)
        if history_list:
            # Actualizar solo los datos de las figuras existentes
            update_chart_data(history_list, threshold)

            # Renderizar las figuras desde session_state (ya actualizadas)
            chart_mse_placeholder.plotly_chart(st.session_state.fig_mse, use_container_width=True)
            chart_score_placeholder.plotly_chart(st.session_state.fig_score, use_container_width=True)

            # Tabla de últimas transacciones con diseño mejorado
            recent_df = pd.DataFrame(history_list[-20:])

            # Aplicar estilos profesionales
            def highlight_fraud(row):
                if row['predicted'] == 'FRAUDE':
                    return ['background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%); color: #fca5a5; font-weight: 500;'] * len(row)
                else:
                    return ['background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%); color: #86efac; font-weight: 500;'] * len(row)

            styled_df = recent_df.style.apply(highlight_fraud, axis=1)

            table_placeholder.dataframe(
                styled_df,
                use_container_width=True,
                height=400
            )

        # Esperar antes de la siguiente transacción
        time.sleep(delay)

    # Si terminamos el dataset, detener streaming
    if st.session_state.current_idx >= len(df_transactions):
        st.session_state.streaming = False

else:
    # Mostrar última información si está pausado (SIN PARPADEO)
    if st.session_state.transactions_history:
        history_list = list(st.session_state.transactions_history)

        # Actualizar métricas (sin recrear)
        update_metrics_display(metric_placeholder1, metric_placeholder2, metric_placeholder3, metric_placeholder4, total_processed_placeholder, threshold)

        # Actualizar datos de las figuras existentes (sin recrear)
        update_chart_data(history_list, threshold)

        # Renderizar las figuras desde session_state
        chart_mse_placeholder.plotly_chart(st.session_state.fig_mse, use_container_width=True)
        chart_score_placeholder.plotly_chart(st.session_state.fig_score, use_container_width=True)

        # Tabla con diseño mejorado
        recent_df = pd.DataFrame(history_list[-20:])

        def highlight_fraud(row):
            if row['predicted'] == 'FRAUDE':
                return ['background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%); color: #fca5a5; font-weight: 500;'] * len(row)
            else:
                return ['background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%); color: #86efac; font-weight: 500;'] * len(row)

        styled_df = recent_df.style.apply(highlight_fraud, axis=1)
        table_placeholder.dataframe(styled_df, use_container_width=True, height=400)
    else:
        # Mensaje de bienvenida mejorado
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
                    border-radius: 16px;
                    padding: 3rem;
                    text-align: center;
                    border: 2px dashed rgba(99, 102, 241, 0.3);
                    margin: 3rem 0;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🚀</div>
            <h2 style="color: #f1f5f9; margin-bottom: 1rem;">Ready to Start Monitoring</h2>
            <p style="color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;">
                Click <strong style="color: #6366f1;">"Start Stream"</strong> in the sidebar to begin real-time fraud detection
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem;">
                <div style="text-align: center;">
                    <div style="font-size: 2rem; color: #10b981;">✓</div>
                    <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">AI Model Loaded</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; color: #10b981;">✓</div>
                    <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">Dataset Ready</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; color: #6366f1;">⏸️</div>
                    <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">Awaiting Start</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# FOOTER PROFESIONAL
# ============================================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%);
            border-radius: 12px;
            padding: 2rem;
            border-top: 2px solid rgba(99, 102, 241, 0.3);
            margin-top: 3rem;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 2rem;">
        <div style="flex: 1; min-width: 250px;">
            <h3 style="color: #f1f5f9; margin: 0 0 0.5rem 0; font-size: 1.3rem;">
                🛡️ Fraud Detection System
            </h3>
            <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">
                Advanced AI-powered real-time transaction monitoring
            </p>
        </div>
        <div style="flex: 1; min-width: 250px; text-align: center;">
            <p style="color: #94a3b8; margin: 0; font-size: 0.85rem;">
                <strong style="color: #6366f1;">Powered by:</strong><br>
                PyTorch • Streamlit • Plotly • Scikit-learn
            </p>
        </div>
        <div style="flex: 1; min-width: 250px; text-align: right;">
            <p style="color: #94a3b8; margin: 0; font-size: 0.85rem;">
                <strong style="color: #8b5cf6;">Model Performance:</strong><br>
                ROC-AUC 95.98% • F1-Score 48.31%
            </p>
        </div>
    </div>
    <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(99, 102, 241, 0.2); text-align: center;">
        <p style="color: #64748b; margin: 0; font-size: 0.8rem;">
            © 2025 Fraud Detection System • Built with ❤️ using Claude Code
        </p>
    </div>
</div>
""", unsafe_allow_html=True)