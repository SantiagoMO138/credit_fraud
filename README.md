# 🛡️ Fraud Detection System - Real-Time Monitoring

Sistema de detección de fraude en tiempo real usando PyTorch Autoencoder con interfaz web profesional.

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 🎨 Características Principales

### Diseño Profesional
- **Tema oscuro moderno** con gradientes y animaciones suaves
- **Dashboard interactivo** con métricas en tiempo real
- **Visualizaciones avanzadas** usando Plotly con zonas de riesgo coloreadas
- **Interfaz responsiva** optimizada para presentaciones

### Capacidades Técnicas
- ✅ Detección de fraude con **95.98% ROC-AUC**
- ⚡ Procesamiento en tiempo real (1-20 transacciones/seg)
- 📊 Gráficos dinámicos sin parpadeo
- 🎯 Threshold optimizado (99.9 percentil)
- 📈 Historial de últimas 100 transacciones
- 🔄 Streaming continuo con control de velocidad

---

## 📁 Estructura del Proyecto

```
credit_fraud/
├── app.py                      # Aplicación Streamlit principal
├── genered_data.py            # Generador de datos sintéticos
├── model/                     # Modelo entrenado
│   ├── optimized_autoencoder.pth
│   ├── scaler.joblib
│   └── config.json
├── creditcard.csv             # Dataset original
├── creditcard_realistic.csv   # Dataset con variaciones
└── requirements.txt           # Dependencias
```

---

## 🚀 Instalación y Uso

### 1. Requisitos Previos
```bash
Python 3.8 o superior
pip (gestor de paquetes)
```

### 2. Activar Entorno Virtual
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Instalar Dependencias (si es necesario)
```bash
pip install -r requirements.txt
```

### 4. Ejecutar la Aplicación
```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

---

## 🎮 Guía de Uso

### Panel de Control (Sidebar)

**🎮 Stream Controls**
- **Start Stream**: Inicia el monitoreo en tiempo real
- **Pause Stream**: Pausa el procesamiento
- **Reset All Data**: Limpia todo el historial

**⚡ Speed Configuration**
- Ajusta la velocidad de procesamiento: 1-20 transacciones/segundo

**🤖 Model Information**
- Arquitectura: Autoencoder
- Framework: PyTorch
- Input Features: 30
- ROC-AUC: 95.98%

**📊 System Status**
- Indicador visual del estado (Streaming/Paused)

### Dashboard Principal

**📊 Real-Time Metrics**
- ✅ **Normal Transactions**: Contador de transacciones legítimas
- ⚠️ **Fraud Detected**: Contador de fraudes detectados
- 📈 **Avg. MSE Error**: Error promedio de reconstrucción
- ⚡ **Anomaly Score**: Puntuación de anomalía promedio

**📈 Live Analytics**
- **Reconstruction Error (MSE)**: Gráfico logarítmico con threshold
- **Anomaly Score Distribution**: Distribución normalizada de scores

**🕐 Recent Transaction History**
- Tabla con las últimas 20 transacciones
- Código de colores: 🟢 Normal / 🔴 Fraude
- Información de precisión de predicciones

---

## 🛠️ Tecnologías Utilizadas

| Tecnología | Versión | Propósito |
|-----------|---------|-----------|
| Python | 3.12+ | Lenguaje base |
| PyTorch | 2.0+ | Deep Learning |
| Streamlit | 1.51+ | Interfaz web |
| Plotly | 5.14+ | Visualizaciones |
| Pandas | 1.5+ | Manipulación de datos |
| Scikit-learn | 1.2+ | Preprocesamiento |

---

## 📊 Rendimiento del Modelo

### Métricas de Evaluación
```
ROC-AUC:     95.98%
F1-Score:    48.31%
Precision:   45.87%
Recall:      51.02%
Accuracy:    99.81%
```

### Impacto de Negocio
- **Alertas diarias**: 29 (vs 1,500+ sin optimización)
- **Tasa de detección**: 51% de todos los fraudes
- **Precisión de alertas**: 45.9% son fraudes reales
- **Reducción de falsos positivos**: 98%

---

## 🎨 Personalización

### Cambiar Velocidad de Procesamiento
Usa el slider en el sidebar para ajustar de 1 a 20 transacciones por segundo.

### Tema de Colores
Los colores se definen en las variables CSS al inicio de `app.py`:
```css
--primary-color: #6366f1;    /* Indigo */
--secondary-color: #8b5cf6;  /* Purple */
--success-color: #10b981;    /* Green */
--danger-color: #ef4444;     /* Red */
```

### Configuración del Threshold
El threshold se carga automáticamente desde `model/config.json`.

---

## 🔧 Solución de Problemas

### La aplicación no inicia
```bash
# Verificar instalación de Streamlit
streamlit --version

# Reinstalar dependencias
pip install -r requirements.txt --force-reinstall
```

### Errores de modelo
```bash
# Verificar que existan los archivos del modelo
ls -la model/
```

### Gráficos no se muestran
- Verifica que Plotly esté instalado: `pip install plotly`
- Limpia la caché del navegador (Ctrl + Shift + Delete)

---

## 📝 Generación de Datos Sintéticos

El proyecto incluye `genered_data.py` con **6 métodos** de generación:

```bash
# Método 1: Ruido correlacionado (5% variación)
python genered_data.py --method 1 --percent 5.0

# Método 2: Interpolación (10,000 muestras)
python genered_data.py --method 2 --samples 10000

# Método 3: Neighbor sampling
python genered_data.py --method 3

# Método 4: Temporal drift
python genered_data.py --method 4

# Método 5: PCA perturbation
python genered_data.py --method 5

# Método 6: Conditional noise
python genered_data.py --method 6
```

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

---

## 👨‍💻 Autor

**Desarrollado con ❤️ usando Claude Code**

- Sistema de IA: Autoencoder PyTorch
- Interfaz: Streamlit + Plotly
- Diseño: Tema oscuro profesional con gradientes

---

## 📞 Soporte

Si encuentras algún problema o tienes sugerencias:
- Abre un Issue en GitHub
- Revisa la documentación de [Streamlit](https://docs.streamlit.io)
- Consulta la documentación de [PyTorch](https://pytorch.org/docs)

---

## 🎯 Roadmap

- [ ] Exportar reportes en PDF
- [ ] Integración con bases de datos en tiempo real
- [ ] Sistema de notificaciones por email/SMS
- [ ] Dashboard de administración
- [ ] API REST para integración
- [ ] Soporte multi-idioma
- [ ] Modo claro/oscuro toggle

---

**¡Gracias por usar el Fraud Detection System!** 🛡️
