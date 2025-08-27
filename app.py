import streamlit as st

st.title("🔧 TAA Dashboard - Instalador")
st.info("🚀 Instalando dependencias...")

# Función para instalar paquetes
def install_package(package):
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except Exception as e:
        st.error(f"Error instalando {package}: {str(e)}")
        return False

# Lista de paquetes necesarios
packages = [
    "yfinance==0.2.41",
    "pandas==2.2.2", 
    "numpy==1.26.4",
    "plotly==5.24.1"
]

# Instalar paquetes uno por uno
for package in packages:
    with st.spinner(f"Instalando {package}..."):
        if install_package(package):
            st.success(f"✅ {package} instalado correctamente")
        else:
            st.error(f"❌ Error instalando {package}")

st.divider()
st.success("🎉 ¡Instalación completada!")
st.info("Ahora puedes reemplazar este código con tu aplicación real")

# Test final
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    st.success("✅ Todas las dependencias funcionan correctamente")
    st.balloons()
except Exception as e:
    st.error(f"❌ Aún hay errores: {str(e)}")
