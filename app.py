import streamlit as st

st.title("TAA Dashboard")
st.info("🚀 Aplicación cargando...")

try:
    import yfinance as yf
    st.success("✅ yfinance importado correctamente")
    
    # Test simple
    data = yf.download("SPY", period="1mo")
    st.write("Datos de SPY descargados:", data.shape)
    
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.code("""
    Por favor, verifica que requirements.txt contiene:
    streamlit==1.38.0
    yfinance==0.2.41
    pandas==2.2.2
    numpy==1.26.4
    plotly==5.24.1
    """)
