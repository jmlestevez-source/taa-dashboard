import streamlit as st

st.title("🎯 TAA Dashboard")
st.success("🎉 ¡La aplicación está funcionando!")

st.markdown("""
### Próximos pasos:
1. Esta es una versión de prueba para confirmar que Streamlit funciona
2. Ahora podemos añadir las dependencias una por una
3. Haz clic abajo para comenzar la instalación
""")

if st.button("🚀 Instalar dependencias"):
    st.info("Instalando paquetes...")
    
    import subprocess
    import sys
    
    packages = [
        "yfinance==0.2.41",
        "pandas==2.2.2", 
        "numpy==1.26.4",
        "plotly==5.24.1"
    ]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            st.success(f"✅ {package}")
        except Exception as e:
            st.error(f"❌ {package}: {str(e)}")
    
    st.balloons()
    st.success("¡Instalación completada! Ahora puedes usar la app completa.")
