import os
import logging
import pandas as pd
import streamlit as st

from views import visualizacao_estatistica
from views import ml_teste  # importa o pipeline completo com PCA + DBSCAN

st.set_page_config(page_title="Dashboard Análise de Interações", layout="wide")

# Configuração do logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ----------- INTERFACE PRINCIPAL ------------------

st.title("🧠 Análise de Interações - Reabilitação Motora")

# -------- Navegação por abas --------
abas = st.tabs(["🏠 Início", "📊 Visualização Estatística", "🤖 Modelo Preditivo"])

# -------- Página 1: Instruções --------
with abas[0]:
    st.title("✨Reabilitação assistida por AR: visualização e análise dos dados")
    st.markdown("""
    Este dashboard tem como objetivo auxiliar na análise de dados obtidos a partir de interações com softwares de reabilitação.
                
    ### 📁 Upload de Arquivo
    - O arquivo deve estar no formato **.CSV**.
    - Faça o envio utilizando a barra lateral à esquerda.
                
    ### ⚙️ Parâmetros
    - **Fonte dos dados**: Tipo de câmera utilizada (Infravermelho ou RGB).

    ### 📊 Visualização Estatística
    - Página destinada a apresentar análises exploratórias iniciais dos dados enviados.

    ### 🤖 Modelo Preditivo
    - Página dedicada à apresentação de resultados gerados pelo modelo de aprendizado de máquina não supervisionado.
    """)

# -------- Página 2: Visualização Estatística --------
with abas[1]:
    visualizacao_estatistica.carregar()

