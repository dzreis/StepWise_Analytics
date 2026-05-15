import streamlit as st
from PIL import Image
from views import visualizacao_obstacles
from views import visualizacao_estatistica  # view da IC — membros superiores

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================
# Carrega o ícone da aba do navegador
icon = Image.open("assets/icon SW.png")

st.set_page_config(
    page_title="StepWise Analytics",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CABEÇALHO
# =============================================================================
st.image("assets/SW w png.png", width=300)

# =============================================================================
# NAVEGAÇÃO POR ABAS
# =============================================================================
abas = st.tabs([
    "🏠 Início",
    "🦵 Membros Inferiores",
    "🦾 Membros Superiores",
    "🤖 Modelo Preditivo",
])

# -----------------------------------------------------------------------
# ABA 1 — Início
# -----------------------------------------------------------------------
with abas[0]:
    st.markdown(
        "Este sistema traduz dados cinemáticos brutos capturados por visão "
        "computacional em indicadores objetivos do **progresso terapêutico**, auxiliando na tomada de decisão clínica."
    )
    st.markdown(
        """
        ### 📋 Como usar

        1. Acesse a aba correspondente ao tipo de análise desejada.
        2. Configure os **parâmetros clínicos** na barra lateral (limiares de ângulo,
           modo de análise).
        3. Faça o upload do(s) arquivo(s) exportados do software
           nos formatos **.CSV**, **.XLSX** ou **.XLSM**.
        4. Os resultados são gerados automaticamente.

        ---

        ### 📌 Abas disponíveis

        | Aba | Descrição |
        |---|---|
        | 🦵 Membros Inferiores | Análise da marcha estacionária: flexão de quadril e joelho, simetria, suavidade e compensação postural |
        | 🦾 Membros Superiores | Análise dos ombros e cotovelos durante o e-Puzzle |
        | 🤖 Modelo Preditivo | Resultados dos modelos de Aprendizado de Máquina (busca de padrões e compensação de movimento) |

        ---

        ### ⚠️ Atenção
        Este dashboard é uma ferramenta de **suporte à decisão** e não substitui
        a avaliação do profissional de saúde.
        """
    )

# -----------------------------------------------------------------------
# ABA 2 — Membros Inferiores (Obstacles) — foco do TCC
# -----------------------------------------------------------------------
with abas[1]:
    visualizacao_obstacles.carregar()

# -----------------------------------------------------------------------
# ABA 3 — Membros Superiores (IC anterior)
# -----------------------------------------------------------------------
with abas[2]:
    visualizacao_estatistica.carregar()

# -----------------------------------------------------------------------
# ABA 4 — Modelo Preditivo (em desenvolvimento)
# -----------------------------------------------------------------------
with abas[3]:
    st.markdown("## 🤖 Modelo Preditivo")
    st.info(
        "Esta seção está em desenvolvimento. "
        "Os modelos LSTM, CNN 1D e SVM serão integrados aqui após o treinamento.",
        icon="🔧",
    )