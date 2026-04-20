import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.processamento import calcular_frames_por_segundo, calcular_tempos_picos, classificar, plot_intervalos_picos

def carregar():
    st.title("📊 Dashboard de Análise de Movimento")
    st.markdown("Envie os arquivos CSV do início e do final da reabilitação para visualizar os gráficos e análises.")

    col1, col2 = st.columns(2)
    with col1:
        inicio_file = st.file_uploader("📁 CSV do Início", type="csv", key="inicio")
    with col2:
        final_file = st.file_uploader("📁 CSV do Final", type="csv", key="final")

    if not (inicio_file and final_file):
        st.info("Envie ambos os arquivos para iniciar a análise.")
        return

    # Leitura dos dados
    inicio_df = pd.read_csv(inicio_file)
    final_df = pd.read_csv(final_file)

    inicio = inicio_df[['shoulderLangle', 'shoulderRangle']]
    final = final_df[['shoulderLangle', 'shoulderRangle']]

    # Cálculo do tempo total
    fps_inicio = calcular_frames_por_segundo(inicio_df, 'time')
    fps_final = calcular_frames_por_segundo(final_df, 'time')

    tempo_total_inicio = np.ceil(len(inicio_df) / fps_inicio)
    tempo_total_final = np.ceil(len(final_df) / fps_final)

    st.success(f"🕒 Tempo no início: **{int(tempo_total_inicio)}s**")
    st.success(f"🕒 Tempo no final: **{int(tempo_total_final)}s**")

    # Normalização do tempo
    tempo_inicio = np.arange(len(inicio_df)) / fps_inicio
    tempo_final = np.arange(len(final_df)) / fps_final

    # === Gráfico 1: Amplitude do movimento (início e final)
    st.markdown("### 📉 Visão geral do movimento (ambos os ombros)")

    fig_amp = go.Figure()
    fig_amp.add_trace(go.Scatter(
        y=inicio['shoulderLangle'],
        mode='lines',
        name='Ombro Esquerdo - Início',
        line=dict(color='blue')
    ))
    fig_amp.add_trace(go.Scatter(
        y=inicio['shoulderRangle'],
        mode='lines',
        name='Ombro Direito - Início',
        line=dict(color='orange')
    ))
    fig_amp.update_layout(
        title="Movimento no Início do Tratamento",
        xaxis_title="Frames",
        yaxis_title="Amplitude (graus)",
        legend=dict(x=0.01, y=0.99),
        height=400
    )
    st.plotly_chart(fig_amp, use_container_width=True)

    # === Escolha do lado para análise detalhada
    st.markdown("### 🦾 Escolha o lado para análise detalhada")

    # inicializa o estado se ainda não existir
    if "lado_escolhido" not in st.session_state:
        st.session_state.lado_escolhido = None

    # menu com placeholder (sem seleção inicial)
    lado_opcoes = ["Esquerdo", "Direito"]
    lado_escolhido = st.selectbox(
        "Selecione o ombro a ser analisado:",
        options=["Selecione..."] + lado_opcoes,
        index=0,
        key="lado_selector"
    )

    # só prossegue se o usuário escolher algo diferente do placeholder
    if lado_escolhido != "Selecione...":
        st.session_state.lado_escolhido = lado_escolhido
        st.success(f"Lado selecionado: **{lado_escolhido}**")

        # Define as colunas com base na escolha
        if lado_escolhido == "Esquerdo":
            coluna_inicio = 'shoulderLangle'
            coluna_final = 'shoulderLangle'
            cor_lado = 'blue'
        else:
            coluna_inicio = 'shoulderRangle'
            coluna_final = 'shoulderRangle'
            cor_lado = 'orange'

        # === Gráfico 2: Amplitude normalizada no tempo
        fig_norm = go.Figure()
        fig_norm.add_trace(go.Scatter(
            x=tempo_inicio,
            y=inicio[coluna_inicio],
            mode='lines',
            name=f'Ombro {lado_escolhido} - Início',
            line=dict(color=cor_lado)
        ))
        fig_norm.add_trace(go.Scatter(
            x=tempo_final,
            y=final[coluna_final],
            mode='lines',
            name=f'Ombro {lado_escolhido} - Final',
            line=dict(color='red')
        ))
        fig_norm.update_layout(
            title=f"Amplitude do Ombro {lado_escolhido} Normalizado",
            xaxis_title="Tempo (s)",
            yaxis_title="Amplitude (graus)",
            legend=dict(x=0.01, y=0.99),
            height=400
        )
        st.plotly_chart(fig_norm, use_container_width=True)

        # === Estatísticas
        st.markdown("### 📊 Estatísticas de Amplitude")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Média - Início", round(inicio[coluna_inicio].mean(), 2))
            st.metric("Mediana - Início", round(inicio[coluna_inicio].median(), 2))
        with col2:
            st.metric("Média - Final", round(final[coluna_final].mean(), 2))
            st.metric("Mediana - Final", round(final[coluna_final].median(), 2))

        # === Limiar definido pelo usuário
        st.markdown("### ⚙️ Definir Limiar para Análise de Picos")
        col1, col2 = st.columns(2)
        with col1:
            limiar_i = st.number_input("Limiar Início (graus)", min_value=0.0, max_value=180.0, value=35.0, step=1.0)
        with col2:
            limiar_f = st.number_input("Limiar Final (graus)", min_value=0.0, max_value=180.0, value=45.0, step=1.0)

        # === Picos e classificação
        st.markdown("### 📌 Análise e Classificação do Movimento Esperado")

        dados_i = inicio[coluna_inicio]
        dados_f = final[coluna_final]

        # Cálculo dos picos
        picos_i, duracoes_i, media_i = calcular_tempos_picos(dados_i, fps_inicio, limiar_i, 5)
        picos_f, duracoes_f, media_f = calcular_tempos_picos(dados_f, fps_final, limiar_f, 2)

        # Classificação com base nas durações
        classificacoes_i = classificar(duracoes_i)
        classificacoes_f = classificar(duracoes_f)

        # Gráficos interativos dos picos
        st.subheader("🔎 Identificação do Movimento Esperado")
        tempo_inicio = np.arange(len(dados_i)) / fps_inicio
        tempo_final = np.arange(len(dados_f)) / fps_final

        fig_i = plot_intervalos_picos(tempo_inicio, dados_i, limiar_i, picos_i, "Movimento Esperado - Início")
        fig_f = plot_intervalos_picos(tempo_final, dados_f, limiar_f, picos_f, "Movimento Esperado - Final")

        st.plotly_chart(fig_i, use_container_width=True)
        st.plotly_chart(fig_f, use_container_width=True)

        # Estatísticas
        st.write(f"⏱️ Média dos movimentos iniciais: **{media_i:.2f}s**")
        st.write(f"⏱️ Média dos movimentos finais: **{media_f:.2f}s**")

        if classificacoes_i:
            st.write("**Classificação Início:**", ', '.join(classificacoes_i))
        else:
            st.write("**Classificação Início:** Nenhum pico identificado")

        if classificacoes_f:
            st.write("**Classificação Final:**", ', '.join(classificacoes_f))
        else:
            st.write("**Classificação Final:** Nenhum pico identificado")

        # === Gráfico de Dispersão dos Picos
        st.markdown("### 🔍 Dispersão - Nível dos Movimentos Iniciais e Finais")

        fig_picos = go.Figure()
        fig_picos.add_trace(go.Scatter(
            x=list(range(len(duracoes_i))),
            y=duracoes_i,
            mode='markers',
            name='Movimentos Iniciais',
            marker=dict(color='blue', size=10)
        ))
        fig_picos.add_trace(go.Scatter(
            x=list(range(len(duracoes_f))),
            y=duracoes_f,
            mode='markers',
            name='Movimentos Finais',
            marker=dict(color='red', size=10)
        ))

        fig_picos.add_shape(type="line", x0=0, x1=max(len(duracoes_i), len(duracoes_f)),
                            y0=5, y1=5, line=dict(color="green", dash="dash"))
        fig_picos.add_shape(type="line", x0=0, x1=max(len(duracoes_i), len(duracoes_f)),
                            y0=9, y1=9, line=dict(color="purple", dash="dash"))

        fig_picos.update_layout(
            title="Dispersão dos Movimentos Iniciais e Finais",
            xaxis_title="Índice",
            yaxis_title="Duração (s)",
            legend=dict(x=0.01, y=0.99),
            height=400
        )
        st.plotly_chart(fig_picos, use_container_width=True)
    
    else:
        st.warning("Selecione o lado para continuar a análise detalhada.")
