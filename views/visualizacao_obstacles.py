import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.processamento import (
    carregar_arquivo,
    calcular_fps,
    gerar_resumo_sessao,
    picos_para_dataframe,
    COL_QUADRIL_ESQ, COL_QUADRIL_DIR,
    COL_JOELHO_ESQ,  COL_JOELHO_DIR,
    COL_TORNOZELO_ESQ, COL_TORNOZELO_DIR,
)

# =============================================================================
# CONSTANTE DE REFERÊNCIA CLÍNICA
# Abaixo desse limiar de simetria, considera-se assimetria clinicamente
# relevante segundo a literatura de biomecânica.
# =============================================================================
LIMIAR_SIMETRIA = 0.85


# =============================================================================
# HELPERS DE VISUALIZAÇÃO
# =============================================================================

def _tempo_eixo(df: pd.DataFrame, fps: float) -> np.ndarray:
    """Converte frames em segundos para o eixo X dos gráficos."""
    return np.arange(len(df)) / fps


def _cor_simetria(valor: float) -> str:
    """Retorna a cor do indicador de simetria conforme limiar clínico."""
    return "normal" if valor >= LIMIAR_SIMETRIA else "inverse"


def _plot_sinal_com_picos(
    tempo: np.ndarray,
    sinal_esq: pd.Series,
    sinal_dir: pd.Series,
    picos_esq: list[dict],
    picos_dir: list[dict],
    limiar: float,
    titulo: str,
    unidade: str = "graus",
) -> go.Figure:
    """
    Gera gráfico do sinal cinemático com o limiar clínico e os picos
    marcados para os lados esquerdo e direito.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Esquerdo", "Direito"),
        vertical_spacing=0.12,
    )

    # --- Lado esquerdo ---
    fig.add_trace(go.Scatter(
        x=tempo, y=sinal_esq,
        mode='lines', name='Esquerdo',
        line=dict(color='#3A86FF', width=1.5),
    ), row=1, col=1)

    # Linha do limiar
    fig.add_hline(
        y=limiar, line_dash="dash", line_color="#FF006E",
        annotation_text=f"Limiar ({limiar}°)",
        annotation_position="top right",
        row=1, col=1,
    )

    # Marcadores de pico (esquerdo)
    for p in picos_esq:
        fig.add_vrect(
            x0=p['inicio_seg'], x1=p['fim_seg'],
            fillcolor="#3A86FF", opacity=0.15,
            layer="below", line_width=0,
            row=1, col=1,
        )
        fig.add_trace(go.Scatter(
            x=[tempo[p['frame_pico']]],
            y=[p['amplitude_maxima']],
            mode='markers+text',
            marker=dict(color='#3A86FF', size=8, symbol='circle'),
            text=[f"P{p['pico']}<br>{p['amplitude_maxima']:.1f}°"],
            textposition='top center',
            textfont=dict(size=9),
            showlegend=False,
            name=f"Pico {p['pico']}",
        ), row=1, col=1)

    # --- Lado direito ---
    fig.add_trace(go.Scatter(
        x=tempo, y=sinal_dir,
        mode='lines', name='Direito',
        line=dict(color='#FB5607', width=1.5),
    ), row=2, col=1)

    fig.add_hline(
        y=limiar, line_dash="dash", line_color="#FF006E",
        annotation_text=f"Limiar ({limiar}°)",
        annotation_position="top right",
        row=2, col=1,
    )

    # Marcadores de pico (direito)
    for p in picos_dir:
        fig.add_vrect(
            x0=p['inicio_seg'], x1=p['fim_seg'],
            fillcolor="#FB5607", opacity=0.15,
            layer="below", line_width=0,
            row=2, col=1,
        )
        fig.add_trace(go.Scatter(
            x=[tempo[p['frame_pico']]],
            y=[p['amplitude_maxima']],
            mode='markers+text',
            marker=dict(color='#FB5607', size=8, symbol='circle'),
            text=[f"P{p['pico']}<br>{p['amplitude_maxima']:.1f}°"],
            textposition='top center',
            textfont=dict(size=9),
            showlegend=False,
            name=f"Pico {p['pico']}",
        ), row=2, col=1)

    fig.update_yaxes(title_text=unidade, row=1, col=1)
    fig.update_yaxes(title_text=unidade, row=2, col=1)
    fig.update_xaxes(title_text="Tempo (s)", row=2, col=1)
    fig.update_layout(
        title=titulo,
        height=520,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=60, b=20),
    )
    return fig


def _plot_tornozelo(
    tempo: np.ndarray,
    sinal_esq: pd.Series,
    sinal_dir: pd.Series,
    titulo: str,
) -> go.Figure:
    """Gráfico simples do tornozelo — sinal de compensação postural."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tempo, y=sinal_esq,
        mode='lines', name='Tornozelo Esquerdo',
        line=dict(color='#8338EC', width=1.2),
    ))
    fig.add_trace(go.Scatter(
        x=tempo, y=sinal_dir,
        mode='lines', name='Tornozelo Direito',
        line=dict(color='#FFBE0B', width=1.2),
    ))
    fig.update_layout(
        title=titulo,
        xaxis_title="Tempo (s)",
        yaxis_title="Amplitude (graus)",
        height=300,
        legend=dict(orientation="h", y=-0.3),
        margin=dict(t=50, b=10),
    )
    return fig


def _plot_comparacao_amplitudes(
    picos_a: list[dict], picos_b: list[dict],
    label_a: str, label_b: str,
    titulo: str,
) -> go.Figure:
    """
    Gráfico de barras comparando as amplitudes máximas de cada pico
    entre duas sessões (modo comparação).
    """
    fig = go.Figure()

    if picos_a:
        fig.add_trace(go.Bar(
            x=[f"P{p['pico']}" for p in picos_a],
            y=[p['amplitude_maxima'] for p in picos_a],
            name=label_a,
            marker_color='#3A86FF',
        ))
    if picos_b:
        fig.add_trace(go.Bar(
            x=[f"P{p['pico']}" for p in picos_b],
            y=[p['amplitude_maxima'] for p in picos_b],
            name=label_b,
            marker_color='#FF006E',
        ))

    fig.update_layout(
        title=titulo,
        xaxis_title="Pico",
        yaxis_title="Amplitude máxima (graus)",
        barmode='group',
        height=350,
        legend=dict(orientation="h", y=-0.3),
        margin=dict(t=50, b=10),
    )
    return fig


# =============================================================================
# SEÇÕES DA VIEW
# =============================================================================

def _secao_metricas(resumo: dict, label: str = ""):
    """
    Renderiza os cards de métricas biomecânicas de uma sessão:
    simetria, jerk e consistência temporal.
    """
    prefixo = f"{label} — " if label else ""

    st.markdown(f"#### {prefixo}Métricas biomecânicas")

    # --- Simetria ---
    st.markdown("**Índice de Simetria** *(1.0 = perfeito | < 0.85 = assimetria clínica)*")
    c1, c2 = st.columns(2)
    sim_q = resumo['simetria']['quadril']
    sim_j = resumo['simetria']['joelho']
    c1.metric(
        "Quadril",
        f"{sim_q:.2f}",
        delta="OK" if sim_q >= LIMIAR_SIMETRIA else "Assimetria",
        delta_color=_cor_simetria(sim_q),
    )
    c2.metric(
        "Joelho",
        f"{sim_j:.2f}",
        delta="OK" if sim_j >= LIMIAR_SIMETRIA else "Assimetria",
        delta_color=_cor_simetria(sim_j),
    )

    st.markdown("---")

    # --- Jerk (suavidade) ---
    st.markdown("**Jerk RMS** *(graus/s³ — quanto menor, mais suave o movimento)*")
    c1, c2, c3, c4 = st.columns(4)
    jerk = resumo['jerk']
    c1.metric("Quadril Esq.", f"{jerk['quadril_esq']:.1f}")
    c2.metric("Quadril Dir.", f"{jerk['quadril_dir']:.1f}")
    c3.metric("Joelho Esq.",  f"{jerk['joelho_esq']:.1f}")
    c4.metric("Joelho Dir.",  f"{jerk['joelho_dir']:.1f}")

    st.markdown("---")

    # --- Consistência temporal ---
    st.markdown("**Consistência Temporal dos Picos** *(CV% — quanto menor, mais regular o ritmo)*")
    c1, c2, c3, c4 = st.columns(4)
    cons = resumo['consistencia']
    c1.metric(
        "Quadril Esq.",
        f"{cons['quadril_esq']['cv_duracao']:.1f}%",
        help=f"{cons['quadril_esq']['n_picos']} picos | "
             f"média {cons['quadril_esq']['media_duracao']:.2f}s ± "
             f"{cons['quadril_esq']['desvio_duracao']:.2f}s",
    )
    c2.metric(
        "Quadril Dir.",
        f"{cons['quadril_dir']['cv_duracao']:.1f}%",
        help=f"{cons['quadril_dir']['n_picos']} picos | "
             f"média {cons['quadril_dir']['media_duracao']:.2f}s ± "
             f"{cons['quadril_dir']['desvio_duracao']:.2f}s",
    )
    c3.metric(
        "Joelho Esq.",
        f"{cons['joelho_esq']['cv_duracao']:.1f}%",
        help=f"{cons['joelho_esq']['n_picos']} picos | "
             f"média {cons['joelho_esq']['media_duracao']:.2f}s ± "
             f"{cons['joelho_esq']['desvio_duracao']:.2f}s",
    )
    c4.metric(
        "Joelho Dir.",
        f"{cons['joelho_dir']['cv_duracao']:.1f}%",
        help=f"{cons['joelho_dir']['n_picos']} picos | "
             f"média {cons['joelho_dir']['media_duracao']:.2f}s ± "
             f"{cons['joelho_dir']['desvio_duracao']:.2f}s",
    )


def _secao_picos(resumo: dict, tempo: np.ndarray, df: pd.DataFrame,
                 limiares: dict, label: str = ""):
    """
    Renderiza os gráficos de sinal com picos marcados e as tabelas
    de detalhes por pico para quadril e joelho.
    """
    prefixo = f"{label} — " if label else ""

    # Quadril
    st.markdown(f"#### {prefixo}Quadril")
    fig_q = _plot_sinal_com_picos(
        tempo=tempo,
        sinal_esq=df[COL_QUADRIL_ESQ],
        sinal_dir=df[COL_QUADRIL_DIR],
        picos_esq=resumo['picos']['quadril_esq'],
        picos_dir=resumo['picos']['quadril_dir'],
        limiar=limiares['quadril'],
        titulo="Amplitude de Flexão do Quadril",
    )
    st.plotly_chart(fig_q, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Picos detectados — Esquerdo")
        df_qe = picos_para_dataframe(resumo['picos']['quadril_esq'])
        if not df_qe.empty:
            st.dataframe(
                df_qe[['amplitude_maxima', 'duracao', 'inicio_seg', 'fim_seg']]
                .rename(columns={
                    'amplitude_maxima': 'Amplitude (°)',
                    'duracao':          'Duração (s)',
                    'inicio_seg':       'Início (s)',
                    'fim_seg':          'Fim (s)',
                }),
                use_container_width=True,
            )
        else:
            st.info("Nenhum pico detectado com o limiar definido.")

    with col2:
        st.caption("Picos detectados — Direito")
        df_qd = picos_para_dataframe(resumo['picos']['quadril_dir'])
        if not df_qd.empty:
            st.dataframe(
                df_qd[['amplitude_maxima', 'duracao', 'inicio_seg', 'fim_seg']]
                .rename(columns={
                    'amplitude_maxima': 'Amplitude (°)',
                    'duracao':          'Duração (s)',
                    'inicio_seg':       'Início (s)',
                    'fim_seg':          'Fim (s)',
                }),
                use_container_width=True,
            )
        else:
            st.info("Nenhum pico detectado com o limiar definido.")

    st.markdown("---")

    # Joelho
    st.markdown(f"#### {prefixo}Joelho")
    fig_j = _plot_sinal_com_picos(
        tempo=tempo,
        sinal_esq=df[COL_JOELHO_ESQ],
        sinal_dir=df[COL_JOELHO_DIR],
        picos_esq=resumo['picos']['joelho_esq'],
        picos_dir=resumo['picos']['joelho_dir'],
        limiar=limiares['joelho'],
        titulo="Amplitude de Flexão do Joelho",
    )
    st.plotly_chart(fig_j, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Picos detectados — Esquerdo")
        df_je = picos_para_dataframe(resumo['picos']['joelho_esq'])
        if not df_je.empty:
            st.dataframe(
                df_je[['amplitude_maxima', 'duracao', 'inicio_seg', 'fim_seg']]
                .rename(columns={
                    'amplitude_maxima': 'Amplitude (°)',
                    'duracao':          'Duração (s)',
                    'inicio_seg':       'Início (s)',
                    'fim_seg':          'Fim (s)',
                }),
                use_container_width=True,
            )
        else:
            st.info("Nenhum pico detectado com o limiar definido.")

    with col2:
        st.caption("Picos detectados — Direito")
        df_jd = picos_para_dataframe(resumo['picos']['joelho_dir'])
        if not df_jd.empty:
            st.dataframe(
                df_jd[['amplitude_maxima', 'duracao', 'inicio_seg', 'fim_seg']]
                .rename(columns={
                    'amplitude_maxima': 'Amplitude (°)',
                    'duracao':          'Duração (s)',
                    'inicio_seg':       'Início (s)',
                    'fim_seg':          'Fim (s)',
                }),
                use_container_width=True,
            )
        else:
            st.info("Nenhum pico detectado com o limiar definido.")


def _secao_tornozelo(resumo: dict, tempo: np.ndarray, df: pd.DataFrame,
                     label: str = ""):
    """Renderiza o painel de compensação do tornozelo."""
    prefixo = f"{label} — " if label else ""
    with st.expander(f"🦶 {prefixo}Tornozelo (compensação postural)", expanded=False):
        st.caption(
            "O tornozelo não é o foco da análise, mas oscilações excessivas podem "
            "indicar compensação postural — relevante para o fisioterapeuta."
        )
        fig_t = _plot_tornozelo(
            tempo=tempo,
            sinal_esq=df[COL_TORNOZELO_ESQ],
            sinal_dir=df[COL_TORNOZELO_DIR],
            titulo=f"Amplitude do Tornozelo",
        )
        st.plotly_chart(fig_t, use_container_width=True)

        comp = resumo['compensacao_tornozelo']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Amplitude média Esq.", f"{comp['amplitude_media_esq']:.1f}°")
        c2.metric("Amplitude média Dir.", f"{comp['amplitude_media_dir']:.1f}°")
        c3.metric("Amplitude máx. Esq.", f"{comp['amplitude_max_esq']:.1f}°")
        c4.metric("Amplitude máx. Dir.", f"{comp['amplitude_max_dir']:.1f}°")


# =============================================================================
# MODO SESSÃO ÚNICA
# =============================================================================

def _modo_sessao_unica(limiares: dict):
    st.markdown("### 📁 Upload da Sessão")
    arquivo = st.file_uploader(
        "Envie o arquivo da sessão (CSV, XLSX ou XLSM)",
        type=["csv", "xlsx", "xlsm"],
        key="unico",
    )
    if not arquivo:
        st.info("Envie um arquivo para iniciar a análise.")
        return

    try:
        df = carregar_arquivo(arquivo)
    except ValueError as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return

    fps    = calcular_fps(df)
    tempo  = _tempo_eixo(df, fps)
    resumo = gerar_resumo_sessao(df, fps, limiares)

    duracao_total = len(df) / fps
    st.success(
        f"✅ Arquivo carregado — **{len(df)} frames** | "
        f"**{duracao_total:.1f}s** | FPS: {fps}"
    )

    st.markdown("---")
    _secao_metricas(resumo)
    st.markdown("---")
    _secao_picos(resumo, tempo, df, limiares)
    st.markdown("---")
    _secao_tornozelo(resumo, tempo, df)


# =============================================================================
# MODO COMPARAÇÃO (INÍCIO vs FINAL)
# =============================================================================

def _modo_comparacao(limiares: dict):
    st.markdown("### 📁 Upload das Sessões")
    col1, col2 = st.columns(2)
    with col1:
        arq_inicio = st.file_uploader(
            "📂 Sessão de Início",
            type=["csv", "xlsx", "xlsm"],
            key="inicio",
        )
    with col2:
        arq_final = st.file_uploader(
            "📂 Sessão Final",
            type=["csv", "xlsx", "xlsm"],
            key="final",
        )

    if not (arq_inicio and arq_final):
        st.info("Envie ambos os arquivos para iniciar a comparação.")
        return

    try:
        df_i = carregar_arquivo(arq_inicio)
        df_f = carregar_arquivo(arq_final)
    except ValueError as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return

    fps_i = calcular_fps(df_i)
    fps_f = calcular_fps(df_f)

    tempo_i = _tempo_eixo(df_i, fps_i)
    tempo_f = _tempo_eixo(df_f, fps_f)

    resumo_i = gerar_resumo_sessao(df_i, fps_i, limiares)
    resumo_f = gerar_resumo_sessao(df_f, fps_f, limiares)

    col1, col2 = st.columns(2)
    col1.success(
        f"✅ Início — **{len(df_i)} frames** | "
        f"**{len(df_i)/fps_i:.1f}s**"
    )
    col2.success(
        f"✅ Final — **{len(df_f)} frames** | "
        f"**{len(df_f)/fps_f:.1f}s**"
    )

    # --- Métricas lado a lado ---
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        _secao_metricas(resumo_i, label="Início")
    with col2:
        _secao_metricas(resumo_f, label="Final")

    # --- Gráfico comparativo de amplitudes ---
    st.markdown("---")
    st.markdown("#### 📊 Comparação de Amplitudes por Pico")

    tab_q_esq, tab_q_dir, tab_j_esq, tab_j_dir = st.tabs([
        "Quadril Esquerdo", "Quadril Direito",
        "Joelho Esquerdo",  "Joelho Direito",
    ])
    with tab_q_esq:
        st.plotly_chart(_plot_comparacao_amplitudes(
            resumo_i['picos']['quadril_esq'], resumo_f['picos']['quadril_esq'],
            "Início", "Final", "Quadril Esquerdo — Amplitude por Pico",
        ), use_container_width=True)
    with tab_q_dir:
        st.plotly_chart(_plot_comparacao_amplitudes(
            resumo_i['picos']['quadril_dir'], resumo_f['picos']['quadril_dir'],
            "Início", "Final", "Quadril Direito — Amplitude por Pico",
        ), use_container_width=True)
    with tab_j_esq:
        st.plotly_chart(_plot_comparacao_amplitudes(
            resumo_i['picos']['joelho_esq'], resumo_f['picos']['joelho_esq'],
            "Início", "Final", "Joelho Esquerdo — Amplitude por Pico",
        ), use_container_width=True)
    with tab_j_dir:
        st.plotly_chart(_plot_comparacao_amplitudes(
            resumo_i['picos']['joelho_dir'], resumo_f['picos']['joelho_dir'],
            "Início", "Final", "Joelho Direito — Amplitude por Pico",
        ), use_container_width=True)

    # --- Sinais detalhados por sessão ---
    st.markdown("---")
    st.markdown("#### 🔍 Sinais Detalhados")
    aba_i, aba_f = st.tabs(["Sessão Início", "Sessão Final"])
    with aba_i:
        _secao_picos(resumo_i, tempo_i, df_i, limiares, label="Início")
        _secao_tornozelo(resumo_i, tempo_i, df_i, label="Início")
    with aba_f:
        _secao_picos(resumo_f, tempo_f, df_f, limiares, label="Final")
        _secao_tornozelo(resumo_f, tempo_f, df_f, label="Final")


# =============================================================================
# PONTO DE ENTRADA DA VIEW
# =============================================================================

def carregar():
    st.markdown(
        "Análise cinemática dos membros inferiores durante a intervenção. "
        "Monitora flexão de quadril e joelho, simetria bilateral, suavidade do movimento e compensação postural."
    )

    # --- Limiares clínicos (sidebar) ---
    with st.sidebar:
        st.markdown("## ⚙️ Parâmetros Clínicos")
        st.caption("Definidos pelo profissional de saúde para esta sessão.")
        limiar_quadril = st.number_input(
            "Limiar de Quadril (graus)",
            min_value=0.0, max_value=180.0,
            value=45.0, step=1.0,
            help="Ângulo mínimo de flexão do quadril para validar o movimento.",
        )
        limiar_joelho = st.number_input(
            "Limiar de Joelho (graus)",
            min_value=0.0, max_value=180.0,
            value=30.0, step=1.0,
            help="Ângulo mínimo de flexão do joelho para validar o movimento.",
        )
        limiares = {'quadril': limiar_quadril, 'joelho': limiar_joelho}

        st.markdown("---")
        st.markdown("## 📋 Modo de Análise")
        modo = st.radio(
            "Selecione o modo:",
            options=["Sessão única", "Comparação (Início vs Final)"],
            index=0,
        )

    if modo == "Sessão única":
        _modo_sessao_unica(limiares)
    else:
        _modo_comparacao(limiares)