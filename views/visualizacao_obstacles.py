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
    extrair_picos_marcha,
    calcular_jerk,
    calcular_consistencia_temporal,
    COL_QUADRIL_ESQ, COL_QUADRIL_DIR,
    COL_JOELHO_ESQ,  COL_JOELHO_DIR,
    COL_TORNOZELO_ESQ, COL_TORNOZELO_DIR,
)

# =============================================================================
# CONSTANTE DE REFERÊNCIA CLÍNICA
# =============================================================================
LIMIAR_SIMETRIA = 0.85


# =============================================================================
# ABA DE GLOSSÁRIO / DESCRIÇÃO DAS MÉTRICAS
# =============================================================================

def _aba_metricas_info():
    """
    Explica ao fisioterapeuta o que é cada métrica calculada pelo dashboard,
    por que ela é relevante e como interpretá-la clinicamente.
    """
    st.markdown("## 📖 Guia das Métricas")
    st.markdown(
        "Esta seção explica cada indicador calculado pelo dashboard, "
        "para que o profissional de saúde possa interpretar os resultados com segurança."
    )
    st.markdown("---")

    with st.expander("🧬 O que são Métricas Biomecânicas?", expanded=True):
        st.markdown(
            """
            **Métricas biomecânicas** são indicadores numéricos extraídos do movimento do paciente
            que permitem avaliar a **qualidade, simetria e eficiência** da marcha de forma objetiva.

            Na prática clínica tradicional, a avaliação do movimento depende da observação visual
            do terapeuta — que é subjetiva e pode variar entre profissionais. As métricas biomecânicas
            transformam esse julgamento em **dados mensuráveis e reproduzíveis**, facilitando o
            acompanhamento da evolução ao longo do tratamento e a comunicação entre a equipe de saúde.

            O dashboard calcula três grupos de métricas: **Índice de Simetria**, **Jerk** e
            **Consistência Temporal**, descritos abaixo.
            """
        )

    st.markdown("---")

    with st.expander("⚖️ Índice de Simetria", expanded=False):
        st.markdown(
            """
            **O que é:**
            Mede o equilíbrio entre os membros inferiores esquerdo e direito durante o movimento.
            É calculado como a razão entre a amplitude média do lado menos ativo e a do lado mais ativo.

            **Como interpretar:**
            | Valor | Interpretação |
            |---|---|
            | 1,00 | Simetria perfeita — os dois lados se movem de forma idêntica |
            | 0,85 a 0,99 | Dentro da faixa considerada normal pela literatura |
            | < 0,85 | **Assimetria clinicamente relevante** — um lado está compensando o outro |

            **Por que importa:**
            Em pacientes neurofuncionais, a assimetria pode indicar fraqueza muscular unilateral,
            dor, espasticidade ou compensação postural. O monitoramento ao longo das sessões
            permite identificar se o tratamento está reduzindo essa diferença entre os lados.

            **Referência:** Limiar de 0,85 baseado em literatura de análise de marcha clínica
            (Robinson & Herzog, 1987; Sadeghi et al., 2000).
            """
        )

    with st.expander("〰️ Jerk (Suavidade do Movimento)", expanded=False):
        st.markdown(
            """
            **O que é:**
            O jerk é a **terceira derivada da posição angular** em relação ao tempo — ou seja,
            a taxa de variação da aceleração do movimento. Em termos simples, mede o quanto
            o movimento é *brusco* ou *suave*.

            É expresso em **graus por segundo ao cubo (°/s³)** e calculado como a raiz quadrada
            da média dos quadrados dos valores de jerk ao longo de toda a sessão (RMS).

            **Como interpretar:**
            - **Valores menores** → movimento mais fluido e controlado (padrão saudável)
            - **Valores maiores** → movimento mais brusco, com acelerações e desacelerações abruptas

            **Por que é relevante aqui:**
            Pacientes em reabilitação neurofuncional frequentemente apresentam movimentos
            descoordenados, com tremores ou compensações que aumentam o jerk. À medida que
            o tratamento avança, espera-se uma **redução progressiva do jerk** — indicando
            que o sistema nervoso está recuperando o controle motor fino.

            Comparar o jerk entre sessões de início e fim do tratamento é uma forma objetiva
            de quantificar a melhora da qualidade do movimento, além da simples amplitude.

            **Atenção:** O jerk é calculado sobre o sinal completo da sessão, não apenas
            sobre os picos. Isso é intencional — movimentos fora dos picos (como a descida
            do membro) também carregam informação clínica relevante.
            """
        )

    with st.expander("🕐 Consistência Temporal", expanded=False):
        st.markdown(
            """
            **O que é:**
            Avalia se o paciente manteve um **ritmo regular** ao longo dos movimentos detectados
            na sessão. É calculada como o **Coeficiente de Variação (CV%)** das durações de
            cada pico de movimento.

            O CV% é o desvio padrão dividido pela média, expresso em porcentagem.

            **Como interpretar:**
            | CV% | Interpretação |
            |---|---|
            | < 15% | Ritmo muito consistente — boa regularidade motora |
            | 15% a 30% | Variabilidade moderada — aceitável em reabilitação |
            | > 30% | **Alta irregularidade** — pode indicar fadiga, dor ou déficit de controle motor |

            **Por que importa:**
            Um paciente que realiza os movimentos em tempos muito diferentes entre si pode estar
            alternando esforço e descanso, sentindo dor, ou com dificuldade de manter o padrão
            motor. A consistência tende a melhorar com o avanço do tratamento.
            """
        )

    with st.expander("📌 Picos de Flexão Detectados", expanded=False):
        st.markdown(
            """
            **O que são:**
            Cada "pico" corresponde a um intervalo de tempo em que o ângulo de flexão da
            articulação ultrapassou o **limiar clínico** definido pelo profissional de saúde
            na barra lateral. Representa um movimento válido dentro do protocolo terapêutico.

            **Campos da tabela:**
            | Campo | Descrição |
            |---|---|
            | Amplitude (°) | Maior ângulo atingido durante aquele movimento |
            | Duração (s) | Tempo que o membro ficou acima do limiar |
            | Início (s) | Momento em que o movimento ultrapassou o limiar |
            | Fim (s) | Momento em que o movimento voltou abaixo do limiar |

            **Filtro anti-ruído:**
            Eventos muito curtos (menos de 5 frames ≈ 0,18s) são automaticamente ignorados
            para evitar que ruídos do sensor sejam confundidos com movimentos reais.
            """
        )

    with st.expander("🦶 Tornozelo — Compensação Postural", expanded=False):
        st.markdown(
            """
            **Por que monitorar o tornozelo se o foco é quadril e joelho?**

            Em pacientes com dificuldade de elevar o membro inferior, é comum que o corpo
            desenvolva **estratégias compensatórias**: o tronco inclina, os braços se abrem
            para equilíbrio, e o tornozelo pode realizar movimentos exagerados para auxiliar
            na elevação da perna.

            O sinal do tornozelo não é analisado com picos e limiares, mas sua **amplitude
            média e máxima** são exibidas para que o terapeuta possa identificar se há
            oscilação postural excessiva que deva ser trabalhada no protocolo.

            Ambos os tornozelos são sempre monitorados, independentemente do modo de análise
            (bilateral ou unilateral), pois a compensação pode ocorrer no membro oposto ao
            que está sendo exercitado.
            """
        )


# =============================================================================
# HELPERS DE VISUALIZAÇÃO
# =============================================================================

def _tempo_eixo(df: pd.DataFrame, fps: float) -> np.ndarray:
    return np.arange(len(df)) / fps


def _cor_simetria(valor: float) -> str:
    return "normal" if valor >= LIMIAR_SIMETRIA else "inverse"


def _plot_sinal_com_picos(
    tempo, sinal_esq, sinal_dir, picos_esq, picos_dir,
    limiar, titulo, unidade="graus",
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Esquerdo", "Direito"),
        vertical_spacing=0.12,
    )
    for row, sinal, picos, cor in [
        (1, sinal_esq, picos_esq, '#3A86FF'),
        (2, sinal_dir, picos_dir, '#FB5607'),
    ]:
        nome = "Esquerdo" if row == 1 else "Direito"
        fig.add_trace(go.Scatter(x=tempo, y=sinal, mode='lines', name=nome,
                                 line=dict(color=cor, width=1.5)), row=row, col=1)
        fig.add_hline(y=limiar, line_dash="dash", line_color="#FF006E",
                      annotation_text=f"Limiar ({limiar}°)",
                      annotation_position="top right", row=row, col=1)
        for p in picos:
            fig.add_vrect(x0=p['inicio_seg'], x1=p['fim_seg'],
                          fillcolor=cor, opacity=0.15, layer="below", line_width=0,
                          row=row, col=1)
            fig.add_trace(go.Scatter(
                x=[tempo[p['frame_pico']]], y=[p['amplitude_maxima']],
                mode='markers+text',
                marker=dict(color=cor, size=8, symbol='circle'),
                text=[f"P{p['pico']}<br>{p['amplitude_maxima']:.1f}°"],
                textposition='top center', textfont=dict(size=9),
                showlegend=False,
            ), row=row, col=1)
        fig.update_yaxes(title_text=unidade, row=row, col=1)

    fig.update_xaxes(title_text="Tempo (s)", row=2, col=1)
    fig.update_layout(title=titulo, height=520,
                      legend=dict(orientation="h", y=-0.15),
                      margin=dict(t=60, b=20))
    return fig


def _plot_sinal_unilateral(tempo, sinal, picos, limiar, titulo, cor, lado) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tempo, y=sinal, mode='lines', name=lado,
                             line=dict(color=cor, width=1.5)))
    fig.add_hline(y=limiar, line_dash="dash", line_color="#FF006E",
                  annotation_text=f"Limiar ({limiar}°)", annotation_position="top right")
    for p in picos:
        fig.add_vrect(x0=p['inicio_seg'], x1=p['fim_seg'],
                      fillcolor=cor, opacity=0.15, layer="below", line_width=0)
        fig.add_trace(go.Scatter(
            x=[tempo[p['frame_pico']]], y=[p['amplitude_maxima']],
            mode='markers+text',
            marker=dict(color=cor, size=8, symbol='circle'),
            text=[f"P{p['pico']}<br>{p['amplitude_maxima']:.1f}°"],
            textposition='top center', textfont=dict(size=9), showlegend=False,
        ))
    fig.update_layout(title=titulo, xaxis_title="Tempo (s)", yaxis_title="graus",
                      height=350, margin=dict(t=50, b=20))
    return fig


def _plot_overview_bilateral(tempo, df, titulo, col_esq, col_dir) -> go.Figure:
    """Overview rápido dos dois lados para o terapeuta identificar o lado ativo."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tempo, y=df[col_esq], mode='lines', name='Esquerdo',
                             line=dict(color='#3A86FF', width=1.2)))
    fig.add_trace(go.Scatter(x=tempo, y=df[col_dir], mode='lines', name='Direito',
                             line=dict(color='#FB5607', width=1.2)))
    fig.update_layout(title=titulo, xaxis_title="Tempo (s)", yaxis_title="graus",
                      height=260, legend=dict(orientation="h", y=-0.4),
                      margin=dict(t=40, b=10))
    return fig


def _plot_tornozelo(tempo, sinal_esq, sinal_dir) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tempo, y=sinal_esq, mode='lines', name='Esquerdo',
                             line=dict(color='#8338EC', width=1.2)))
    fig.add_trace(go.Scatter(x=tempo, y=sinal_dir, mode='lines', name='Direito',
                             line=dict(color='#FFBE0B', width=1.2)))
    fig.update_layout(xaxis_title="Tempo (s)", yaxis_title="graus",
                      height=260, legend=dict(orientation="h", y=-0.4),
                      margin=dict(t=10, b=10))
    return fig


def _plot_comparacao_amplitudes(picos_a, picos_b, label_a, label_b, titulo) -> go.Figure:
    fig = go.Figure()
    if picos_a:
        fig.add_trace(go.Bar(x=[f"P{p['pico']}" for p in picos_a],
                             y=[p['amplitude_maxima'] for p in picos_a],
                             name=label_a, marker_color='#3A86FF'))
    if picos_b:
        fig.add_trace(go.Bar(x=[f"P{p['pico']}" for p in picos_b],
                             y=[p['amplitude_maxima'] for p in picos_b],
                             name=label_b, marker_color='#FF006E'))
    fig.update_layout(title=titulo, xaxis_title="Pico",
                      yaxis_title="Amplitude máxima (graus)",
                      barmode='group', height=350,
                      legend=dict(orientation="h", y=-0.3),
                      margin=dict(t=50, b=10))
    return fig


# =============================================================================
# SEÇÕES REUTILIZÁVEIS
# =============================================================================

def _tabela_picos(picos: list[dict]):
    df = picos_para_dataframe(picos)
    if not df.empty:
        st.dataframe(
            df[['amplitude_maxima', 'duracao', 'inicio_seg', 'fim_seg']].rename(columns={
                'amplitude_maxima': 'Amplitude (°)',
                'duracao':          'Duração (s)',
                'inicio_seg':       'Início (s)',
                'fim_seg':          'Fim (s)',
            }),
            use_container_width=True,
        )
    else:
        st.info("Nenhum pico detectado com o limiar definido.")


def _secao_metricas_bilateral(resumo: dict, label: str = ""):
    prefixo = f"{label} — " if label else ""
    st.markdown(f"#### {prefixo}Métricas Biomecânicas")
    st.caption(
        "Indicadores numéricos que quantificam qualidade, simetria e eficiência do movimento. "
        "Consulte a aba **📖 Guia de Métricas** para uma explicação detalhada de cada um."
    )

    st.markdown("**Índice de Simetria** *(1.0 = perfeito | < 0.85 = assimetria clínica)*")
    c1, c2 = st.columns(2)
    sim_q, sim_j = resumo['simetria']['quadril'], resumo['simetria']['joelho']
    c1.metric("Quadril", f"{sim_q:.2f}",
              delta="OK" if sim_q >= LIMIAR_SIMETRIA else "Assimetria",
              delta_color=_cor_simetria(sim_q))
    c2.metric("Joelho", f"{sim_j:.2f}",
              delta="OK" if sim_j >= LIMIAR_SIMETRIA else "Assimetria",
              delta_color=_cor_simetria(sim_j))

    st.markdown("---")
    st.markdown("**Jerk RMS** *(°/s³ — quanto menor, mais fluido e controlado o movimento)*")
    c1, c2, c3, c4 = st.columns(4)
    jerk = resumo['jerk']
    c1.metric("Quadril Esq.", f"{jerk['quadril_esq']:.1f}")
    c2.metric("Quadril Dir.", f"{jerk['quadril_dir']:.1f}")
    c3.metric("Joelho Esq.",  f"{jerk['joelho_esq']:.1f}")
    c4.metric("Joelho Dir.",  f"{jerk['joelho_dir']:.1f}")

    st.markdown("---")
    st.markdown("**Consistência Temporal** *(CV% das durações — quanto menor, mais regular o ritmo)*")
    c1, c2, c3, c4 = st.columns(4)
    cons = resumo['consistencia']
    for col, chave, nome in [
        (c1, 'quadril_esq', 'Quadril Esq.'),
        (c2, 'quadril_dir', 'Quadril Dir.'),
        (c3, 'joelho_esq',  'Joelho Esq.'),
        (c4, 'joelho_dir',  'Joelho Dir.'),
    ]:
        col.metric(nome, f"{cons[chave]['cv_duracao']:.1f}%",
                   help=(f"{cons[chave]['n_picos']} picos detectados\n"
                         f"Média: {cons[chave]['media_duracao']:.2f}s "
                         f"± {cons[chave]['desvio_duracao']:.2f}s"))


def _secao_metricas_unilateral(picos_q, picos_j, sinal_q, sinal_j, fps, lado, label=""):
    prefixo = f"{label} — " if label else ""
    st.markdown(f"#### {prefixo}Métricas Biomecânicas — {lado}")
    st.caption(
        "No modo unilateral, o índice de simetria não é calculado. "
        "Consulte a aba **📖 Guia de Métricas** para detalhes sobre cada indicador."
    )

    jerk_q = calcular_jerk(sinal_q, fps)
    jerk_j = calcular_jerk(sinal_j, fps)
    cons_q = calcular_consistencia_temporal(picos_q)
    cons_j = calcular_consistencia_temporal(picos_j)

    st.markdown("**Jerk RMS** *(°/s³ — quanto menor, mais suave o movimento)*")
    c1, c2 = st.columns(2)
    c1.metric(f"Quadril {lado}", f"{jerk_q:.1f}")
    c2.metric(f"Joelho {lado}",  f"{jerk_j:.1f}")

    st.markdown("---")
    st.markdown("**Consistência Temporal** *(CV% — quanto menor, mais regular o ritmo)*")
    c1, c2 = st.columns(2)
    c1.metric(f"Quadril {lado}", f"{cons_q['cv_duracao']:.1f}%",
              help=f"{cons_q['n_picos']} picos | média {cons_q['media_duracao']:.2f}s ± {cons_q['desvio_duracao']:.2f}s")
    c2.metric(f"Joelho {lado}", f"{cons_j['cv_duracao']:.1f}%",
              help=f"{cons_j['n_picos']} picos | média {cons_j['media_duracao']:.2f}s ± {cons_j['desvio_duracao']:.2f}s")


def _secao_picos_bilateral(resumo, tempo, df, limiares, label=""):
    prefixo = f"{label} — " if label else ""

    st.markdown(f"#### {prefixo}Quadril")
    st.plotly_chart(_plot_sinal_com_picos(
        tempo, df[COL_QUADRIL_ESQ], df[COL_QUADRIL_DIR],
        resumo['picos']['quadril_esq'], resumo['picos']['quadril_dir'],
        limiares['quadril'], "Amplitude de Flexão do Quadril",
    ), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Picos — Esquerdo")
        _tabela_picos(resumo['picos']['quadril_esq'])
    with c2:
        st.caption("Picos — Direito")
        _tabela_picos(resumo['picos']['quadril_dir'])

    st.markdown("---")

    st.markdown(f"#### {prefixo}Joelho")
    st.plotly_chart(_plot_sinal_com_picos(
        tempo, df[COL_JOELHO_ESQ], df[COL_JOELHO_DIR],
        resumo['picos']['joelho_esq'], resumo['picos']['joelho_dir'],
        limiares['joelho'], "Amplitude de Flexão do Joelho",
    ), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Picos — Esquerdo")
        _tabela_picos(resumo['picos']['joelho_esq'])
    with c2:
        st.caption("Picos — Direito")
        _tabela_picos(resumo['picos']['joelho_dir'])


def _secao_tornozelo(resumo, tempo, df, label=""):
    prefixo = f"{label} — " if label else ""
    with st.expander(f"🦶 {prefixo}Tornozelo (compensação postural)", expanded=False):
        st.caption(
            "Ambos os tornozelos são sempre monitorados — inclusive no modo unilateral, "
            "pois a compensação pode ocorrer no membro oposto ao exercitado. "
            "Consulte a aba **📖 Guia de Métricas** para mais detalhes."
        )
        st.plotly_chart(_plot_tornozelo(tempo, df[COL_TORNOZELO_ESQ], df[COL_TORNOZELO_DIR]),
                        use_container_width=True)
        comp = resumo['compensacao_tornozelo']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Média Esq.", f"{comp['amplitude_media_esq']:.1f}°")
        c2.metric("Média Dir.", f"{comp['amplitude_media_dir']:.1f}°")
        c3.metric("Máx. Esq.", f"{comp['amplitude_max_esq']:.1f}°")
        c4.metric("Máx. Dir.", f"{comp['amplitude_max_dir']:.1f}°")


# =============================================================================
# MODO BILATERAL
# =============================================================================

def _modo_bilateral(limiares: dict):
    st.markdown("### 📁 Upload da Sessão")
    arquivo = st.file_uploader("Envie o arquivo da sessão (CSV, XLSX ou XLSM)",
                               type=["csv", "xlsx", "xlsm"], key="bilateral")
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

    st.success(f"✅ {len(df)} frames | {len(df)/fps:.1f}s | {fps} FPS")
    st.markdown("---")
    _secao_metricas_bilateral(resumo)
    st.markdown("---")
    _secao_picos_bilateral(resumo, tempo, df, limiares)
    st.markdown("---")
    _secao_tornozelo(resumo, tempo, df)


# =============================================================================
# MODO UNILATERAL
# =============================================================================

def _modo_unilateral(limiares: dict):
    st.markdown("### 📁 Upload da Sessão")
    arquivo = st.file_uploader("Envie o arquivo da sessão (CSV, XLSX ou XLSM)",
                               type=["csv", "xlsx", "xlsm"], key="unilateral")
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

    st.success(f"✅ {len(df)} frames | {len(df)/fps:.1f}s | {fps} FPS")

    # Overview bilateral para o terapeuta identificar o lado ativo
    st.markdown("---")
    st.markdown("#### 👀 Visão Geral — Identifique o lado que executou o movimento")
    st.caption(
        "Os gráficos abaixo mostram os dois lados simultaneamente. "
        "Use-os para confirmar qual membro realizou o exercício antes de selecionar o lado para análise detalhada."
    )
    st.plotly_chart(_plot_overview_bilateral(tempo, df, "Quadril — Ambos os lados",
                    COL_QUADRIL_ESQ, COL_QUADRIL_DIR), use_container_width=True)
    st.plotly_chart(_plot_overview_bilateral(tempo, df, "Joelho — Ambos os lados",
                    COL_JOELHO_ESQ, COL_JOELHO_DIR), use_container_width=True)

    # Seleção do lado
    st.markdown("---")
    lado = st.radio("Selecione o membro exercitado para análise detalhada:",
                    options=["Esquerdo", "Direito"],
                    horizontal=True, key="lado_unilateral")

    cor         = '#3A86FF' if lado == "Esquerdo" else '#FB5607'
    col_quadril = COL_QUADRIL_ESQ if lado == "Esquerdo" else COL_QUADRIL_DIR
    col_joelho  = COL_JOELHO_ESQ  if lado == "Esquerdo" else COL_JOELHO_DIR
    chave_q     = 'quadril_esq'   if lado == "Esquerdo" else 'quadril_dir'
    chave_j     = 'joelho_esq'    if lado == "Esquerdo" else 'joelho_dir'

    picos_q = resumo['picos'][chave_q]
    picos_j = resumo['picos'][chave_j]

    # Métricas
    st.markdown("---")
    _secao_metricas_unilateral(picos_q, picos_j, df[col_quadril], df[col_joelho], fps, lado)

    # Gráficos do lado selecionado
    st.markdown("---")
    st.markdown(f"#### Quadril {lado}")
    st.plotly_chart(_plot_sinal_unilateral(
        tempo, df[col_quadril], picos_q,
        limiares['quadril'], f"Flexão do Quadril — {lado}", cor, lado,
    ), use_container_width=True)
    _tabela_picos(picos_q)

    st.markdown("---")
    st.markdown(f"#### Joelho {lado}")
    st.plotly_chart(_plot_sinal_unilateral(
        tempo, df[col_joelho], picos_j,
        limiares['joelho'], f"Flexão do Joelho — {lado}", cor, lado,
    ), use_container_width=True)
    _tabela_picos(picos_j)

    # Tornozelo (sempre bilateral)
    st.markdown("---")
    _secao_tornozelo(resumo, tempo, df)


# =============================================================================
# MODO COMPARAÇÃO (INÍCIO vs FINAL)
# =============================================================================

def _modo_comparacao(limiares: dict):
    st.markdown("### 📁 Upload das Sessões")
    col1, col2 = st.columns(2)
    with col1:
        arq_inicio = st.file_uploader("📂 Sessão de Início",
                                      type=["csv", "xlsx", "xlsm"], key="inicio")
    with col2:
        arq_final = st.file_uploader("📂 Sessão Final",
                                     type=["csv", "xlsx", "xlsm"], key="final")

    if not (arq_inicio and arq_final):
        st.info("Envie ambos os arquivos para iniciar a comparação.")
        return

    try:
        df_i = carregar_arquivo(arq_inicio)
        df_f = carregar_arquivo(arq_final)
    except ValueError as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return

    fps_i, fps_f     = calcular_fps(df_i), calcular_fps(df_f)
    tempo_i, tempo_f = _tempo_eixo(df_i, fps_i), _tempo_eixo(df_f, fps_f)
    resumo_i = gerar_resumo_sessao(df_i, fps_i, limiares)
    resumo_f = gerar_resumo_sessao(df_f, fps_f, limiares)

    col1, col2 = st.columns(2)
    col1.success(f"✅ Início — {len(df_i)} frames | {len(df_i)/fps_i:.1f}s")
    col2.success(f"✅ Final  — {len(df_f)} frames | {len(df_f)/fps_f:.1f}s")

    # Métricas lado a lado
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        _secao_metricas_bilateral(resumo_i, label="Início")
    with col2:
        _secao_metricas_bilateral(resumo_f, label="Final")

    # Comparação de amplitudes por pico
    st.markdown("---")
    st.markdown("#### 📊 Comparação de Amplitudes por Pico")
    tab_qe, tab_qd, tab_je, tab_jd = st.tabs([
        "Quadril Esquerdo", "Quadril Direito",
        "Joelho Esquerdo",  "Joelho Direito",
    ])
    for tab, chave, titulo in [
        (tab_qe, 'quadril_esq', 'Quadril Esquerdo'),
        (tab_qd, 'quadril_dir', 'Quadril Direito'),
        (tab_je, 'joelho_esq',  'Joelho Esquerdo'),
        (tab_jd, 'joelho_dir',  'Joelho Direito'),
    ]:
        with tab:
            st.plotly_chart(_plot_comparacao_amplitudes(
                resumo_i['picos'][chave], resumo_f['picos'][chave],
                "Início", "Final", f"{titulo} — Amplitude por Pico",
            ), use_container_width=True)

    # Sinais detalhados por sessão
    st.markdown("---")
    st.markdown("#### 🔍 Sinais Detalhados")
    aba_i, aba_f = st.tabs(["Sessão Início", "Sessão Final"])
    with aba_i:
        _secao_picos_bilateral(resumo_i, tempo_i, df_i, limiares, label="Início")
        _secao_tornozelo(resumo_i, tempo_i, df_i, label="Início")
    with aba_f:
        _secao_picos_bilateral(resumo_f, tempo_f, df_f, limiares, label="Final")
        _secao_tornozelo(resumo_f, tempo_f, df_f, label="Final")


# =============================================================================
# PONTO DE ENTRADA DA VIEW
# =============================================================================

def carregar():
    st.markdown(
        "Análise cinemática dos membros inferiores durante a intervenção **Obstacles**. "
        "Monitora flexão de quadril e joelho, suavidade do movimento e compensação postural."
    )

    with st.sidebar:
        st.markdown("## ⚙️ Parâmetros Clínicos")
        st.caption("Definidos pelo profissional de saúde para esta sessão.")
        limiar_quadril = st.number_input(
            "Limiar de Quadril (graus)",
            min_value=0.0, max_value=180.0, value=45.0, step=1.0,
            help="Ângulo mínimo de flexão do quadril para validar o movimento.",
        )
        limiar_joelho = st.number_input(
            "Limiar de Joelho (graus)",
            min_value=0.0, max_value=180.0, value=30.0, step=1.0,
            help="Ângulo mínimo de flexão do joelho para validar o movimento.",
        )
        limiares = {'quadril': limiar_quadril, 'joelho': limiar_joelho}

        st.markdown("---")
        st.markdown("## 📋 Modo de Análise")
        modo = st.radio(
            "Selecione o modo:",
            options=[
                "Sessão única — Bilateral",
                "Sessão única — Unilateral",
                "Comparação (Início vs Final)",
            ],
            index=0,
        )

    # Duas abas: análise e guia de métricas
    aba_analise, aba_guia = st.tabs(["📊 Análise", "📖 Guia de Métricas"])

    with aba_analise:
        if modo == "Sessão única — Bilateral":
            _modo_bilateral(limiares)
        elif modo == "Sessão única — Unilateral":
            _modo_unilateral(limiares)
        else:
            _modo_comparacao(limiares)

    with aba_guia:
        _aba_metricas_info()
