"""
Funções de processamento originais da Iniciação Científica (membros superiores).
Mantidas separadas para não conflitar com o novo processamento.py do TCC.
Usadas exclusivamente pela view visualizacao_estatistica.py.
"""

import numpy as np
import plotly.graph_objects as go


def calcular_frames_por_segundo(df, time_col='time'):
    """
    Calcula a taxa de quadros (FPS) da sessão capturada.
    """
    df['seconds'] = df[time_col].astype(str).str[-2:]
    df['count'] = (df['seconds'] != df['seconds'].shift(1)).cumsum()
    result = df.groupby('count').size().reset_index(name='counts')
    return result.loc[1, 'counts'] if len(result) > 1 else 30


def calcular_tempos_picos(dados, fps, limiar, min_duracao=2):
    """
    Calcula os intervalos de tempo em que o sinal ultrapassa o limiar.
    """
    acima_limiar = (dados > limiar).astype(int)
    mudancas = np.diff(acima_limiar)

    inicios = np.where(mudancas == 1)[0]
    finais  = np.where(mudancas == -1)[0]

    if finais.size > 0 and inicios.size > 0:
        if finais[0] < inicios[0]:
            finais = finais[1:]
        min_len = min(len(inicios), len(finais))
        inicios, finais = inicios[:min_len], finais[:min_len]

    picos     = []
    duracoes  = []

    for i, f in zip(inicios, finais):
        duracao = (f - i) / fps
        if duracao >= min_duracao:
            picos.append((i, f))
            duracoes.append(duracao)

    media = np.mean(duracoes) if duracoes else 0.0
    return picos, duracoes, media


def classificar(duracoes):
    """
    Classifica os movimentos com base na duração dos picos.
    """
    classificacoes = []
    for d in duracoes:
        if d < 5:
            classificacoes.append("Rápido")
        elif d <= 9:
            classificacoes.append("Moderado")
        else:
            classificacoes.append("Lento")
    return classificacoes


def plot_intervalos_picos(tempo, dados, limiar, picos, titulo):
    """
    Gera gráfico com o sinal e os intervalos de pico destacados.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=tempo, y=dados,
        mode='lines',
        name='Sinal',
        line=dict(color='blue'),
    ))

    fig.add_hline(
        y=limiar,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Limiar ({limiar}°)",
        annotation_position="top right",
    )

    for idx, (i, f) in enumerate(picos):
        fig.add_vrect(
            x0=tempo[i], x1=tempo[f],
            fillcolor="green", opacity=0.2,
            layer="below", line_width=0,
        )

    fig.update_layout(
        title=titulo,
        xaxis_title="Tempo (s)",
        yaxis_title="Amplitude (graus)",
        height=400,
    )

    return fig