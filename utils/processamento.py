import numpy as np
import plotly.graph_objects as go

def calcular_frames_por_segundo(df, time_col='time'):
    """
    Calcula a taxa de quadros (FPS) da sessão capturada.
    """

    df['seconds'] = df[time_col].astype(str).str[-2:]
    df['count'] = (df['seconds'] != df['seconds'].shift(1)).cumsum()
    result = df.groupby('count').size().reset_index(name='counts')
    # Retorna o FPS do primeiro segundo estável
    return result.loc[1, 'counts'] if len(result) > 1 else 30


def extrair_picos_marcha(dados, fps, limiar_selecionado):
    """
    Identifica as tentativas de transposição de obstáculos na marcha estacionária.
    
    Parâmetros:
        dados: Coluna de ângulos (ex: hipLangle ou kneeLangle).
        fps: Frames por segundo calculados.
        limiar_selecionado: O ângulo mínimo definido pelo clínico para validar o movimento.
    """
    tempo = np.arange(len(dados)) / fps
    
    # Mascara onde o movimento supera o limiar clínico
    acima_limiar = (dados > limiar_selecionado).astype(int)
    mudancas = np.diff(acima_limiar)
    
    inicios = np.where(mudancas == 1)[0]
    finais = np.where(mudancas == -1)[0]

    # Garantir que cada início tenha um fim correspondente
    if finais.size > 0 and inicios.size > 0:
        if finais[0] < inicios[0]:
            finais = finais[1:]
        min_len = min(len(inicios), len(finais))
        inicios, finais = inicios[:min_len], finais[:min_len]

    picos = []
    for i, f in zip(inicios, finais):
        # Extrai a amplitude máxima atingida naquele passo específico
        amplitude_pico = dados[i:f].max()
        
        picos.append({
            'inicio_seg': tempo[i],
            'fim_seg': tempo[f],
            'duracao': tempo[f] - tempo[i],
            'amplitude_maxima': amplitude_pico
        })
            
    return picos


def calcular_indice_simetria(picos_esq, picos_dir):
    """
    Calcula a simetria entre membros inferiores (biomarcador digital).
    Baseado na amplitude média alcançada por cada lado.
    """
    if not picos_esq or not picos_dir:
        return 0.0
    
    media_esq = np.mean([p['amplitude_maxima'] for p in picos_esq])
    media_dir = np.mean([p['amplitude_maxima'] for p in picos_dir])
    
    # Razão de simetria (menor valor / maior valor)
    return min(media_esq, media_dir) / max(media_esq, media_dir)