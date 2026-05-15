import numpy as np
import pandas as pd


# =============================================================================
# COLUNAS ESPERADAS NO ARQUIVO DE DADOS
# =============================================================================
# O software BRAINN_XR exporta os seguintes nomes de colunas (após tratamento):
#   Frame
#   Range of Motion Hip (Left)
#   Range of Motion Hip (Right)
#   Range of Motion Knee (Left)
#   Range of Motion Knee (Right)
#   Range of Motion Ankle (Left)   <- monitorado como compensação, não é foco
#   Range of Motion Ankle (Right)  <- monitorado como compensação, não é foco

COLUNAS_ESPERADAS = [
    'Frame',
    'Range of Motion Hip (Left)',
    'Range of Motion Hip (Right)',
    'Range of Motion Knee (Left)',
    'Range of Motion Knee (Right)',
    'Range of Motion Ankle (Left)',
    'Range of Motion Ankle (Right)',
]

# Atalhos internos para facilitar o uso no código
COL_QUADRIL_ESQ = 'Range of Motion Hip (Left)'
COL_QUADRIL_DIR = 'Range of Motion Hip (Right)'
COL_JOELHO_ESQ  = 'Range of Motion Knee (Left)'
COL_JOELHO_DIR  = 'Range of Motion Knee (Right)'
COL_TORNOZELO_ESQ = 'Range of Motion Ankle (Left)'
COL_TORNOZELO_DIR = 'Range of Motion Ankle (Right)'


# =============================================================================
# 1. CARREGAMENTO DO ARQUIVO
# =============================================================================

def carregar_arquivo(arquivo) -> pd.DataFrame:
    """
    Carrega um arquivo de dados cinemáticos nos formatos CSV, XLSX ou XLSM.

    Parâmetros:
        arquivo: objeto de arquivo (ex: st.file_uploader ou caminho string).

    Retorna:
        DataFrame com os dados carregados e validados.

    Lança:
        ValueError se o formato não for suportado ou as colunas esperadas
        não forem encontradas.
    """
    nome = arquivo.name if hasattr(arquivo, 'name') else str(arquivo)
    extensao = nome.lower().split('.')[-1]

    if extensao == 'csv':
        df = pd.read_csv(arquivo)
    elif extensao in ('xlsx', 'xlsm'):
        # O arquivo do BRAINN_XR tem os dados na aba "KinesiOS"
        try:
            df = pd.read_excel(arquivo, sheet_name='KinesiOS', engine='openpyxl')
        except Exception:
            # Fallback: lê a primeira aba caso o nome mude
            df = pd.read_excel(arquivo, sheet_name=0, engine='openpyxl')
    else:
        raise ValueError(f"Formato '{extensao}' não suportado. Use CSV, XLSX ou XLSM.")

    # Valida se as colunas esperadas estão presentes
    colunas_faltando = [c for c in COLUNAS_ESPERADAS if c not in df.columns]
    if colunas_faltando:
        raise ValueError(
            f"Colunas não encontradas no arquivo:\n{colunas_faltando}\n\n"
            f"Colunas presentes: {list(df.columns)}"
        )

    # Remove linhas completamente vazias, se houver
    df = df.dropna(how='all').reset_index(drop=True)

    return df


# =============================================================================
# 2. CÁLCULO DE FPS
# =============================================================================

FPS_KINECT = 28.0  # Taxa de captura do Kinect usada pelo BRAINN_XR


def calcular_fps(df: pd.DataFrame) -> float:
    """
    Retorna a taxa de quadros (FPS) do sensor Kinect integrado ao BRAINN_XR.

    O arquivo exportado contém apenas a coluna 'Frame' como índice sequencial
    (sem timestamps reais). A taxa de captura do Kinect é fixa em 28 frames
    por segundo, valor usado para converter frames em segundos em todas as
    métricas temporais.

    Parâmetros:
        df: DataFrame carregado (usado apenas para consistência de interface).

    Retorna:
        28.0 (FPS fixo do Kinect).
    """
    return FPS_KINECT


# =============================================================================
# 3. EXTRAÇÃO DE PICOS (OBSTÁCULOS)
# =============================================================================

def extrair_picos_marcha(
    dados: pd.Series,
    fps: float,
    limiar_selecionado: float,
    janela_minima_frames: int = 5
) -> list[dict]:
    """
    Identifica todos os picos de flexão detectados na sessão de marcha
    estacionária, independentemente da quantidade.

    Um pico é detectado quando o ângulo ultrapassa o limiar clínico definido
    pelo profissional de saúde e permanece acima dele por pelo menos
    'janela_minima_frames' frames (filtro anti-ruído). O número de picos
    encontrados pode ser menor que 10 — isso é esperado em pacientes com
    mobilidade reduzida e é em si uma informação clínica relevante.

    Parâmetros:
        dados: Série de ângulos ao longo do tempo (ex: quadril esquerdo).
        fps: Taxa de quadros por segundo (28.0 para o Kinect).
        limiar_selecionado: Ângulo mínimo (graus) para validar o movimento.
        janela_minima_frames: Frames mínimos acima do limiar para ser um pico
                              válido (padrão: 5 frames ≈ 0.18s a 28fps).

    Retorna:
        Lista de dicionários, um por pico detectado, com as chaves:
            - 'pico': índice sequencial do pico (1, 2, 3, ...)
            - 'inicio_frame': frame de início do pico
            - 'fim_frame': frame de fim do pico
            - 'inicio_seg': tempo de início em segundos
            - 'fim_seg': tempo de fim em segundos
            - 'duracao': duração do pico em segundos
            - 'amplitude_maxima': maior ângulo atingido no pico (graus)
            - 'frame_pico': frame onde a amplitude máxima ocorre
    """
    dados = dados.reset_index(drop=True)
    tempo = np.arange(len(dados)) / fps

    # Máscara binária: 1 onde está acima do limiar, 0 abaixo
    acima = (dados > limiar_selecionado).astype(int)
    mudancas = np.diff(acima.values)

    inicios = np.where(mudancas == 1)[0] + 1  # +1 para apontar para o frame real
    finais  = np.where(mudancas == -1)[0] + 1

    # Alinha inícios e finais (garante pares completos)
    if len(finais) > 0 and len(inicios) > 0:
        # Se o sinal começa já acima do limiar, descarta esse final solto
        if finais[0] < inicios[0]:
            finais = finais[1:]
        # Se termina sem baixar do limiar, descarta o início sem fim
        if len(inicios) > len(finais):
            inicios = inicios[:len(finais)]
        min_len = min(len(inicios), len(finais))
        inicios = inicios[:min_len]
        finais  = finais[:min_len]
    else:
        return []  # Nenhum pico encontrado

    picos = []
    for i, f in zip(inicios, finais):
        # Filtra eventos curtos demais (ruído)
        if (f - i) < janela_minima_frames:
            continue

        segmento = dados[i:f]
        idx_max  = segmento.idxmax()

        picos.append({
            'pico':             len(picos) + 1,
            'inicio_frame':     int(i),
            'fim_frame':        int(f),
            'inicio_seg':       float(tempo[i]),
            'fim_seg':          float(tempo[f]),
            'duracao':          float(tempo[f] - tempo[i]),
            'amplitude_maxima': float(segmento.max()),
            'frame_pico':       int(idx_max),
        })

    return picos


def picos_para_dataframe(picos: list[dict]) -> pd.DataFrame:
    """
    Converte a lista de picos em um DataFrame organizado para exibição
    no dashboard ou uso nos modelos de AM.
    """
    if not picos:
        return pd.DataFrame()
    return pd.DataFrame(picos).set_index('pico')


# =============================================================================
# 4. MÉTRICAS BIOMECÂNICAS
# =============================================================================

def calcular_indice_simetria(picos_esq: list[dict], picos_dir: list[dict]) -> float:
    """
    Calcula o índice de simetria bilateral entre os membros inferiores.

    Baseado na razão entre as amplitudes médias de cada lado. Um valor de
    1.0 indica simetria perfeita; valores abaixo de 0.85 são clinicamente
    relevantes como assimetria.

    Parâmetros:
        picos_esq: Lista de picos do membro esquerdo.
        picos_dir: Lista de picos do membro direito.

    Retorna:
        Índice de simetria entre 0.0 e 1.0.
    """
    if not picos_esq or not picos_dir:
        return 0.0

    media_esq = np.mean([p['amplitude_maxima'] for p in picos_esq])
    media_dir = np.mean([p['amplitude_maxima'] for p in picos_dir])

    if max(media_esq, media_dir) == 0:
        return 0.0

    return float(min(media_esq, media_dir) / max(media_esq, media_dir))


def calcular_jerk(dados: pd.Series, fps: float) -> float:
    """
    Calcula o jerk médio quadrático (RMS) do sinal cinemático.

    O jerk é a derivada da aceleração (terceira derivada da posição), e
    quantifica a suavidade do movimento. Valores menores indicam movimentos
    mais fluidos e controlados — um biomarcador importante para avaliar
    a qualidade da marcha em pacientes neurofuncionais.

    Parâmetros:
        dados: Série de ângulos ao longo do tempo.
        fps: Taxa de quadros por segundo (necessária para escalar as derivadas).

    Retorna:
        Jerk RMS (graus/s³). Quanto menor, mais suave o movimento.
    """
    if len(dados) < 4:
        return 0.0

    dt = 1.0 / fps
    velocidade    = np.gradient(dados.values, dt)
    aceleracao    = np.gradient(velocidade, dt)
    jerk          = np.gradient(aceleracao, dt)

    return float(np.sqrt(np.mean(jerk ** 2)))


def calcular_consistencia_temporal(picos: list[dict]) -> dict:
    """
    Avalia a consistência temporal entre os obstáculos transpostos.

    Analisa se o paciente manteve um ritmo regular ao longo dos 10
    obstáculos, comparando as durações de cada pico.

    Parâmetros:
        picos: Lista de dicionários retornada por extrair_picos_marcha().

    Retorna:
        Dicionário com:
            - 'media_duracao': duração média dos picos (s)
            - 'desvio_duracao': desvio padrão das durações (s)
            - 'cv_duracao': coeficiente de variação (%) — quanto menor,
                            mais consistente o ritmo
            - 'n_picos': quantidade de picos detectados
    """
    if not picos:
        return {
            'media_duracao':  0.0,
            'desvio_duracao': 0.0,
            'cv_duracao':     0.0,
            'n_picos':        0,
        }

    duracoes = np.array([p['duracao'] for p in picos])
    media    = float(np.mean(duracoes))
    desvio   = float(np.std(duracoes))
    cv       = float((desvio / media * 100) if media > 0 else 0.0)

    return {
        'media_duracao':  round(media, 3),
        'desvio_duracao': round(desvio, 3),
        'cv_duracao':     round(cv, 2),
        'n_picos':        len(picos),
    }


# =============================================================================
# 5. RESUMO COMPLETO DA SESSÃO
# =============================================================================

def gerar_resumo_sessao(df: pd.DataFrame, fps: float, limiares: dict) -> dict:
    """
    Gera um resumo completo de todas as métricas biomecânicas de uma sessão.

    Este é o ponto de entrada principal para o dashboard e para os modelos
    de AM: recebe o DataFrame bruto e os limiares definidos pelo clínico,
    e retorna todas as métricas calculadas.

    O tornozelo é tratado como sinal de compensação postural: não extraímos
    picos dele, mas calculamos sua amplitude média e jerk para que o
    fisioterapeuta possa identificar se o paciente está compensando a falta
    de flexão de quadril/joelho com movimentos excessivos do tornozelo.

    Parâmetros:
        df: DataFrame carregado via carregar_arquivo().
        fps: FPS obtido via calcular_fps() — 28.0 para o Kinect.
        limiares: Dicionário com os limiares clínicos definidos pelo clínico:
                  {
                      'quadril': 45.0,   # graus mínimos de flexão de quadril
                      'joelho':  30.0,   # graus mínimos de flexão de joelho
                  }

    Retorna:
        Dicionário aninhado com todas as métricas da sessão.
    """
    limiar_quadril = limiares.get('quadril', 45.0)
    limiar_joelho  = limiares.get('joelho',  30.0)

    # --- Extração de picos (quadril e joelho — foco clínico) ---
    picos_quadril_esq = extrair_picos_marcha(df[COL_QUADRIL_ESQ], fps, limiar_quadril)
    picos_quadril_dir = extrair_picos_marcha(df[COL_QUADRIL_DIR], fps, limiar_quadril)
    picos_joelho_esq  = extrair_picos_marcha(df[COL_JOELHO_ESQ],  fps, limiar_joelho)
    picos_joelho_dir  = extrair_picos_marcha(df[COL_JOELHO_DIR],  fps, limiar_joelho)

    # --- Simetria bilateral ---
    simetria_quadril = calcular_indice_simetria(picos_quadril_esq, picos_quadril_dir)
    simetria_joelho  = calcular_indice_simetria(picos_joelho_esq,  picos_joelho_dir)

    # --- Jerk — suavidade (quadril e joelho) ---
    jerk_quadril_esq = calcular_jerk(df[COL_QUADRIL_ESQ], fps)
    jerk_quadril_dir = calcular_jerk(df[COL_QUADRIL_DIR], fps)
    jerk_joelho_esq  = calcular_jerk(df[COL_JOELHO_ESQ],  fps)
    jerk_joelho_dir  = calcular_jerk(df[COL_JOELHO_DIR],  fps)

    # --- Consistência temporal dos picos ---
    consistencia_quadril_esq = calcular_consistencia_temporal(picos_quadril_esq)
    consistencia_quadril_dir = calcular_consistencia_temporal(picos_quadril_dir)
    consistencia_joelho_esq  = calcular_consistencia_temporal(picos_joelho_esq)
    consistencia_joelho_dir  = calcular_consistencia_temporal(picos_joelho_dir)

    # --- Tornozelo: sinal de compensação (amplitude média + jerk, sem picos) ---
    compensacao_tornozelo = {
        'amplitude_media_esq': float(df[COL_TORNOZELO_ESQ].mean()),
        'amplitude_media_dir': float(df[COL_TORNOZELO_DIR].mean()),
        'amplitude_max_esq':   float(df[COL_TORNOZELO_ESQ].max()),
        'amplitude_max_dir':   float(df[COL_TORNOZELO_DIR].max()),
        'jerk_esq':            calcular_jerk(df[COL_TORNOZELO_ESQ], fps),
        'jerk_dir':            calcular_jerk(df[COL_TORNOZELO_DIR], fps),
    }

    return {
        'picos': {
            'quadril_esq': picos_quadril_esq,
            'quadril_dir': picos_quadril_dir,
            'joelho_esq':  picos_joelho_esq,
            'joelho_dir':  picos_joelho_dir,
        },
        'simetria': {
            'quadril': simetria_quadril,
            'joelho':  simetria_joelho,
        },
        'jerk': {
            'quadril_esq': jerk_quadril_esq,
            'quadril_dir': jerk_quadril_dir,
            'joelho_esq':  jerk_joelho_esq,
            'joelho_dir':  jerk_joelho_dir,
        },
        'consistencia': {
            'quadril_esq': consistencia_quadril_esq,
            'quadril_dir': consistencia_quadril_dir,
            'joelho_esq':  consistencia_joelho_esq,
            'joelho_dir':  consistencia_joelho_dir,
        },
        'compensacao_tornozelo': compensacao_tornozelo,
        'n_frames': len(df),
        'fps':      fps,
        'limiares': limiares,
    }