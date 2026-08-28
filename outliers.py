"""Deteccao de outliers por Z-score."""
import math
import statistics
from typing import Sequence


def detectar_outliers(dados: Sequence[float], limiar: float = 3.0) -> list[int]:
    """Retorna os indices dos outliers em `dados` pelo criterio de Z-score.

    Um ponto e' outlier quando |(x - media) / desvio_padrao| >= limiar.

    Decisoes de design:
    - Lista vazia -> retorna [] (sem erro).
    - Desvio-padrao igual a 0 (todos os valores iguais) -> retorna [],
      pois nao ha' variacao e a divisao por zero nao e' definida.
    - Valores NaN ou infinitos -> ValueError, para nao propagar
      resultado silenciosamente incorreto para o pipeline.
    - Limiar deve ser > 0, senao ValueError.

    Args:
        dados: sequencia de numeros.
        limiar: numero de desvios-padrao a partir do qual um ponto e' outlier.

    Returns:
        Lista de indices (ordem crescente) dos pontos considerados outliers.
    """
    if limiar <= 0:
        raise ValueError("limiar deve ser maior que zero")

    if any(math.isnan(x) or math.isinf(x) for x in dados):
        raise ValueError("dados contem NaN ou infinito")

    if len(dados) == 0:
        return []

    media = statistics.mean(dados)
    desvio = statistics.pstdev(dados)

    if desvio == 0:
        return []

    return [
        i for i, valor in enumerate(dados)
        if abs(valor - media) / desvio >= limiar
    ]
