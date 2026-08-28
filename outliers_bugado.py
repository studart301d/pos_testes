"""Versao PROPOSITALMENTE BUGADA para demonstrar que o property test pega o bug.

BUG: esqueceu de subtrair a media (nao centralizou o z-score).
Usa abs(valor) / desvio em vez de abs(valor - media) / desvio.
"""
import math
import statistics
from typing import Sequence


def detectar_outliers_bugado(dados: Sequence[float], limiar: float = 3.0) -> list[int]:
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

    # BUG: deveria ser abs(valor - media) / desvio
    return [
        i for i, valor in enumerate(dados)
        if abs(valor) / desvio >= limiar
    ]
