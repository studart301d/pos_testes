import math
import pytest
from outliers import detectar_outliers


def test_detecta_outlier_obvio():
    # 100 e' claramente um outlier no meio de valores pequenos
    dados = [10, 11, 9, 10, 12, 8, 100]
    assert detectar_outliers(dados) == [6]  # indice do 100


def test_sem_outliers_retorna_lista_vazia():
    dados = [10, 11, 9, 10, 12]
    assert detectar_outliers(dados) == []


def test_lista_vazia():
    assert detectar_outliers([]) == []
