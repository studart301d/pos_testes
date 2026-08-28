import pytest
from outliers import detectar_outliers


def test_detecta_outlier_obvio():
    # Muitos valores em torno de 10 e um 100 gritante no fim.
    # Com base grande, o z-score do 100 passa de 3 com folga.
    dados = [10, 11, 9, 10, 12, 8, 10, 9, 11, 10, 100]
    assert detectar_outliers(dados) == [10]  # indice do 100


def test_sem_outliers_retorna_lista_vazia():
    dados = [10, 11, 9, 10, 12]
    assert detectar_outliers(dados) == []


def test_lista_vazia():
    assert detectar_outliers([]) == []


def test_todos_valores_iguais_sem_outlier():
    # desvio-padrao zero: nao ha' variacao, logo nao ha' outlier
    assert detectar_outliers([5, 5, 5, 5]) == []
