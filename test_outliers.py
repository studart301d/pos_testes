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


def test_nan_levanta_erro():
    with pytest.raises(ValueError):
        detectar_outliers([1.0, 2.0, float("nan"), 3.0])


def test_inf_levanta_erro():
    with pytest.raises(ValueError):
        detectar_outliers([1.0, 2.0, float("inf")])


def test_limiar_invalido_levanta_erro():
    with pytest.raises(ValueError):
        detectar_outliers([1, 2, 3], limiar=0)


# ---------- Testes de propriedade (hypothesis) ----------
from hypothesis import given, strategies as st

# Numeros "bem-comportados": finitos, sem NaN/inf, magnitude controlada
numeros = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
listas = st.lists(numeros, min_size=1, max_size=50)


@given(listas)
def test_indices_sempre_validos_e_unicos(dados):
    resultado = detectar_outliers(dados)
    assert all(0 <= i < len(dados) for i in resultado)
    assert len(resultado) == len(set(resultado))


@given(listas, st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False))
def test_invariante_a_deslocamento(dados, deslocamento):
    # Somar uma constante a todos os pontos nao muda quem e' outlier
    base = detectar_outliers(dados)
    deslocado = detectar_outliers([x + deslocamento for x in dados])
    assert base == deslocado


@given(listas, st.floats(min_value=0.001, max_value=1000, allow_nan=False, allow_infinity=False))
def test_invariante_a_escala(dados, fator):
    # Multiplicar todos os pontos por um fator positivo nao muda quem e' outlier
    base = detectar_outliers(dados)
    escalado = detectar_outliers([x * fator for x in dados])
    assert base == escalado
