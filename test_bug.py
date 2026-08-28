"""Prova de que o property test derruba a versao bugada.

O invariante de deslocamento (somar constante nao muda os outliers) vale
para a funcao correta, mas NAO vale para a versao que esquece de centralizar.
Este teste EXISTE PARA FALHAR contra a funcao bugada.
"""
from hypothesis import given, strategies as st
from outliers_bugado import detectar_outliers_bugado

numeros = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
listas = st.lists(numeros, min_size=1, max_size=50)


@given(listas, st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False))
def test_bug_quebra_invariante_de_deslocamento(dados, deslocamento):
    base = detectar_outliers_bugado(dados)
    deslocado = detectar_outliers_bugado([x + deslocamento for x in dados])
    assert base == deslocado
