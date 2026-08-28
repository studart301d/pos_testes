"""Prova de que o property test derruba a versao bugada.

O invariante de deslocamento (somar uma constante a todos os pontos nao muda
quem e' outlier) vale para a funcao correta, mas NAO vale para a versao que
esquece de centralizar o z-score. Este teste EXISTE PARA FALHAR contra a
versao bugada -- e' a prova de que o property test pegaria o bug em producao.

Nota: o bug e' invariante a escala (multiplicar tudo por um fator nao muda
abs(v*f)/(desvio*f)), entao so' o invariante de DESLOCAMENTO o expoe. Alem
disso, o contraexemplo e' relativamente raro, entao ampliamos max_examples
para o hypothesis explorar o espaco o suficiente e encontra-lo de forma
confiavel.
"""
from hypothesis import given, strategies as st, settings
from outliers_bugado import detectar_outliers_bugado

numeros = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
listas = st.lists(numeros, min_size=1, max_size=50)


@given(listas, st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False))
@settings(max_examples=500)
def test_bug_quebra_invariante_de_deslocamento(dados, deslocamento):
    base = detectar_outliers_bugado(dados)
    deslocado = detectar_outliers_bugado([x + deslocamento for x in dados])
    assert base == deslocado
