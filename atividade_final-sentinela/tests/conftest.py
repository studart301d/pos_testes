"""Fixtures compartilhadas. A suite testa o pacote `sentinela` de fora, sem edita-lo."""
import numpy as np
import pytest
import sentinela as sn

# Ficha tecnica dos sensores (README do Sentinela)
FAIXAS = {
    "temperatura_c": (45.0, 95.0),
    "vibracao_rms": (1.2, 8.0),
    "corrente_a": (12.0, 30.0),
    "idade_equipamento_meses": (6, 180),
}
PRESSAO_BAR = (3.0, 4.5)
RUIDO_TEMPERATURA = 1.0


@pytest.fixture(scope="session")
def bruto_teste():
    return sn.dados.carregar("teste")


@pytest.fixture(scope="session")
def bruto_producao():
    return sn.dados.carregar("producao")


@pytest.fixture(scope="session")
def limpo_teste(bruto_teste):
    return sn.preprocessamento.limpar(bruto_teste)


@pytest.fixture(scope="session")
def limpo_sem_rotulo(bruto_teste):
    """Lote como chega em producao: sem a coluna de rotulo."""
    return sn.preprocessamento.limpar(bruto_teste.drop(columns="falha_72h"))


@pytest.fixture(scope="session")
def X_teste(limpo_sem_rotulo):
    return sn.features.construir(limpo_sem_rotulo)


@pytest.fixture(scope="session")
def y_teste(limpo_teste):
    return limpo_teste["falha_72h"].to_numpy()


@pytest.fixture(scope="session", params=["v1", "v2"])
def modelo(request):
    return sn.modelo.carregar(request.param)


def recall(y, p):
    return sn.avaliacao.metricas(y, p)["recall"]


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(2026)
