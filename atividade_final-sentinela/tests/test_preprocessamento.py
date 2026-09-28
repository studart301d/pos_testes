"""Bloco A: preprocessamento.limpar e contrato de dados da ficha tecnica."""
import pandas as pd
import pytest
import sentinela as sn
from conftest import FAIXAS, PRESSAO_BAR


def test_limpar_nao_modifica_a_entrada(bruto_teste):
    copia = bruto_teste.copy()
    sn.preprocessamento.limpar(bruto_teste)
    pd.testing.assert_frame_equal(bruto_teste, copia)


def test_limpar_converte_tipos(limpo_teste):
    assert pd.api.types.is_datetime64_any_dtype(limpo_teste["timestamp"])
    assert pd.api.types.is_integer_dtype(limpo_teste["turno"])
    assert pd.api.types.is_integer_dtype(limpo_teste["idade_equipamento_meses"])


def test_limpar_nao_deixa_valores_faltantes(limpo_teste):
    assert limpo_teste.isna().sum().sum() == 0


def test_limpar_normaliza_rotulos(bruto_teste):
    amostra = bruto_teste.head(3).copy()
    amostra["id_maquina"] = ["  m01", "m02 ", " M03 "]
    amostra["id_operador"] = [" op-01", "OP-02 ", "op-03"]
    limpo = sn.preprocessamento.limpar(amostra)
    assert list(limpo["id_maquina"]) == ["M01", "M02", "M03"]
    assert list(limpo["id_operador"]) == ["OP-01", "OP-02", "OP-03"]


@pytest.mark.parametrize("coluna", ["temperatura_c", "corrente_a", "idade_equipamento_meses"])
@pytest.mark.parametrize("conjunto", ["treino", "teste", "producao"])
def test_dados_crus_respeitam_faixa_da_ficha(conjunto, coluna):
    dados = sn.dados.carregar(conjunto)
    lo, hi = FAIXAS[coluna]
    assert dados[coluna].between(lo, hi).all()


def test_DEFEITO_vibracao_imputada_fora_da_faixa_fisica(limpo_teste):
    """limpar() preenche o dropout do sensor com 0.0 mm/s, valor impossivel
    para um motor girando (faixa de operacao 1.2 a 8.0 mm/s)."""
    lo, hi = FAIXAS["vibracao_rms"]
    fora = (~limpo_teste["vibracao_rms"].between(lo, hi)).sum()
    assert fora == 0, f"{fora} leituras de vibracao fora da faixa fisica apos limpar()"


def test_DEFEITO_pressao_em_psi_nao_convertida(limpo_teste):
    """A ficha declara que alguns CLPs reportam em psi. limpar() normaliza o
    rotulo da unidade, mas nunca converte o valor para bar."""
    lo, hi = PRESSAO_BAR
    fora = (~limpo_teste["pressao"].between(lo, hi)).sum()
    assert fora == 0, f"{fora} leituras de pressao fora de 3.0-4.5 bar (valores em psi)"


def test_DEFEITO_leitura_fora_da_ficha_passa_sem_alerta(bruto_teste):
    """Caso-limite: uma temperatura de 150 C, fora da faixa 45-95 C, deveria
    ser rejeitada ou sinalizada. O pipeline a aceita e gera decisao."""
    lote = bruto_teste.head(30).copy()
    lote.loc[lote.index[-1], "temperatura_c"] = 150.0
    with pytest.raises(ValueError):
        sn.pipeline.executar(lote, versao="v1")
