"""Bloco A: features.construir e contrato do modelo."""
import numpy as np
import pandas as pd
import pytest
import sentinela as sn


def test_features_na_ordem_canonica_e_indice_preservado(limpo_sem_rotulo, X_teste):
    assert tuple(X_teste.columns) == sn.features.ORDEM_FEATURES
    assert X_teste.index.equals(limpo_sem_rotulo.index)


def test_features_independem_da_ordem_das_linhas(limpo_sem_rotulo, X_teste):
    embaralhado = limpo_sem_rotulo.sample(frac=1, random_state=0)
    X_emb = sn.features.construir(embaralhado)
    pd.testing.assert_frame_equal(X_emb.loc[X_teste.index], X_teste)


def test_features_sao_deterministicas(limpo_sem_rotulo, X_teste):
    pd.testing.assert_frame_equal(sn.features.construir(limpo_sem_rotulo), X_teste)


def test_construir_rejeita_colunas_ausentes(limpo_sem_rotulo):
    with pytest.raises(ValueError):
        sn.features.construir(limpo_sem_rotulo.drop(columns="temperatura_c"))


@pytest.mark.parametrize("posicao", [30, 80, 140])
def test_DEFEITO_feature_usa_leitura_do_futuro(limpo_sem_rotulo, posicao):
    """A feature do instante t so pode depender de leituras ate t. Alteramos
    uma leitura 2 horas no futuro e verificamos se alguma feature de t mudou."""
    motor = limpo_sem_rotulo[limpo_sem_rotulo["id_maquina"] == "M01"].sort_values("timestamp")
    idx_t, idx_futuro = motor.index[posicao], motor.index[posicao + 2]
    alterado = limpo_sem_rotulo.copy()
    alterado.loc[idx_futuro, "temperatura_c"] += 20.0
    antes = sn.features.construir(limpo_sem_rotulo).loc[idx_t]
    depois = sn.features.construir(alterado).loc[idx_t]
    mudou = (antes - depois).abs()
    assert (mudou < 1e-9).all(), f"features de t alteradas pelo futuro: {mudou[mudou > 1e-9].to_dict()}"


def test_DEFEITO_features_dependem_do_rotulo(bruto_teste):
    """maquina_risco e recalculado a partir de falha_72h quando o lote vem
    rotulado. A mesma leitura gera features diferentes com e sem rotulo."""
    com = sn.features.construir(sn.preprocessamento.limpar(bruto_teste))
    sem = sn.features.construir(sn.preprocessamento.limpar(bruto_teste.drop(columns="falha_72h")))
    diferentes = (~np.isclose(com["maquina_risco"], sem["maquina_risco"])).mean()
    assert diferentes == 0, f"{diferentes:.1%} das linhas mudam maquina_risco so pela presenca do rotulo"


def test_prever_proba_devolve_probabilidades(modelo, X_teste):
    proba = modelo.prever_proba(X_teste)
    assert proba.shape == (len(X_teste),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_modelo_rejeita_nan(modelo, X_teste):
    X = X_teste.head(5).copy()
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        modelo.prever_proba(X)


def test_DEFEITO_prever_registro_ignora_nomes_das_chaves(modelo, X_teste):
    """prever_registro usa registro.values() e ignora as chaves. Um dicionario
    com as mesmas features em outra ordem gera decisao diferente, sem erro."""
    amostra = X_teste.head(300)
    certo = np.array([modelo.prever_registro(r) for r in amostra.to_dict("records")])
    invertido = np.array([modelo.prever_registro(dict(reversed(list(r.items()))))
                          for r in amostra.to_dict("records")])
    taxa = (certo != invertido).mean()
    assert taxa == 0, f"{taxa:.1%} das decisoes mudam so pela ordem das chaves"
