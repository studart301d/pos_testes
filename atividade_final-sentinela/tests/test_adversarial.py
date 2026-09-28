"""Bloco C: perturbacoes abaixo do ruido, contrafactuais e casos-limite."""
import numpy as np
import pytest
import sentinela as sn
from conftest import RUIDO_TEMPERATURA


@pytest.mark.parametrize("versao", ["v1", "v2"])
def test_perturbacao_menor_que_ruido_do_sensor(limpo_sem_rotulo, versao):
    """Perturbar a temperatura em ate metade do ruido do sensor (+-0.5 C) nao
    deveria alterar mais de 1% das decisoes."""
    rng = np.random.default_rng(7)
    modelo = sn.modelo.carregar(versao)
    base = modelo.prever(sn.features.construir(limpo_sem_rotulo))
    perturbado = limpo_sem_rotulo.copy()
    amplitude = RUIDO_TEMPERATURA / 2
    perturbado["temperatura_c"] += rng.uniform(-amplitude, amplitude, len(perturbado))
    taxa = (modelo.prever(sn.features.construir(perturbado)) != base).mean()
    print(f"\n[ruido] {versao} +-{amplitude} C altera {taxa:.2%} das decisoes")
    assert taxa <= 0.01


def test_DEFEITO_contrafactual_operador_muda_decisao(modelo, X_teste):
    """Quem esta no turno nao altera o estado fisico do motor. Trocar a marca
    operador_senior (OP-07) nao deveria mudar nenhuma decisao."""
    alterado = X_teste.copy()
    alterado["operador_senior"] = 1 - alterado["operador_senior"]
    taxa = (modelo.prever(X_teste) != modelo.prever(alterado)).mean()
    print(f"\n[contrafactual operador] {modelo.versao} {taxa:.2%} das decisoes mudam")
    assert taxa == 0


def test_lote_com_uma_linha(bruto_teste):
    saida = sn.pipeline.executar(bruto_teste.head(1), versao="v1")
    assert len(saida) == 1 and saida["predicao"].isin([0, 1]).all()


def test_lote_com_um_motor_so(bruto_teste):
    lote = bruto_teste[bruto_teste["id_maquina"] == "M05"]
    saida = sn.pipeline.executar(lote, versao="v1")
    assert len(saida) == len(lote)


def test_leituras_repetidas_dao_decisoes_estaveis(bruto_teste):
    """Motor com leitura constante por 24 h: a decisao nao pode oscilar."""
    lote = bruto_teste[bruto_teste["id_maquina"] == "M02"].head(24).copy()
    for col in ["temperatura_c", "vibracao_rms", "corrente_a", "pressao", "rpm"]:
        lote[col] = lote[col].iloc[0]
    lote["unidade_pressao"] = lote["unidade_pressao"].iloc[0]
    saida = sn.pipeline.executar(lote, versao="v1")
    assert saida["predicao"].iloc[6:].nunique() == 1


def test_conteudo_dos_artefatos_bate_com_manifest_ignorando_quebra_de_linha():
    """Os dados nao foram adulterados: normalizando LF para CRLF, todo hash bate."""
    import hashlib
    for rel, esperado in sn.artefatos.ler_manifest()["arquivos"].items():
        b = (sn.artefatos.RAIZ / rel).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        crlf_ok = hashlib.sha256(b).hexdigest() == esperado
        original_ok = sn.artefatos.sha256(sn.artefatos.RAIZ / rel) == esperado
        assert crlf_ok or original_ok, rel


def test_DEFEITO_verificar_manifest_depende_do_sistema_operacional():
    """O manifest foi gerado com arquivos em CRLF (Windows). O repositorio os
    guarda em LF, entao a checagem de integridade reprova em Linux/WSL/Colab
    mesmo com conteudo identico."""
    problemas = sn.artefatos.verificar_manifest()
    assert problemas == [], f"integridade reprovada: {problemas}"
