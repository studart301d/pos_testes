"""Suite: validacao de dados (Aula 3) + testes estatisticos e fairness (Aula 4).

Contexto: aprovacao de credito. Atributo sensivel = genero.
Comparamos dois modelos: A (legado, enviesado) e B (revisado).
"""
import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check
import pytest
from scipy import stats

RNG = np.random.default_rng(7)


@pytest.fixture(scope="module")
def df():
    return pd.read_csv("dados_credito.csv")


# ==================== 1. VALIDACAO DE DADOS ====================
schema = DataFrameSchema({
    "id_cliente":     Column(int, unique=True, nullable=False),
    "genero":         Column(str, Check.isin(["F", "M"]), nullable=False),
    "idade":          Column(int, Check.in_range(18, 100), nullable=False),
    "renda":          Column(float, Check.greater_than(0), nullable=False),
    "score_credito":  Column(float, Check.in_range(300, 850), nullable=False),
    "aprovado_real":  Column(int, Check.isin([0, 1]), nullable=False),
    "pred_modelo_a":  Column(int, Check.isin([0, 1]), nullable=False),
    "pred_modelo_b":  Column(int, Check.isin([0, 1]), nullable=False),
})


def test_schema_valido(df):
    """Schema: colunas esperadas, tipos corretos e dominio de valores."""
    schema.validate(df)


def test_completude(df):
    """Completude: nenhuma coluna pode ter valor faltante."""
    assert df.isnull().sum().sum() == 0


def test_unicidade_id(df):
    """Unicidade: id_cliente e' chave, nao pode repetir."""
    assert df["id_cliente"].is_unique


def test_grupos_representados(df):
    """Fairness so' e' valida se os dois grupos tem amostra suficiente."""
    contagem = df["genero"].value_counts()
    assert set(contagem.index) == {"F", "M"}
    assert contagem.min() >= 30


def test_schema_rejeita_dado_corrompido(df):
    """Prova de que a validacao pega dado fora da faixa."""
    ruim = df.copy()
    ruim.loc[0, "score_credito"] = 9999.0
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(ruim)


# ==================== 2. TESTES ESTATISTICOS ====================
def bootstrap_ic(valores, n_boot=2000, alpha=0.05):
    """IC percentil por bootstrap para a media de `valores`."""
    valores = np.asarray(valores)
    medias = [RNG.choice(valores, size=len(valores), replace=True).mean()
              for _ in range(n_boot)]
    return np.percentile(medias, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def test_ic_bootstrap_acuracia(df):
    """IC 95% bootstrap da acuracia: a metrica e' confiavel ou e' ruido?"""
    acertos = (df["pred_modelo_a"] == df["aprovado_real"]).astype(int)
    low, high = bootstrap_ic(acertos)
    print(f"\n[IC bootstrap] Acuracia modelo A: {acertos.mean():.3f} "
          f"| IC95% = [{low:.3f}, {high:.3f}]")
    assert low > 0.5, "acuracia nao e' melhor que o acaso"
    assert high - low < 0.15, "IC muito largo: amostra insuficiente"


def test_comparacao_modelo_a_vs_b(df):
    """Comparacao estatistica: a diferenca de acuracia A vs B e' real ou ruido?"""
    acc_a = (df["pred_modelo_a"] == df["aprovado_real"]).astype(int)
    acc_b = (df["pred_modelo_b"] == df["aprovado_real"]).astype(int)
    stat, pval = stats.ttest_rel(acc_a, acc_b)
    print(f"\n[Comparacao A vs B] acc_A={acc_a.mean():.3f} acc_B={acc_b.mean():.3f} "
          f"| p-valor={pval:.4f}")
    assert np.isfinite(pval)


# ==================== 3. FAIRNESS ====================
def taxa_aprovacao(df, col_pred, grupo):
    return df.loc[df["genero"] == grupo, col_pred].mean()


@pytest.mark.xfail(strict=True,
                   reason="Modelo A e' enviesado: viola a regra dos 80%. Falha esperada e documentada.")
def test_paridade_demografica_modelo_a(df):
    """Paridade demografica (regra dos 80%) no modelo legado."""
    tf = taxa_aprovacao(df, "pred_modelo_a", "F")
    tm = taxa_aprovacao(df, "pred_modelo_a", "M")
    ratio = min(tf, tm) / max(tf, tm)
    print(f"\n[Fairness A] taxa F={tf:.3f} M={tm:.3f} "
          f"| dif={abs(tf-tm):.3f} | ratio={ratio:.3f}")
    assert ratio >= 0.80, f"regra dos 80% violada: ratio={ratio:.3f}"


def test_paridade_demografica_modelo_b(df):
    """Paridade demografica (regra dos 80%) no modelo revisado."""
    tf = taxa_aprovacao(df, "pred_modelo_b", "F")
    tm = taxa_aprovacao(df, "pred_modelo_b", "M")
    ratio = min(tf, tm) / max(tf, tm)
    print(f"\n[Fairness B] taxa F={tf:.3f} M={tm:.3f} "
          f"| dif={abs(tf-tm):.3f} | ratio={ratio:.3f}")
    assert ratio >= 0.80, f"regra dos 80% violada: ratio={ratio:.3f}"


def test_diferenca_de_grupo_e_significativa(df):
    """Qui-quadrado: a disparidade do modelo A e' estatistica ou acaso?"""
    tab = pd.crosstab(df["genero"], df["pred_modelo_a"])
    chi2, pval, _, _ = stats.chi2_contingency(tab)
    print(f"\n[Qui-quadrado modelo A] chi2={chi2:.2f} p-valor={pval:.6f}")
    assert np.isfinite(pval)
