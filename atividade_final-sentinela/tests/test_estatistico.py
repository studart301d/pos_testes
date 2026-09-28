"""Bloco B: comparacao v1 x v2 com incerteza, classe rara, limiar, calibracao e drift."""
import numpy as np
import pytest
from scipy import stats
import sentinela as sn
from conftest import recall


@pytest.fixture(scope="module")
def probas(X_teste):
    return {v: sn.modelo.carregar(v).prever_proba(X_teste) for v in ("v1", "v2")}


def pr_auc(y, p):
    """Average precision (area sob a curva precisao x recall)."""
    ordem = np.argsort(-p)
    y_ord = y[ordem]
    vp = np.cumsum(y_ord)
    precisao = vp / np.arange(1, len(y) + 1)
    return float((precisao * y_ord).sum() / y_ord.sum())


def test_acuracia_nao_serve_para_classe_rara(y_teste):
    """Um modelo que nunca preve falha ja tem acuracia alta e recall zero."""
    nunca = np.zeros_like(y_teste)
    m = sn.avaliacao.metricas(y_teste, nunca)
    print(f"\n[classe rara] prevalencia={y_teste.mean():.3f} acuracia_trivial={m['acuracia']:.3f} recall={m['recall']:.1f}")
    assert m["acuracia"] > 0.8 and m["recall"] == 0


@pytest.mark.parametrize("versao", ["v1", "v2"])
def test_pr_auc_acima_da_prevalencia(probas, y_teste, versao):
    ap = pr_auc(y_teste, probas[versao])
    print(f"\n[PR-AUC] {versao}={ap:.3f} (prevalencia={y_teste.mean():.3f})")
    assert ap > y_teste.mean()


def test_mcnemar_v1_vs_v2_diferenca_significativa(probas, y_teste):
    a1 = (probas["v1"] >= 0.5) == y_teste
    a2 = (probas["v2"] >= 0.5) == y_teste
    b, c = int((a1 & ~a2).sum()), int((~a1 & a2).sum())
    p = stats.binomtest(min(b, c), b + c).pvalue
    print(f"\n[McNemar] so v1 acerta={b} so v2 acerta={c} p={p:.2e}")
    assert p < 0.05


def test_DEFEITO_v2_nao_pode_perder_recall_para_v1(probas, y_teste, rng):
    """Gate de promocao. Na planta, falso negativo e motor queimado. A v2 so
    pode substituir a v1 se nao reduzir o recall. IC 95% bootstrap da
    diferenca recall(v2) - recall(v1)."""
    p1, p2 = (probas["v1"] >= 0.5).astype(int), (probas["v2"] >= 0.5).astype(int)
    n, difs = len(y_teste), []
    for _ in range(2000):
        i = rng.integers(0, n, n)
        difs.append(recall(y_teste[i], p2[i]) - recall(y_teste[i], p1[i]))
    lo, hi = np.percentile(difs, [2.5, 97.5])
    fn1 = int(((y_teste == 1) & (p1 == 0)).sum())
    fn2 = int(((y_teste == 1) & (p2 == 0)).sum())
    print(f"\n[bootstrap] recall v2-v1 IC95%=[{lo:.3f}, {hi:.3f}] | falsos negativos v1={fn1} v2={fn2}")
    assert lo >= 0, "v2 reduz o recall de forma significativa; promocao deve ser bloqueada"


def test_varredura_de_limiar(probas, y_teste):
    """0.5 nao e sagrado. Mostra o trade-off e verifica que existe limiar
    que leva a v2 a recall >= 0.90."""
    print("\n[limiar] versao limiar precisao recall")
    melhor = None
    for lim in np.arange(0.1, 0.91, 0.1):
        for v in ("v1", "v2"):
            m = sn.avaliacao.metricas(y_teste, (probas[v] >= lim).astype(int))
            print(f"  {v} {lim:.1f} {m['precisao']:.3f} {m['recall']:.3f}")
            if v == "v2" and m["recall"] >= 0.90:
                melhor = lim
    assert melhor is not None


def ece(y, p, bins=10):
    arestas = np.linspace(0, 1, bins + 1)
    total = 0.0
    for a, b in zip(arestas[:-1], arestas[1:]):
        m = (p >= a) & (p < b) if b < 1 else (p >= a) & (p <= b)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return total


@pytest.mark.parametrize("versao", ["v1", "v2"])
def test_calibracao_das_probabilidades(probas, y_teste, versao):
    """A probabilidade media prevista em cada faixa deve bater com a frequencia
    de falha observada. Tolerancia: ECE <= 0.10."""
    valor = ece(y_teste, probas[versao])
    print(f"\n[calibracao] {versao} ECE={valor:.3f}")
    assert valor <= 0.10, f"{versao} descalibrado: ECE={valor:.3f}"


def test_medias_teste_producao_parecem_iguais(bruto_teste, bruto_producao):
    """Comparar medias sugere que nada mudou."""
    d = abs(bruto_teste["temperatura_c"].mean() - bruto_producao["temperatura_c"].mean())
    print(f"\n[medias] diferenca de temperatura media={d:.2f} C")
    assert d < 1.5


@pytest.mark.parametrize("coluna", ["temperatura_c", "vibracao_rms", "corrente_a"])
def test_DEFEITO_drift_entre_teste_e_producao(bruto_teste, bruto_producao, coluna):
    """Kolmogorov-Smirnov compara a distribuicao inteira, nao so a media."""
    t, p = bruto_teste[coluna].dropna(), bruto_producao[coluna].dropna()
    ks = stats.ks_2samp(t, p)
    print(f"\n[drift] {coluna} KS={ks.statistic:.3f} p={ks.pvalue:.2e} "
          f"desvio teste={t.std():.2f} producao={p.std():.2f}")
    assert ks.pvalue > 0.01, f"drift em {coluna}: distribuicao de producao difere do teste"
