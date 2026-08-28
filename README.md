# Detecção de Outliers por Z-score — TDD e Testes de Propriedade

Implementação de uma função de detecção de outliers por **Z-score** para pipelines de Machine Learning, desenvolvida com **TDD** (Test-Driven Development) e validada com **testes de propriedade** usando a biblioteca `hypothesis`.

**Autor:** Gabriel Moraes Ramos Studart
**Disciplina:** Orquestração de Workflows

---

## Função escolhida

A função `detectar_outliers` recebe uma sequência de números e um limiar, e retorna os **índices** dos pontos cujo Z-score (distância da média em desvios-padrão) atinge ou ultrapassa o limiar. Optei por retornar índices, e não valores, para que o pipeline consiga rastrear exatamente qual posição do lote é anômala mesmo quando há valores repetidos.

```python
def detectar_outliers(dados, limiar=3.0):
    if limiar <= 0:
        raise ValueError("limiar deve ser maior que zero")
    if any(math.isnan(x) or math.isinf(x) for x in dados):
        raise ValueError("dados contem NaN ou infinito")
    if len(dados) == 0:
        return []

    media = statistics.mean(dados)
    desvio = statistics.pstdev(dados)
    if desvio == 0:
        return []

    return [
        i for i, valor in enumerate(dados)
        if abs(valor - media) / desvio >= limiar
    ]
```

---

## Como rodar

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install pytest hypothesis

# Suite principal (funcao correta): deve passar 100%
pytest test_outliers.py -v

# Demonstracao do bug (PROPOSITALMENTE falha, ver secao 3)
pytest test_bug.py -v
```

> **Nota:** `test_bug.py` foi feito para **falhar** de propósito. Ele aponta um teste de propriedade contra uma versão bugada da função, provando que o teste pegaria o bug. A suíte que valida a função correta é a `test_outliers.py`.

---

## 1. Ciclo TDD (Red → Green → Refactor)

O desenvolvimento seguiu TDD, com **um commit por fase**. O histórico do Git é a evidência:

### Red — o teste falha primeiro

Escrevi os testes antes da função. Com a função ainda vazia (retornando `None`), a suíte falha, como esperado. Essa falha inicial garante que o teste realmente exercita o comportamento.

### Green — código mínimo que faz passar

Implementei a lógica do Z-score com o suficiente para passar. Nessa fase apareceu uma descoberta: meu primeiro exemplo de teste estava estatisticamente errado. Em uma amostra pequena, um valor extremo infla a própria média e o desvio-padrão a ponto de o seu Z-score não passar do limiar (o outlier se **mascara**). A função estava correta; o teste é que fazia uma afirmação falsa. Corrigi o teste e o ciclo fechou em verde.

### Refactor — melhorar sem mudar o comportamento

Com os testes verdes, refatorei: docstring, type hints e tratamento explícito de NaN, infinito e limiar inválido. Cada novo comportamento ganhou seu teste. A suíte permaneceu verde.

---

## 2. Testes de propriedade (hypothesis)

Além dos exemplos pontuais, escrevi três testes de propriedade. Em vez de um caso fixo, o `hypothesis` gera centenas de entradas aleatórias e verifica se um **invariante** se mantém para toda entrada válida.

| Invariante | O que garante | Bug que pega |
|---|---|---|
| Índices válidos e únicos | todo índice existe em `dados` e não se repete | erros de indexação |
| Invariância a deslocamento | somar constante a todos não muda os outliers | esquecer de centralizar |
| Invariância a escala positiva | multiplicar por fator > 0 não muda os outliers | erro no fator de escala |

```python
numeros = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
listas = st.lists(numeros, min_size=1, max_size=50)

@given(listas)
def test_indices_sempre_validos_e_unicos(dados):
    resultado = detectar_outliers(dados)
    assert all(0 <= i < len(dados) for i in resultado)
    assert len(resultado) == len(set(resultado))

@given(listas, st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False))
def test_invariante_a_deslocamento(dados, deslocamento):
    base = detectar_outliers(dados)
    deslocado = detectar_outliers([x + deslocamento for x in dados])
    assert base == deslocado
```

Suíte completa (exemplos + propriedade) verde:

---

## 3. Prova de que a propriedade pega um bug real

Criei uma versão propositalmente **bugada**: esquecer de centralizar o Z-score, usando `abs(valor) / desvio` em vez de `abs(valor - media) / desvio`.

```python
# BUG: deveria ser abs(valor - media) / desvio
return [
    i for i, valor in enumerate(dados)
    if abs(valor) / desvio >= limiar
]
```

Apontei o invariante de deslocamento contra essa versão. O `hypothesis` encontrou um contraexemplo e o teste **falhou**, como deveria:

**Interpretação:** em `[0.0, 1.0]` nenhum ponto é outlier. Depois de somar `1.0` em todos, viram `[1.0, 2.0]` e a versão bugada marca o `2.0` como outlier, porque compara o valor absoluto direto contra o desvio, sem descontar a média. O resultado passa a depender de onde os dados estão na reta, o que está errado. **Este teste teria barrado o bug antes da produção.**

Detalhe técnico: esse bug é *invariante à escala* (multiplicar por um fator não altera `abs(v·f)/(desvio·f)`), então só o invariante de **deslocamento** o expõe. Como o contraexemplo é relativamente raro, ampliei `max_examples` para a demonstração ser confiável.

---

## 4. Decisões de design

- **Lista vazia → retorna `[]`.** Um lote vazio é entrada legítima em pipeline; a função não deve quebrar.
- **Desvio-padrão zero (valores iguais) → retorna `[]`.** Sem variação não há outlier, e evita divisão por zero.
- **NaN ou infinito → `ValueError`.** Melhor falhar alto do que propagar resultado silenciosamente errado.
- **Limiar ≤ 0 → `ValueError`.** Um limiar não positivo não tem sentido estatístico.
- **Fronteira inclusiva (`>=`).** Um ponto exatamente no limiar conta como outlier (convenção documentada).
- **Desvio populacional (`pstdev`), não amostral.** Não afeta os invariantes de deslocamento e escala.
- **Retorno de índices, não de valores.** Permite rastrear a posição anômala e evita ambiguidade com valores repetidos.

**Limitação assumida:** o Z-score usa média e desvio globais, então é sensível a *mascaramento* em amostras pequenas — um valor extremo pode inflar o desvio a ponto de não ser detectado. É característica conhecida do método, não defeito da implementação, e foi o próprio TDD que me levou a percebê-la e documentá-la.

---

## Estrutura do repositório
