# Detecção de Outliers por Z-score — TDD e Testes de Propriedade

Implementação de uma função de detecção de outliers por **Z-score** para pipelines de Machine Learning, desenvolvida com **TDD** (Test-Driven Development) e validada com **testes de propriedade** usando a biblioteca `hypothesis`.

**Autor:** Gabriel Moraes Ramos Studart  
**Disciplina:** Orquestração de Workflows

---

## Função escolhida

A função `detectar_outliers` recebe uma sequência de números e um limiar, e retorna os **índices** dos pontos cujo Z-score (distância da média em desvios-padrão) atinge ou ultrapassa o limiar.

Optei por retornar índices, e não valores, para que o pipeline consiga rastrear exatamente qual posição do lote é anômala, mesmo quando há valores repetidos.

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

Crie e ative um ambiente virtual e instale as dependências:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install pytest hypothesis
```

Para executar a suíte principal, que valida a implementação correta:

```bash
pytest test_outliers.py -v
```

Para executar a demonstração do bug proposital:

```bash
pytest test_bug.py -v
```

> **Nota:** `test_bug.py` foi feito para **falhar de propósito**. Ele executa um teste de propriedade contra uma versão propositalmente incorreta da função, demonstrando que o teste seria capaz de detectar o bug. A suíte que valida a implementação correta é `test_outliers.py`.

---

## 1. Ciclo TDD — Red → Green → Refactor

O desenvolvimento seguiu a abordagem TDD, com os testes sendo escritos antes da implementação e com o histórico do Git servindo como evidência da evolução do código.

### Histórico de commits

```text
$ git log --oneline

56d2308 test: torna a demonstracao do bug confiavel (max_examples)
b71a795 test: demonstra property test pegando bug real
d30d847 test: adiciona property tests (indices, deslocamento, escala)
6e1bccd refactor: docstring, type hints e tratamento de NaN/inf/limiar
fe77655 feat(green): implementa deteccao de outliers por z-score
a868b69 test(red): testes de outliers falhando (funcao nao implementada)
fe2597a chore: setup inicial do projeto
```

### Red — o teste falha primeiro

Inicialmente, escrevi os testes antes da implementação da função.

Nesse momento, `detectar_outliers` ainda não possuía a lógica necessária e retornava `None`. Dessa forma, os testes falharam, comprovando que realmente estavam verificando o comportamento esperado.

```text
test_detecta_outlier_obvio                  FAILED
test_sem_outliers_retorna_lista_vazia       FAILED
test_lista_vazia                            FAILED

E   assert None == [6]

========================= 3 failed in 0.71s =========================
```

Essa etapa representa o **Red** do ciclo TDD.

---

### Green — código mínimo que faz passar

Depois dos testes falhando, implementei a lógica necessária para calcular o Z-score e identificar os índices considerados outliers.

Durante essa fase surgiu uma descoberta importante: meu primeiro exemplo de teste estava estatisticamente incorreto.

Em uma amostra pequena, um valor extremo pode aumentar a própria média e o desvio-padrão a ponto de seu Z-score não ultrapassar o limiar definido. Nesse caso, o outlier acaba se **mascarando**.

A função estava correta; o teste é que fazia uma afirmação estatisticamente falsa. O cenário de teste foi corrigido e a suíte passou.

```text
test_detecta_outlier_obvio                  PASSED
test_sem_outliers_retorna_lista_vazia       PASSED
test_lista_vazia                            PASSED
test_todos_valores_iguais_sem_outlier       PASSED

========================= 4 passed in 0.26s =========================
```

Essa etapa representa o **Green** do ciclo TDD.

---

### Refactor — melhorar sem mudar o comportamento

Com a implementação funcionando, realizei uma etapa de refatoração.

Foram adicionados:

- Docstring.
- Type hints.
- Tratamento para valores `NaN`.
- Tratamento para valores infinitos.
- Validação de limiar inválido.
- Tratamento explícito para desvio-padrão igual a zero.

Cada novo comportamento recebeu seu respectivo teste.

Depois da refatoração, a suíte continuou verde:

```text
========================= 7 passed in 0.21s =========================
```

Dessa forma, o comportamento externo da função foi preservado enquanto sua robustez e legibilidade foram melhoradas.

---

## 2. Testes de propriedade com Hypothesis

Além dos testes tradicionais baseados em exemplos específicos, foram implementados **testes de propriedade** usando a biblioteca `hypothesis`.

Em vez de verificar somente algumas entradas previamente escolhidas, o Hypothesis gera diversas combinações de dados automaticamente e verifica se determinadas propriedades permanecem verdadeiras.

Foram utilizadas três propriedades principais:

| Invariante | O que garante | Bug que pode detectar |
|---|---|---|
| Índices válidos e únicos | Todo índice retornado existe em `dados` e não aparece duplicado | Erros de indexação |
| Invariância a deslocamento | Somar uma constante a todos os valores não altera os outliers | Esquecer de centralizar os dados pela média |
| Invariância a escala positiva | Multiplicar todos os valores pelo mesmo fator positivo não altera os outliers | Erros relacionados ao tratamento da escala |

Exemplo dos testes:

```python
numeros = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False
)

listas = st.lists(
    numeros,
    min_size=1,
    max_size=50
)


@given(listas)
def test_indices_sempre_validos_e_unicos(dados):
    resultado = detectar_outliers(dados)

    assert all(0 <= i < len(dados) for i in resultado)
    assert len(resultado) == len(set(resultado))


@given(
    listas,
    st.floats(
        min_value=-1000,
        max_value=1000,
        allow_nan=False,
        allow_infinity=False
    )
)
def test_invariante_a_deslocamento(dados, deslocamento):
    base = detectar_outliers(dados)
    deslocado = detectar_outliers(
        [x + deslocamento for x in dados]
    )

    assert base == deslocado
```

Após a inclusão dos testes tradicionais e dos testes de propriedade, a suíte principal permaneceu totalmente verde:

```text
$ pytest test_outliers.py -v

========================= 10 passed in 0.70s =========================
```

---

## 3. Prova de que o teste de propriedade detecta um bug real

Para demonstrar que os testes de propriedade realmente são capazes de encontrar problemas que poderiam passar despercebidos em testes tradicionais, criei uma versão propositalmente **bugada** da função.

Na implementação correta, o Z-score é calculado utilizando:

```python
abs(valor - media) / desvio
```

Na versão incorreta, a centralização pela média foi removida:

```python
# BUG: deveria ser abs(valor - media) / desvio

return [
    i for i, valor in enumerate(dados)
    if abs(valor) / desvio >= limiar
]
```

Nesse caso, o resultado passa a depender da posição absoluta dos dados na reta numérica, quando deveria depender apenas da distância de cada ponto em relação à média.

O teste de **invariância a deslocamento** foi executado contra essa implementação propositalmente incorreta.

O Hypothesis encontrou automaticamente um contraexemplo:

```text
test_bug.py::test_bug_quebra_invariante_de_deslocamento FAILED

Falsifying example:
dados=[0.0, 1.0],
deslocamento=1.0

E   assert [] == [1]

========================= 1 failed in 0.70s =========================
```

### Interpretação do contraexemplo

Para os dados:

```text
[0.0, 1.0]
```

a versão bugada não identifica nenhum outlier.

Ao somarmos `1.0` a todos os elementos, temos:

```text
[1.0, 2.0]
```

Como todos os valores foram deslocados pela mesma constante, a distribuição relativa dos dados permanece exatamente a mesma.

Portanto, uma implementação correta deveria retornar os mesmos índices antes e depois do deslocamento.

Entretanto, a implementação bugada passa a marcar o índice do valor `2.0` como outlier.

Isso acontece porque ela utiliza:

```python
abs(valor) / desvio
```

em vez de:

```python
abs(valor - media) / desvio
```

Assim, o teste de propriedade detecta que o comportamento da função mudou mesmo sem nenhuma alteração na distribuição relativa dos dados.

**Esse teste teria impedido que esse bug chegasse à produção.**

### Por que o teste de escala não detecta esse bug?

Esse bug específico continua sendo invariante à escala.

Ao multiplicar todos os valores por um fator positivo `f`, temos:

```text
abs(valor × f) / (desvio × f)
```

O fator aparece tanto no numerador quanto no denominador e acaba sendo cancelado.

Por isso, o teste que realmente expõe esse erro é o de **invariância a deslocamento**.

Como determinados contraexemplos podem ser relativamente raros entre os valores gerados aleatoriamente, aumentei o parâmetro `max_examples` na demonstração para tornar a detecção confiável.

---

## 4. Decisões de design

Durante o desenvolvimento, algumas decisões foram tomadas explicitamente.

### Lista vazia

Uma lista vazia retorna:

```python
[]
```

Um lote sem dados pode ser uma entrada legítima em um pipeline e não precisa provocar uma exceção.

### Desvio-padrão igual a zero

Quando todos os valores são iguais, o desvio-padrão é zero.

Nesse cenário, nenhum elemento pode ser considerado anômalo em relação aos demais, então a função retorna:

```python
[]
```

Isso também evita uma divisão por zero.

### NaN ou infinito

Caso os dados contenham `NaN` ou infinito, a função lança:

```python
ValueError
```

A decisão foi falhar explicitamente em vez de permitir que valores inválidos produzam resultados silenciosamente incorretos.

### Limiar menor ou igual a zero

Um limiar não positivo não possui significado adequado para a regra utilizada.

Por isso:

```python
limiar <= 0
```

resulta em:

```python
ValueError
```

### Fronteira inclusiva

A comparação utilizada é:

```python
>= limiar
```

Portanto, um ponto cujo Z-score esteja exatamente no limiar também é considerado outlier.

### Desvio-padrão populacional

Foi utilizado:

```python
statistics.pstdev
```

em vez do desvio-padrão amostral.

Essa escolha também preserva as propriedades de invariância a deslocamento e escala verificadas pelos testes.

### Retorno de índices

A função retorna os **índices** dos outliers, e não seus valores.

Isso facilita o uso em pipelines de Machine Learning, pois permite rastrear exatamente qual posição do lote contém o dado anômalo.

Também evita ambiguidades quando existem valores repetidos.

---

## Limitação conhecida do método

O Z-score utiliza a média e o desvio-padrão globais da distribuição.

Por esse motivo, ele pode sofrer com o fenômeno de **mascaramento**, principalmente em amostras pequenas.

Um valor extremo pode influenciar a própria média e aumentar o desvio-padrão de tal forma que seu Z-score não ultrapasse o limiar necessário para ser classificado como outlier.

Essa é uma característica conhecida do método estatístico utilizado, e não um defeito específico desta implementação.

O próprio processo de TDD ajudou a identificar essa característica quando um dos primeiros casos de teste fazia uma suposição estatística incorreta.

---

## Estrutura do repositório

```text
outliers.py           # implementação correta utilizando Z-score

outliers_bugado.py    # versão propositalmente bugada para demonstração

test_outliers.py      # testes tradicionais + testes de propriedade

test_bug.py           # teste contra a versão bugada (falha propositalmente)
```

A suíte principal está concentrada em:

```text
test_outliers.py
```

e deve passar completamente.

Já:

```text
test_bug.py
```

é utilizado exclusivamente como demonstração e **deve falhar**, pois testa uma implementação propositalmente incorreta.

---

## Conclusão

A atividade demonstra o desenvolvimento de uma função de detecção de outliers utilizando **Test-Driven Development**, começando pelos testes, seguindo para uma implementação mínima funcional e posteriormente realizando a refatoração.

Além dos testes tradicionais, os testes de propriedade com Hypothesis verificam invariantes importantes da função utilizando diferentes entradas geradas automaticamente.

Por fim, a criação de uma implementação propositalmente bugada demonstra, de forma prática, que o teste de invariância a deslocamento consegue identificar um erro real na fórmula do Z-score.

O histórico de commits mantém registradas as etapas **Red → Green → Refactor**, servindo como evidência do processo de desenvolvimento adotado.
