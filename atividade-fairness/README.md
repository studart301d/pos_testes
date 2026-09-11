# Validação de Dados, Testes Estatísticos e Fairness

Suíte automatizada sobre um dataset com atributo sensível (gênero) em cenário de aprovação de crédito.

**Disciplina:** Testes Automatizados para Modelos de IA
**Atividade 2**

## Contexto e dataset

Escolhi um cenário de aprovação de crédito, caso clássico onde decisão automatizada pode discriminar grupos. O dataset é sintético (1200 registros), opção permitida pelo enunciado. Gerei os dados eu mesmo de propósito: como conheço a verdade do processo gerador, consigo verificar se a suíte realmente detecta o viés que plantei.

O rótulo real depende apenas de score e renda, nunca de gênero. No Modelo A introduzi deliberadamente um termo que penaliza o grupo F. O Modelo B é idêntico, sem esse termo.

## Como rodar

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install pandas numpy pandera scipy pytest
python gerar_dataset.py
pytest test_suite.py -v -s
```

## 1. Validação de dados

Usei pandera para declarar o schema, cobrindo schema (colunas, tipos e domínio), completude (nenhum valor faltante) e unicidade/faixa (id_cliente como chave, faixas plausíveis de idade, renda e score).

Acrescentei dois testes além do mínimo: um verifica se os dois grupos do atributo sensível têm amostra mínima, porque fairness sobre grupo pequeno não tem valor estatístico; outro injeta um score de 9999 e confirma que o schema levanta erro, provando que a validação está ativa e não passando por sorte.

```
test_schema_valido                  PASSED
test_completude                     PASSED
test_unicidade_id                   PASSED
test_grupos_representados           PASSED
test_schema_rejeita_dado_corrompido PASSED
```

## 2. Testes estatísticos

Cobri os dois pontos, não apenas um.

Intervalo de confiança por bootstrap (2000 reamostragens):

```
[IC bootstrap] Acuracia modelo A: 0.627 | IC95% = [0.600, 0.653]
```

O intervalo não cruza 0.5, então o modelo é melhor que o acaso de forma confiável. Também assertei largura do IC abaixo de 0.15, pois um intervalo largo indicaria amostra insuficiente.

Comparação entre versões com teste t pareado (mesmos clientes nas duas):

```
[Comparacao A vs B] acc_A=0.627 acc_B=0.651 | p-valor=0.1655
```

Com p = 0.166 não rejeito a hipótese de igualdade: a diferença é compatível com ruído de amostra.

## 3. Teste de fairness

Escolhi paridade demográfica pela regra dos 80% (disparate impact ratio). Justifico: é o critério de referência em crédito e contratação, com respaldo regulatório, e é diretamente interpretável para quem não é técnico.

```
[Fairness A] taxa F=0.290 M=0.488 | dif=0.198 | ratio=0.594   <- VIOLA
[Fairness B] taxa F=0.526 M=0.471 | dif=0.055 | ratio=0.896   <- OK
```

Marquei o teste do modelo A como xfail estrito: a falha é esperada e documentada, e o modo estrito acusa se alguém alterar o modelo sem atualizar a suíte.

Qui-quadrado sobre gênero x predição, para descartar acaso:

```
[Qui-quadrado modelo A] chi2=48.63 p-valor=0.000000
========================= 9 passed, 1 xfailed in 1.24s =========================
```

## 4. Interpretação

**Dataset.** Passou em schema, completude, unicidade e faixas, e os grupos têm amostra suficiente. O teste com dado corrompido confirma que a validação funciona. Logo, os problemas adiante não vêm de dado sujo: são do modelo.

**Performance.** Acurácia de 0.627 com IC95% [0.600, 0.653]. O modelo funciona e o intervalo estreito mostra que a amostra sustenta a estimativa.

**Comparação.** A e B são estatisticamente equivalentes em acurácia (p = 0.166). Decidindo só por performance, não haveria argumento para trocar um pelo outro.

**Equidade.** Aqui os modelos se separam. O A aprova 48.8% dos homens contra 29.0% das mulheres, razão de 0.594, bem abaixo de 0.80. O qui-quadrado (p < 0.000001) descarta acaso: a disparidade é sistemática. O B, com 0.896, fica aceitável.

**Conclusão.** Os dois modelos são indistinguíveis em acurácia mas radicalmente diferentes em equidade. Isso demonstra o ponto central: métrica de performance não detecta discriminação. Um pipeline que validasse só acurácia aprovaria o modelo enviesado sem alerta nenhum. Só o teste de fairness expôs o problema. Na prática, paridade demográfica precisa ser portão automatizado a cada alteração de modelo, não auditoria manual esporádica.

Limitação do que fiz: paridade demográfica olha só a taxa de aprovação e ignora se os grupos têm perfis de risco diferentes. Em cenário real eu complementaria com métricas condicionadas ao rótulo verdadeiro, como igualdade de oportunidade, já que as duas podem apontar direções distintas.

## Estrutura

```
gerar_dataset.py    # gera o dataset sintetico com vies controlado
test_suite.py       # validacao + testes estatisticos + fairness (10 testes)
dados_credito.csv   # dataset gerado
```
