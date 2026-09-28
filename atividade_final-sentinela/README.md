# Suíte de testes do Sentinela (Trilha 1, ML clássico)

**Disciplina:** Testes Automatizados
**Sistema testado:** [sentinela-nortemec](https://github.com/felipehp/sentinela-nortemec), previsão de falha de motor elétrico em 72 horas

Esta suíte testa o pacote `sentinela` de fora, sem editar uma linha dele. Todo defeito encontrado está registrado como um teste que falha, com o número medido na mensagem.

## Como rodar

```bash
git clone https://github.com/felipehp/sentinela-nortemec.git sentinela
cd sentinela
python3 -m venv .venv && source .venv/bin/activate
pip install -e . && pip install pytest scipy

# copie a pasta tests/ desta entrega para dentro de sentinela/ e rode:
pytest tests/ -v -s --tb=line | tee log_execucao.txt
```

O `-s` mostra os números que cada teste mede (IC, p-valor, taxas). O log completo da minha execução está em [`log_execucao.txt`](log_execucao.txt).

## Resultado da execução

```
======================== 18 failed, 33 passed in 6.35s =========================
```

Os 33 testes verdes mostram o que o sistema faz certo: `limpar()` não altera a entrada, converte tipos, normaliza rótulos e não deixa faltantes. `construir()` devolve as 13 features na ordem canônica, é determinístico e independe da ordem das linhas. O modelo devolve probabilidades válidas e rejeita NaN. Os dados crus respeitam a ficha em temperatura, corrente e idade. O pipeline aguenta lote de uma linha e de um motor só.

Os 18 vermelhos correspondem a 12 defeitos distintos.

## Defeitos encontrados

| # | Defeito | Evidência medida | Teste |
|---|---|---|---|
| 1 | Vibração faltante vira 0,0 mm/s | 408 leituras fora de 1,2 a 8,0 | `test_DEFEITO_vibracao_imputada...` |
| 2 | Pressão em psi nunca convertida | 1.344 leituras (32%) entre 48 e 65 "bar" | `test_DEFEITO_pressao_em_psi...` |
| 3 | Leitura fora da ficha aceita | 150 °C gera decisão sem alerta | `test_DEFEITO_leitura_fora_da_ficha...` |
| 4 | Feature usa leitura do futuro | `temp_media_6h` muda com dado 2 h à frente | `test_DEFEITO_feature_usa_leitura_do_futuro` |
| 5 | Feature depende do rótulo | `maquina_risco` recalculado de `falha_72h` | `test_DEFEITO_features_dependem_do_rotulo` |
| 6 | `prever_registro` ignora as chaves | 29% das decisões mudam com o dicionário em outra ordem | `test_DEFEITO_prever_registro...` |
| 7 | v2 perde recall para v1 | IC95% da diferença [-0,272; -0,207], FN 41 → 198 | `test_DEFEITO_v2_nao_pode_perder_recall...` |
| 8 | v1 descalibrada | ECE 0,191 (v2: 0,058) | `test_calibracao...[v1]` |
| 9 | Drift teste → produção | KS p = 3,5e-33 na temperatura com média quase igual | `test_DEFEITO_drift...` |
| 10 | v2 instável abaixo do ruído | ±0,5 °C muda 1,45% das decisões (v1: 0,93%) | `test_perturbacao...[v2]` |
| 11 | Decisão depende do operador | trocar OP-07 muda 8,4% (v1) e 11,0% (v2) | `test_DEFEITO_contrafactual_operador...` |
| 12 | Integridade não portável | manifest gerado em CRLF, repositório em LF | `test_DEFEITO_verificar_manifest...` |

## Os defeitos mais graves, em detalhe

### A promoção da v2 colocaria motores em risco (defeitos 5 e 7)

**Teste.** `test_DEFEITO_v2_nao_pode_perder_recall_para_v1` faz 2.000 reamostragens bootstrap da diferença de recall entre as versões e exige que o limite inferior do IC não seja negativo.

**Evidência.** Avaliando o lote sem rótulo, que é como ele chega em produção, a v2 tem recall de 0,698 contra 0,937 da v1. O IC95% da diferença fica inteiro abaixo de zero, em [-0,272; -0,207]. Os falsos negativos sobem de 41 para 198. O McNemar confirma que as versões diferem de fato (p = 8,7e-45), então não é ruído de amostra.

**Causa raiz.** São dois mecanismos somados. A v2 foi treinada sem reponderação de classe, então com limiar 0,5 ela quase só acerta a classe majoritária. E os +3,5 pp de acurácia que motivaram a promoção foram medidos com o lote rotulado, onde `_risco_por_maquina` recalcula o risco de cada motor a partir do próprio `falha_72h`. O modelo recebe a resposta como feature. Na mesma base, sem o rótulo, o recall da v2 cai de 0,814 para 0,698.

**Impacto.** Na planta, falso negativo é motor queimado sem ordem de manutenção. Promover a v2 com base na acurácia significaria cerca de 157 falhas a mais não sinalizadas por semana de operação desta planta. A varredura de limiar mostra que a v2 só volta a recall acima de 0,90 com limiar 0,2, o que o sistema não usa.

### A feature olha para o futuro (defeito 4)

**Teste.** Altero em +20 °C uma leitura duas horas à frente de um instante `t` do motor M01 e verifico se alguma feature de `t` mudou.

**Evidência.** `temp_media_6h` de `t` muda. As outras 12 features não.

**Causa raiz.** A janela usa `rolling(6, center=True)`. Com a janela centrada, a média no instante `t` inclui leituras de `t+1`, `t+2` e `t+3`, que não existem no momento da decisão.

**Impacto.** O modelo foi treinado e avaliado vendo parte do futuro, então o desempenho reportado é otimista. Em tempo real a última leitura de cada motor sempre tem a janela truncada, e a feature tem distribuição diferente da que o modelo aprendeu.

### A decisão depende de quem está no turno (defeito 11)

**Teste.** Contrafactual que inverte só `operador_senior` e mantém todas as leituras físicas.

**Evidência.** 8,4% das decisões da v1 e 11,0% da v2 mudam.

**Causa raiz.** O código marca o OP-07 porque ele "é escalado para os motores mais críticos". Isso faz o modelo aprender que a presença do técnico sênior indica falha, uma relação causalmente ilegítima: o técnico está lá porque alguém já suspeitava do motor.

**Impacto.** Trocar a escala de turno altera ordens de manutenção sem nenhuma mudança no equipamento.

### O pré-processamento viola a ficha técnica (defeitos 1 e 2)

A ficha declara vibração de operação entre 1,2 e 8,0 mm/s, mas `limpar()` preenche o dropout com 0,0, um valor que só existiria com o motor parado. E 32% das leituras de pressão chegam em psi. O código normaliza o texto da unidade para minúsculas, mas nunca converte o número, então o modelo recebe pressões entre 48 e 65 misturadas com valores entre 3 e 4,5 na mesma coluna.

## Decisões da suíte

Avaliei sempre com o lote **sem rótulo**, porque é assim que o dado chega em produção e porque a presença do rótulo contamina `maquina_risco`. Escolhi recall e PR-AUC como métricas principais por causa da classe rara: um modelo que nunca prevê falha já acerta 84,4% (teste `test_acuracia_nao_serve_para_classe_rara`). Para drift usei Kolmogorov-Smirnov em vez de comparar médias. O teste `test_medias_teste_producao_parecem_iguais` passa com diferença de 1,2 °C, enquanto o KS detecta que o desvio da temperatura subiu de 3,98 para 5,25. Todos os testes com aleatoriedade têm semente fixa, então a execução é reproduzível.

## Estrutura

```
tests/
  conftest.py                 fixtures e ficha técnica dos sensores
  test_preprocessamento.py    Bloco A: limpar() e contrato de dados
  test_features.py            Bloco A: construir(), causalidade temporal, contrato do modelo
  test_estatistico.py         Bloco B: bootstrap, McNemar, PR-AUC, limiar, calibração, drift
  test_adversarial.py         Bloco C: ruído, contrafactuais, casos-limite, integridade
log_execucao.txt              saída completa do pytest
```
