"""Dataset sintetico de aprovacao de credito com vies de genero embutido no Modelo A."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1200

genero = rng.choice(["F", "M"], size=N)
idade = rng.integers(21, 70, size=N)
renda = np.round(rng.normal(4500, 1500, size=N).clip(1200, 20000), 2)
score = np.round(rng.normal(600, 90, size=N).clip(300, 850), 0)

# Rotulo real: depende de score e renda, NAO de genero
p_real = 1 / (1 + np.exp(-((score - 600) / 70 + (renda - 4500) / 3000)))
y_real = (rng.random(N) < p_real).astype(int)

# MODELO A (enviesado): penaliza o grupo F artificialmente
p_a = 1 / (1 + np.exp(-((score - 600) / 70 + (renda - 4500) / 3000 - 1.6 * (genero == "F"))))
pred_a = (rng.random(N) < p_a).astype(int)

# MODELO B (revisado): sem o termo de genero
p_b = 1 / (1 + np.exp(-((score - 600) / 70 + (renda - 4500) / 3000)))
pred_b = (rng.random(N) < p_b).astype(int)

df = pd.DataFrame({
    "id_cliente": np.arange(1, N + 1),
    "genero": genero,
    "idade": idade,
    "renda": renda,
    "score_credito": score,
    "aprovado_real": y_real,
    "pred_modelo_a": pred_a,
    "pred_modelo_b": pred_b,
})
df.to_csv("dados_credito.csv", index=False)
print(f"Dataset gerado: {df.shape[0]} linhas, {df.shape[1]} colunas")
print(df.head().to_string())
