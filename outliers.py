import statistics


def detectar_outliers(dados, limiar=3.0):
    if len(dados) == 0:
        return []

    media = statistics.mean(dados)
    desvio = statistics.pstdev(dados)

    if desvio == 0:
        return []

    outliers = []
    for i, valor in enumerate(dados):
        z = abs(valor - media) / desvio
        if z >= limiar:
            outliers.append(i)
    return outliers
