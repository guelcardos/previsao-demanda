"""
Gerador de dados sintéticos de demanda.

Simula 5 anos (2021-2025) de vendas diárias de cimento (sacos de 50kg)
de uma rede fictícia de lojas de materiais de construção. A série é
construída com componentes realistas:

- tendência de crescimento gradual;
- sazonalidade anual (demanda maior na estação seca, menor no período
  de chuvas dez-fev);
- efeito de dia da semana (domingo fraco, sábado forte);
- promoções pontuais (picos localizados);
- ruído aleatório.

Uso:
    python src/gerar_dados.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

PASTA_DADOS = Path(__file__).resolve().parent.parent / "dados"
SEMENTE = 42

DATA_INICIO = "2021-01-01"
DATA_FIM = "2025-12-31"

BASE_DIARIA = 820          # nível médio inicial de vendas (sacos/dia)
CRESCIMENTO_ANUAL = 0.06   # tendência: +6% ao ano
AMPLITUDE_SAZONAL = 0.22   # +/-22% ao longo do ano
RUIDO = 0.07               # desvio do ruído diário (7%)

# fator multiplicativo por dia da semana (seg=0 ... dom=6)
FATOR_DIA_SEMANA = [1.00, 1.02, 1.03, 1.05, 1.10, 1.18, 0.45]

# meses com promoção: (ano, mes) -> acréscimo percentual
PROMOCOES = {
    (2022, 5): 0.15,
    (2023, 9): 0.18,
    (2024, 5): 0.12,
    (2025, 8): 0.16,
}


def gerar_serie() -> pd.DataFrame:
    rng = np.random.default_rng(SEMENTE)
    datas = pd.date_range(DATA_INICIO, DATA_FIM, freq="D")
    dias = np.arange(len(datas))

    # tendência de crescimento composto
    tendencia = BASE_DIARIA * (1 + CRESCIMENTO_ANUAL) ** (dias / 365.25)

    # sazonalidade anual: pico por volta de agosto (estação seca),
    # vale no período de chuvas (dez-fev)
    dia_do_ano = datas.dayofyear.to_numpy()
    sazonalidade = 1 + AMPLITUDE_SAZONAL * np.sin(
        2 * np.pi * (dia_do_ano - 110) / 365.25
    )

    # efeito de dia da semana
    fator_semana = np.array([FATOR_DIA_SEMANA[d] for d in datas.dayofweek])

    # promoções pontuais
    fator_promo = np.ones(len(datas))
    for (ano, mes), acrescimo in PROMOCOES.items():
        mascara = (datas.year == ano) & (datas.month == mes)
        fator_promo[mascara] += acrescimo

    # ruído aleatório
    ruido = rng.normal(1.0, RUIDO, len(datas))

    vendas = tendencia * sazonalidade * fator_semana * fator_promo * ruido
    vendas = np.maximum(vendas, 0).round().astype(int)

    return pd.DataFrame({"data": datas, "vendas_sacos": vendas})


def main() -> None:
    PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    df = gerar_serie()
    caminho = PASTA_DADOS / "vendas_diarias.csv"
    df.to_csv(caminho, index=False)
    print(f"{len(df)} registros diarios gravados em {caminho}")
    print(f"Periodo: {df['data'].min().date()} a {df['data'].max().date()}")
    print(f"Media diaria: {df['vendas_sacos'].mean():.0f} sacos")


if __name__ == "__main__":
    main()
