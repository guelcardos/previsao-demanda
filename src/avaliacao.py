"""
Avaliação dos modelos por backtesting.

A série é dividida em treino e teste (holdout dos últimos meses).
Cada modelo é ajustado no treino e avaliado no teste com três métricas:

- MAE  (erro absoluto médio, em sacos)
- RMSE (raiz do erro quadrático médio, penaliza erros grandes)
- MAPE (erro percentual absoluto médio, comparável entre séries)
"""

import numpy as np
import pandas as pd


def mae(real: np.ndarray, previsto: np.ndarray) -> float:
    return float(np.mean(np.abs(real - previsto)))


def rmse(real: np.ndarray, previsto: np.ndarray) -> float:
    return float(np.sqrt(np.mean((real - previsto) ** 2)))


def mape(real: np.ndarray, previsto: np.ndarray) -> float:
    return float(np.mean(np.abs((real - previsto) / real)) * 100)


def backtest(
    serie: pd.Series, modelos: dict, meses_teste: int
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.Series, pd.Series]:
    """Treina cada modelo no período de treino e avalia no holdout.

    Devolve (tabela de métricas, previsões por modelo, treino, teste).
    """
    treino = serie.iloc[:-meses_teste]
    teste = serie.iloc[-meses_teste:]
    real = teste.to_numpy()

    linhas, previsoes = [], {}
    for nome, funcao in modelos.items():
        previsto = funcao(treino, meses_teste)
        previsoes[nome] = previsto
        linhas.append(
            {
                "modelo": nome,
                "MAE": mae(real, previsto),
                "RMSE": rmse(real, previsto),
                "MAPE (%)": mape(real, previsto),
            }
        )

    metricas = (
        pd.DataFrame(linhas).sort_values("MAPE (%)").reset_index(drop=True)
    )
    return metricas, previsoes, treino, teste
