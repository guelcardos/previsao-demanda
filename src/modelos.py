"""
Modelos de previsão de séries temporais.

Todos seguem a mesma interface:
    prever(treino: pd.Series, horizonte: int) -> np.ndarray

A série de treino tem frequência mensal (MS) e o retorno é um vetor
com `horizonte` previsões.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

PERIODO_SAZONAL = 12  # sazonalidade anual em dados mensais


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def naive_sazonal(treino: pd.Series, horizonte: int) -> np.ndarray:
    """Repete o valor do mesmo mês do ano anterior.

    Baseline obrigatório: qualquer modelo que não superar o naive
    sazonal não justifica a própria complexidade.
    """
    ultimos = treino.iloc[-PERIODO_SAZONAL:].to_numpy()
    repeticoes = int(np.ceil(horizonte / PERIODO_SAZONAL))
    return np.tile(ultimos, repeticoes)[:horizonte]


def media_movel(treino: pd.Series, horizonte: int, janela: int = 12) -> np.ndarray:
    """Projeta a média dos últimos `janela` meses (previsão constante).

    Baseline ingênuo: ignora tendência e sazonalidade de propósito,
    para evidenciar o ganho dos modelos estruturados.
    """
    return np.full(horizonte, treino.iloc[-janela:].mean())


# ---------------------------------------------------------------------------
# Modelos estruturados
# ---------------------------------------------------------------------------

def holt_winters(treino: pd.Series, horizonte: int) -> np.ndarray:
    """Suavização exponencial tripla (tendência aditiva, sazonalidade
    multiplicativa)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = ExponentialSmoothing(
            treino,
            trend="add",
            seasonal="mul",
            seasonal_periods=PERIODO_SAZONAL,
        ).fit()
    return modelo.forecast(horizonte).to_numpy()


def sarima(treino: pd.Series, horizonte: int) -> np.ndarray:
    """SARIMA(1,1,1)(1,1,1)12 — ordem clássica para série mensal com
    tendência e sazonalidade anual."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = SARIMAX(
            treino,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, PERIODO_SAZONAL),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    return modelo.forecast(horizonte).to_numpy()


def sarima_com_intervalo(
    treino: pd.Series, horizonte: int, confianca: float = 0.95
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SARIMA com intervalo de confiança para a previsão final."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = SARIMAX(
            treino,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, PERIODO_SAZONAL),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        resultado = modelo.get_forecast(horizonte)
        intervalo = resultado.conf_int(alpha=1 - confianca)
    return (
        resultado.predicted_mean.to_numpy(),
        intervalo.iloc[:, 0].to_numpy(),
        intervalo.iloc[:, 1].to_numpy(),
    )


def holt_winters_com_intervalo(
    treino: pd.Series, horizonte: int, confianca: float = 0.95,
    simulacoes: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Holt-Winters com intervalo de confiança via simulação.

    A suavização exponencial não possui intervalo analítico no
    statsmodels; o intervalo é estimado simulando trajetórias futuras
    e tomando os percentis correspondentes.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = ExponentialSmoothing(
            treino,
            trend="add",
            seasonal="mul",
            seasonal_periods=PERIODO_SAZONAL,
        ).fit()
        trajetorias = modelo.simulate(
            horizonte, repetitions=simulacoes, error="mul",
            random_state=42,
        )
    alfa = (1 - confianca) / 2
    return (
        modelo.forecast(horizonte).to_numpy(),
        trajetorias.quantile(alfa, axis=1).to_numpy(),
        trajetorias.quantile(1 - alfa, axis=1).to_numpy(),
    )


MODELOS = {
    "Naive sazonal": naive_sazonal,
    "Media movel 12m": media_movel,
    "Holt-Winters": holt_winters,
    "SARIMA": sarima,
}

# modelos capazes de gerar a previsão final com intervalo de confiança
MODELOS_COM_INTERVALO = {
    "Holt-Winters": holt_winters_com_intervalo,
    "SARIMA": sarima_com_intervalo,
}
