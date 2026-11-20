"""
Geração dos gráficos do projeto (matplotlib).

Paleta categórica fixa (validada para daltonismo) e estilo sóbrio:
grades discretas, sem bordas superiores/direitas, linhas finas.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

PASTA_GRAFICOS = Path(__file__).resolve().parent.parent / "graficos"

# paleta categórica em ordem fixa (slots 1-4) + cor da série real
COR_SERIE = "#2a78d6"      # azul: série principal
CORES_MODELOS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
COR_REAL = "#333333"       # cinza escuro: valores reais
COR_INTERVALO = "#cde2fb"  # azul claro: banda de confiança
COR_TEXTO = "#333333"
COR_GRADE = "#e0e0e0"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": COR_GRADE,
        "axes.labelcolor": COR_TEXTO,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": COR_GRADE,
        "grid.linewidth": 0.6,
        "xtick.color": COR_TEXTO,
        "ytick.color": COR_TEXTO,
        "font.size": 10,
        "legend.frameon": False,
    }
)


def _preparar_eixo(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))


def _salvar(figura, nome: str) -> None:
    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
    caminho = PASTA_GRAFICOS / nome
    figura.tight_layout()
    figura.savefig(caminho, dpi=150)
    plt.close(figura)
    print(f"Grafico gravado: {caminho}")


def grafico_historico(serie: pd.Series) -> None:
    """Série mensal completa."""
    figura, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(serie.index, serie.values, color=COR_SERIE, linewidth=2)
    ax.set_title("Demanda mensal de cimento — histórico 2021-2025")
    ax.set_ylabel("Sacos vendidos / mês")
    _preparar_eixo(ax)
    _salvar(figura, "01_historico_mensal.png")


def grafico_decomposicao(serie: pd.Series) -> None:
    """Decomposição da série em tendência, sazonalidade e resíduo."""
    decomposicao = seasonal_decompose(serie, model="multiplicative", period=12)
    figura, eixos = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    partes = [
        (serie, "Série observada"),
        (decomposicao.trend, "Tendência"),
        (decomposicao.seasonal, "Sazonalidade"),
        (decomposicao.resid, "Resíduo"),
    ]
    for ax, (dados, titulo) in zip(eixos, partes):
        ax.plot(dados.index, dados.values, color=COR_SERIE, linewidth=1.6)
        ax.set_title(titulo, fontsize=10, loc="left")
        _preparar_eixo(ax)
    figura.suptitle("Decomposição da série de demanda", fontweight="bold")
    _salvar(figura, "02_decomposicao.png")


def grafico_comparacao(
    treino: pd.Series,
    teste: pd.Series,
    previsoes: dict[str, np.ndarray],
) -> None:
    """Real × previsto de cada modelo no período de holdout."""
    figura, ax = plt.subplots(figsize=(10, 4.5))

    contexto = pd.concat([treino.iloc[-12:], teste.iloc[:1]])
    ax.plot(contexto.index, contexto.values, color=COR_REAL,
            linewidth=2, alpha=0.45)
    ax.plot(teste.index, teste.values, color=COR_REAL, linewidth=2.4,
            label="Real", marker="o", markersize=4)

    for cor, (nome, previsto) in zip(CORES_MODELOS, previsoes.items()):
        ax.plot(teste.index, previsto, color=cor, linewidth=2,
                linestyle="--", label=nome)

    ax.axvline(teste.index[0], color=COR_GRADE, linewidth=1)
    ax.set_title("Backtesting — real × previsto no holdout (12 meses)")
    ax.set_ylabel("Sacos vendidos / mês")
    ax.legend(loc="upper left", ncols=2)
    _preparar_eixo(ax)
    _salvar(figura, "03_comparacao_modelos.png")


def grafico_previsao_final(
    serie: pd.Series,
    datas_futuras: pd.DatetimeIndex,
    previsao: np.ndarray,
    inferior: np.ndarray,
    superior: np.ndarray,
    nome_modelo: str,
) -> None:
    """Histórico + previsão de 12 meses com intervalo de confiança."""
    figura, ax = plt.subplots(figsize=(10, 4.5))

    historico = serie.iloc[-30:]
    ax.plot(historico.index, historico.values, color=COR_REAL,
            linewidth=2, label="Histórico")

    # conecta a previsão ao último ponto do histórico
    datas_plot = historico.index[-1:].append(datas_futuras)
    ultimo = historico.iloc[-1]
    previsao_plot = np.concatenate([[ultimo], previsao])
    inferior_plot = np.concatenate([[ultimo], inferior])
    superior_plot = np.concatenate([[ultimo], superior])

    ax.plot(datas_plot, previsao_plot, color=COR_SERIE, linewidth=2.2,
            label=f"Previsão ({nome_modelo})")
    ax.fill_between(datas_plot, inferior_plot, superior_plot,
                    color=COR_INTERVALO, label="Intervalo de 95%")

    ax.axvline(datas_futuras[0], color=COR_GRADE, linewidth=1)
    ax.set_title("Previsão de demanda — próximos 12 meses")
    ax.set_ylabel("Sacos vendidos / mês")
    ax.legend(loc="upper left")
    _preparar_eixo(ax)
    _salvar(figura, "04_previsao_final.png")
