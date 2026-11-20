"""
Pipeline de previsão de demanda.

Etapas:
1. Carrega as vendas diárias e agrega em série mensal.
2. Backtesting: treina 4 modelos nos primeiros 48 meses e avalia
   nos últimos 12 (holdout).
3. Compara as métricas (MAE, RMSE, MAPE) e identifica o melhor modelo.
4. Reajusta o melhor modelo na série completa e projeta 12 meses à
   frente com intervalo de confiança de 95%.
5. Gera gráficos e exporta os resultados em CSV/Excel.

Uso:
    python src/pipeline.py
"""

from pathlib import Path

import pandas as pd

import graficos
from avaliacao import backtest
from modelos import MODELOS, MODELOS_COM_INTERVALO

PASTA_DADOS = Path(__file__).resolve().parent.parent / "dados"
MESES_HOLDOUT = 12
MESES_PREVISAO = 12


def carregar_serie_mensal() -> pd.Series:
    """Lê as vendas diárias e agrega em total mensal."""
    df = pd.read_csv(PASTA_DADOS / "vendas_diarias.csv", parse_dates=["data"])
    serie = (
        df.set_index("data")["vendas_sacos"]
        .resample("MS")
        .sum()
        .astype(float)
    )
    serie.index.freq = "MS"
    return serie


def executar() -> None:
    # 1. Carga e agregação
    serie = carregar_serie_mensal()
    print(f"Serie mensal: {len(serie)} meses "
          f"({serie.index.min():%m/%Y} a {serie.index.max():%m/%Y})")

    # 2. Backtesting
    metricas, previsoes, treino, teste = backtest(serie, MODELOS, MESES_HOLDOUT)

    print("\nMETRICAS NO HOLDOUT (12 meses)")
    print("-" * 56)
    print(metricas.to_string(index=False,
                             formatters={"MAE": "{:.0f}".format,
                                         "RMSE": "{:.0f}".format,
                                         "MAPE (%)": "{:.2f}".format}))

    melhor = metricas.iloc[0]["modelo"]
    baseline = metricas.loc[
        metricas["modelo"] == "Naive sazonal", "MAPE (%)"
    ].iloc[0]
    ganho = baseline - metricas.iloc[0]["MAPE (%)"]
    print(f"\nMelhor modelo: {melhor} "
          f"(MAPE {metricas.iloc[0]['MAPE (%)']:.2f}% | "
          f"ganho de {ganho:.2f} p.p. sobre o naive sazonal)")

    # 3. Previsão final com o modelo vencedor do backtesting.
    # Se o vencedor for um baseline (sem intervalo de confiança),
    # usa o melhor modelo estruturado disponível.
    if melhor in MODELOS_COM_INTERVALO:
        modelo_final = melhor
    else:
        modelo_final = metricas.loc[
            metricas["modelo"].isin(MODELOS_COM_INTERVALO), "modelo"
        ].iloc[0]
        print(f"(baseline venceu; previsao final com {modelo_final})")

    previsao, inferior, superior = MODELOS_COM_INTERVALO[modelo_final](
        serie, MESES_PREVISAO
    )
    datas_futuras = pd.date_range(
        serie.index[-1] + pd.offsets.MonthBegin(), periods=MESES_PREVISAO,
        freq="MS",
    )
    df_previsao = pd.DataFrame(
        {
            "mes": datas_futuras,
            "previsao_sacos": previsao.round(0),
            "limite_inferior_95": inferior.round(0),
            "limite_superior_95": superior.round(0),
        }
    )

    # 4. Exportação
    df_previsao.to_csv(PASTA_DADOS / "previsao_12m.csv", index=False)
    with pd.ExcelWriter(PASTA_DADOS / "resultado_previsao.xlsx") as escritor:
        serie.rename("vendas_sacos").to_excel(escritor, sheet_name="Historico")
        metricas.to_excel(escritor, sheet_name="Metricas", index=False)
        df_previsao.to_excel(escritor, sheet_name="Previsao_12m", index=False)
    print(f"\nResultados exportados em {PASTA_DADOS}")

    # 5. Gráficos
    graficos.grafico_historico(serie)
    graficos.grafico_decomposicao(serie)
    graficos.grafico_comparacao(treino, teste, previsoes)
    graficos.grafico_previsao_final(
        serie, datas_futuras, previsao, inferior, superior, modelo_final
    )


if __name__ == "__main__":
    executar()
