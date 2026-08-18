"""Genera las figuras del documento a partir de los datos curados y de las
predicciones de cada modelo.

Produce dos familias de figuras:

* ``iniciales/<activo>.png`` --- gráfico OHLC del precio, sobre los primeros
  5 000 registros de la serie curada (o la serie completa si tiene menos),
  que es exactamente el recorte que mostraban las versiones originales.
* ``<activo>_<modelo>.png`` --- volatilidad predicha vs. realizada sobre el
  conjunto de prueba, a partir de los CSV de predicciones de cada modelo.

Uso::

    python proyecto/generar_figuras.py                 # sobrescribe las_figuras/
    python proyecto/generar_figuras.py --outdir /tmp/x # genera en otra carpeta
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "src" / "finance_modeling" / "data" / "files"
PREDICCIONES = RAIZ / "src" / "finance_modeling" / "results" / "models" / "20260629"
FIGURAS = Path(__file__).resolve().parent / "las_figuras"

# Recorte usado por las figuras originales de precios.
MAX_REGISTROS = 5000

# Lienzo lógico y factor de escala del render. El PNG resultante mide
# ANCHO*ESCALA x ALTO*ESCALA px, lo que a \textwidth (~15 cm) deja unos 400 ppp.
ANCHO, ALTO, ESCALA = 1200, 620, 2

# Tipografía. Las figuras se insertan a \textwidth (~15 cm), así que el texto
# se reduce bastante al imprimir: con ~2,2 % del ancho del lienzo las marcas
# quedan en torno a 7 pt en el PDF, legibles junto a un cuerpo de 11 pt.
FUENTE_MARCAS = 26
FUENTE_TITULO_EJE = 28
FUENTE_LEYENDA = 28

# Paleta tomada de las figuras originales.
VERDE_ALZA = "#1aab40"
ROJO_BAJA = "#d64554"
ROJO_PREDICCION = "#c65054"
NEGRO_REAL = "#000000"
GRIS_REJILLA = "#e6e6e6"

# Marcas, títulos de eje y leyenda en negro: los grises de Power BI se
# apagaban demasiado al reducir la figura al ancho del texto.
NEGRO_TEXTO = "#000000"

# El documento no carga ningún paquete de fuentes, así que compone con
# Computer Modern; Latin Modern Roman es su versión OpenType y es la que
# resuelve el navegador con el que kaleido exporta los PNG. Debe estar
# instalada para el usuario (las OTF vienen con MiKTeX, en
# fonts/opentype/public/lm); si falta, se cae a los alternativos con serifa.
FUENTE = "Latin Modern Roman, CMU Serif, Times New Roman, serif"

# activo -> (carpeta de datos, símbolo en los CSV de predicciones, prefijo de figura)
ACTIVOS = {
    "bitcoin": ("bitcoin", "BTC-USD", "bitcoin"),
    "ethereum": ("ethereum", "ETH-USD", "ethereum"),
    "eur_usd": ("eur_usd", "EURUSD=X", "eurusd"),
    "gold": ("gold", "GC=F", "gold"),
    "sp500": ("sp500", "^GSPC", "sp500"),
    "nasdaq": ("nasdaq", "^IXIC", "nasdaq"),
}

# sufijo de figura -> prefijo del CSV de predicciones
MODELOS = {
    "garch": "GARCH",
    "ceemdan": "CEEMDAN_LSTM",
    "pso": "PSOQRNN",
}


def _disenio_base() -> dict:
    """Layout común a todas las figuras."""
    eje = dict(
        showgrid=True,
        gridcolor=GRIS_REJILLA,
        griddash="dot",
        gridwidth=1,
        showline=False,
        zeroline=False,
        ticks="",
        tickfont=dict(size=FUENTE_MARCAS, color=NEGRO_TEXTO, family=FUENTE),
        title_font=dict(size=FUENTE_TITULO_EJE, color=NEGRO_TEXTO, family=FUENTE),
    )
    return dict(
        template="none",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=FUENTE, color=NEGRO_TEXTO),
        xaxis=dict(eje),
        yaxis=dict(eje),
        margin=dict(l=135, r=30, t=30, b=95),
        showlegend=False,
    )


# Marcas objetivo del eje de volatilidad y holgura que plotly añade al rango
# de los datos al autoescalar (medida sobre las figuras originales).
MARCAS_OBJETIVO = 6
HOLGURA_AUTORANGO = 1.18


def _escala_ticks(rango: float) -> tuple[float, str]:
    """Paso y decimales del eje de volatilidad según el rango de la serie.

    La escala varía dos órdenes de magnitud entre activos (el GARCH del
    Bitcoin llega a 0,24 mientras que el CEEMDAN-LSTM del EUR/USD se mueve
    entre 0,004 y 0,008). Se fija el paso de forma explícita en lugar de
    dejarlo al autoescalado para que todas las marcas del eje lleven el mismo
    número de decimales, como en las figuras originales; sin esto plotly
    recorta los ceros finales y una misma figura mezcla "0,015" con "0,02".
    """
    if rango <= 0:
        return 0.01, ".2f"
    crudo = rango * HOLGURA_AUTORANGO / MARCAS_OBJETIVO
    exponente = math.floor(math.log10(crudo))
    mantisa = crudo / 10**exponente
    # Se elige la mantisa "bonita" más cercana en escala logarítmica.
    bonita = min((1, 2, 5, 10), key=lambda c: abs(math.log10(c) - math.log10(mantisa)))
    paso = bonita * 10**exponente
    return paso, f".{max(0, -math.floor(math.log10(paso)))}f"


def _leer_curado(carpeta: str) -> pd.DataFrame:
    (ruta,) = (DATOS / carpeta).glob("*_curated.csv")
    df = pd.read_csv(ruta, sep=";", parse_dates=["timestamp"])
    return df.sort_values("timestamp").head(MAX_REGISTROS)


def figura_precio(carpeta: str) -> go.Figure:
    """Gráfico OHLC del precio de un activo."""
    df = _leer_curado(carpeta)
    fig = go.Figure(
        go.Ohlc(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing=dict(line=dict(color=VERDE_ALZA, width=1.4)),
            decreasing=dict(line=dict(color=ROJO_BAJA, width=1.4)),
            tickwidth=0.25,
        )
    )
    fig.update_layout(**_disenio_base())
    fig.update_xaxes(title_text="Fecha", rangeslider_visible=False, showgrid=True)
    fig.update_yaxes(title_text="Open/High/Low/Close", tickformat=",")
    return fig


def figura_modelo(simbolo: str, modelo: str) -> go.Figure:
    """Volatilidad predicha vs. realizada para un modelo y un activo."""
    ruta = PREDICCIONES / f"{modelo}_{simbolo}_predictions.csv"
    df = pd.read_csv(ruta, parse_dates=["timestamp"]).sort_values("timestamp")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["predicted_value"],
            name="Prediccion",
            mode="lines",
            line=dict(color=ROJO_PREDICCION, width=1.6, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["real_value"],
            name="Real",
            mode="lines",
            line=dict(color=NEGRO_REAL, width=2.6),
        )
    )

    disenio = _disenio_base()
    disenio["showlegend"] = True
    disenio["margin"] = dict(l=150, r=30, t=75, b=95)
    fig.update_layout(**disenio)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=FUENTE_LEYENDA, color=NEGRO_TEXTO, family=FUENTE),
            title_text="",
        )
    )
    fig.update_xaxes(title_text="Fecha/Hora")
    valores = pd.concat([df["predicted_value"], df["real_value"]])
    paso, formato = _escala_ticks(valores.max() - valores.min())
    fig.update_yaxes(
        title_text="Volatilidad Realizada (Cada 252 registros)",
        dtick=paso,
        tickformat=formato,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=FIGURAS,
        help="carpeta destino (por defecto proyecto/las_figuras)",
    )
    parser.add_argument(
        "--solo",
        choices=["precios", "modelos"],
        help="generar únicamente una de las dos familias de figuras",
    )
    args = parser.parse_args()

    destino_precios = args.outdir / "iniciales"
    destino_precios.mkdir(parents=True, exist_ok=True)

    if args.solo != "modelos":
        for nombre, (carpeta, _, _) in ACTIVOS.items():
            salida = destino_precios / f"{nombre}.png"
            figura_precio(carpeta).write_image(
                salida, width=ANCHO, height=ALTO, scale=ESCALA
            )
            print(f"  {salida.relative_to(args.outdir)}")

    if args.solo != "precios":
        for _, (_, simbolo, prefijo) in ACTIVOS.items():
            for sufijo, modelo in MODELOS.items():
                salida = args.outdir / f"{prefijo}_{sufijo}.png"
                figura_modelo(simbolo, modelo).write_image(
                    salida, width=ANCHO, height=ALTO, scale=ESCALA
                )
                print(f"  {salida.relative_to(args.outdir)}")


if __name__ == "__main__":
    main()
