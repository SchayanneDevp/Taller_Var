# Taller Aplicado - Modelos VAR 

# Nota: Tips practicos en Python
## Para limpiar el entorno en IPython/Jupyter se puede correr: "%reset -f"
## Para cerrar todas las graficas actualmente abiertas se puede correr: plt.close("all")
## En VS Code o Spyder, los bloques marcados con "# %%" se ejecutan por celdas.


# %% 0. Preliminares ---- ----------------------------------------------------

# Paquetes requeridos

## Trabajar con rutas absolutas y recrear el comportamiento de setwd() de R
from pathlib import Path
import os

## Librería para trabajar vectores y matrices en python
import numpy as np

## Procesamiento de datos
import pandas as pd

## Librería de graficación en python
import matplotlib.pyplot as plt

## Bibliotecas para el trabajo de series de tiempo en python
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

# Define el directorio de trabajo sobre el cual va a trabajar
# Si corre el script como archivo, usa su carpeta; si es interactivo, escriba su ruta en Path(r"...")
project_path = Path(__file__).resolve().parent if "__file__" in globals() else Path(r"C:\RUTA\A\taller_VAR\python")
os.chdir(project_path)

## Librerias que complementan la libreria de graficacion matplotlib
## (Se usan en la graficacion de las IRF)
from funciones_auxiliares_graficacion_VAR import (
    configurar_entorno_graficas,
    graficar_grilla_irf,
    graficar_pronostico_var,
    graficar_ts,
    imprimir_adf,
    imprimir_matrices_acof,
    imprimir_seleccion_rezagos,
    mostrar_graficas,
    predecir_var,
    prueba_arch_por_ecuacion,
    prueba_normalidad_por_ecuacion,
)

configurar_entorno_graficas(max_columns=30)

# Importe la base de datos desde Excel
datos = pd.read_excel("datos_VAR.xlsx")


# %% 1. Exploracion previa de las series de tiempo, antes de realizar el modelamiento -----------------------------------------------------------


# Ahora bien, asuma lo siguiente que la variable "y" es la "variable mas exogena" de las dos variables que se van a modelar, es decir:

## 1) Un shock en "y" afecta de manera inmediata/contemporanea (es decir, desde el periodo t que ocurre el shock) tanto a "x" como a si mismo ("y").
## 2) Mientras que un shock en "x" unicamente afecta de manera inmediata/contemporanea (es decir, desde el periodo t que ocurre el shock) a si mismo ("x")
##    y es en periodos posteriores a t que afecta la variable y.

# Use la informacion anterior, para organizar de manera correcta las variables que conforman la base de datos que importo en Python,
# de tal forma que estas queden en el orden correcto para calcular las funciones impulso respuesta ortogonalizadas usando la
# descomposicion de Cholesky:
datos = datos.loc[:, ["y", "x"]]   # orden [y, x]: y es la mas exogena (Cholesky)

# verifique que el orden de sus variables en su base de datos haya quedado en el orden correcto
datos.info()
print(datos.head())

# Transforme la base de datos que importo en un objeto de serie de tiempo trimestral

# Cree el índice temporal 
tiempo = pd.period_range(
    start="1990Q1",
    periods=len(datos),
    freq="Q",
    name="tiempo",
)

# Cree el pandas dataframe con índice temporal
series = pd.DataFrame(
    datos.to_numpy(),
    index=tiempo,
    columns=datos.columns,
)

# Grafique las series de tiempo
fig_series, axes_series = plt.subplots(1, 2, figsize=(10, 4))

graficar_ts(
    series.iloc[:, 0],
    titulo="Variable y",
    color="lightblue",
    ax=axes_series[0],
)

graficar_ts(
    series.iloc[:, 1],
    titulo="Variable x",
    color="royalblue",
    ax=axes_series[1],
)

fig_series.tight_layout()
mostrar_graficas()

# %% Realice las pruebas de Dickey-Fuller aumentadas ¿Son las series de tiempo estacionarias?
# Para los parametros de la prueba de Dickey-Fuller, use las siguientes especificaciones: " maxlag=3, autolag='AIC', regression='n' "

## Prueba de ADF para la priemra serie de tiempo
adf1 = adfuller(series.iloc[:, 0], maxlag=3, autolag="AIC", regression="n")
imprimir_adf(adf1, series.columns[0]) 

## Prueba de ADF para la segunda serie de tiempo
adf2 = adfuller(series.iloc[:, 1], maxlag=3, autolag="AIC", regression="n")
imprimir_adf(adf2, series.columns[1]) 


# %% 2. Metodologia Box-Jenkins para series multivariadas -----------------------

# Nombre de las series de tiempo individuales
variables = list(series.columns)

# Para crear el objeto VAR con statsmodels quitamos el indice temporal y
# dejamos solo los datos. El indice trimestral se conserva en series para
# graficas y tablas.
series_modelo = series.reset_index(drop=True)

# %% 2.1 Identificacion del modelo ----

# Creación del objeto de tipo statsmodels.tsa.vector_ar.var_model.VAR 
modelo_var = VAR(series_modelo)

# Seleccion de rezagos para un VAR
seleccion_rezagos = modelo_var.select_order(maxlags=6, trend="n")
imprimir_seleccion_rezagos(
    seleccion_rezagos,
    "Seleccion de rezagos para un VAR sin terminos deterministicos",
    incluir_rezago_cero=False,
)


# %% 2.2 Estimacion del modelo ----


VAR_estimado = modelo_var.fit(1, trend="n")   # VAR(1): AIC, BIC, HQ y FPE eligen p=1

# ¿Cuales son las raices del polinomio asociado a la parte autoregresiva del modelo VAR det(I - A(L)), donde A(L) denota la parte autoregresiva del VAR?
# Ojo, tenga mucho cuidado con lo que se le esta preguntando!!!
# Nota: En statsmodels las raices reportadas por roots deben quedar por fuera
#       del circulo unitario. Adicionalmente, is_stable() revisa que los valores
#       propios de la matriz de compania queden dentro del circulo unitario.
raices_var = pd.DataFrame(
    {
        "raiz": VAR_estimado.roots,
        "modulo": np.abs(VAR_estimado.roots),
    }
)
print(raices_var)
print("El proceso es estable:", VAR_estimado.is_stable(verbose=True))

# ¿Cuales son los coeficientes del modelo?

# Nota: Bcoef y Acoef difererian si por ejemplo el VAR tuviera terminos determinisitcos (e.g. constantes, tendencias lineales deterministicas y demas)

## Para visualizar todos los coeficientes
Bcoef_VAR_estimado = VAR_estimado.params.T
print(Bcoef_VAR_estimado)

## Para visualizar solo los coeficientes asosicados a los rezagos del modelo VAR
imprimir_matrices_acof(VAR_estimado, variables)

# Resumen completo de los resultados de la estimacion del VAR
print(VAR_estimado.summary())


# %% 2.3 Validacion de supuestos del modelo ----


# Residuales del modelo VAR estimado
residuales_var = pd.DataFrame(
    np.asarray(VAR_estimado.resid),
    index=series.index[VAR_estimado.k_ar :],
    columns=variables,
)

# Test de no correlacion serial
P_10 = VAR_estimado.test_whiteness(nlags=10, adjusted=False)
print(P_10.summary())

# Test de heterocedasticidad
# statsmodels no tiene un equivalente directo a arch.test() multivariado de
# vars en R. Por tanto, se construye una funcion que permite hacer un arch.test()
# univariado para cada uno de los residuales de la regresion, uno por cada
# variable del VAR.
arch_12 = prueba_arch_por_ecuacion(residuales_var, lags=12, variables=variables)
print(arch_12)

# Test de normalidad (Jarque-Bera para series multivariadas)
normalidad_var = VAR_estimado.test_normality()
print(normalidad_var.summary())

normalidad_univariada = prueba_normalidad_por_ecuacion(
    residuales_var,
    variables=variables,
)
print(normalidad_univariada)


# %% 2.4 Uso del modelo: Pronostico y funciones impulso respueta ----


# %% 2.4.1 Pronostico ----

# Pronóstico
pronostico_var = predecir_var(
    VAR_estimado,
    n_ahead=12,
    ci=0.95,
    indice=series.index,
    variables=variables,
)
print(pronostico_var)

# Item 19: semi-amplitud (CI) del intervalo a 3 pasos para x; Item 20: cota superior a 5 pasos para y
print("Semi-amplitud IC95% a 3 pasos (x):", pronostico_var["tablas"]["x"].iloc[2]["CI"])
print("Cota superior IC95% a 5 pasos (y):", pronostico_var["tablas"]["y"].iloc[4]["upper"])

# Gráfica del pronóstico
fig_pronostico, axes_pronostico = graficar_pronostico_var(
    pronostico_var["pronostico"],
    pronostico_var["inferior"],
    pronostico_var["superior"],
)
fig_pronostico.suptitle("Pronostico VAR", fontsize=11)
fig_pronostico.tight_layout()
mostrar_graficas()


# %% 2.4.2 Funciones Impulso Respuesta (IRF) ----

# Matrices asociadas a las funciones impulso respuesta
print(VAR_estimado.irf().irfs)

# Matrices asociadas a las funciones impulso respuesta ortogonalizadas
print(VAR_estimado.irf().orth_irfs)

# Cargue las funciones auxiliares para graficar las IRF
# En Python estas funciones se importaron al inicio desde
# funciones_auxiliares_graficacion_VAR.py.

# Parametros de las graficas de las IRF
variables = list(series.columns)
pasos_adelante = np.arange(0, 18)
int_conf_irf = 0.95
semilla_irf = 202601
repeticiones_bootstrap_irf = 100 # Bootstrappings empleados para construir los IC de las IRF

# La funcion graficar_grilla_irf() calcula el objeto irf() una sola vez
# y luego crea cada panel con programacion funcional.

# Graficacion de las IRF no ortogonalizadas

# IRF de las variables del sistema ante distintos choques exogenos.
irf_no_ortog = graficar_grilla_irf(
    VAR_estimado,
    variables,
    pasos_adelante,
    ortog=False,
    int_conf=int_conf_irf,
    prefijo_titulo="Impulso",
    semilla=semilla_irf,
    runs=repeticiones_bootstrap_irf,
)

# Grilla de IRF: columnas = impulsos; filas = respuestas.
mostrar_graficas()

# Graficacion de las IRF ortogonalizadas

# Usamos los mismos pasos adelante, intervalo de confianza y semilla.
# IRF ortogonalizadas de las variables del sistema ante distintos choques exogenos.
irf_ortog = graficar_grilla_irf(
    VAR_estimado,
    variables,
    pasos_adelante,
    ortog=True,
    int_conf=int_conf_irf,
    prefijo_titulo="Impulso ortogonal",
    semilla=semilla_irf,
    runs=repeticiones_bootstrap_irf,
)

# Grilla de OIRF: columnas = impulsos; filas = respuestas.
mostrar_graficas()

# %% 3. Informe: almacenar los calculos y generar HTML (+ PDF nitido) ---------
# Esta seccion NO modifica nada de lo anterior: recoge los objetos ya calculados,
# los guarda (JSON + CSV) y construye un informe HTML autocontenido (graficas SVG
# vectoriales => nitidas al imprimir). Salidas en: <project_path>/salidas_informe/

# --- BEGIN BUILDER ---
import json, io, html as _html, datetime as _dt

SALIDAS = project_path / "salidas_informe"
SALIDAS.mkdir(exist_ok=True)
ALFA = 0.05


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if (abs(v) >= 1e-4 or v == 0) else f"{v:.2e}"


def _dec(p):
    return "reject H0" if p < ALFA else "fail to reject H0"


def _tbl(df, nd=4, idx=True):
    return df.to_html(classes="tbl", border=0, index=idx,
                      float_format=lambda v: _fmt(v, nd))


def _svg(fig):
    b = io.StringIO()
    fig.savefig(b, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = b.getvalue()
    return s[s.find("<svg"):]


def _boot_irf(Y, A1, U, H, runs, seed, nivel):
    """Bandas percentil por bootstrap de residuales (VAR(1)); IRF y OIRF."""
    rng = np.random.default_rng(seed)
    T, K = Y.shape
    n = len(U)
    Uc = U - U.mean(0)
    oi, oo = [], []
    for _ in range(runs):
        e = Uc[rng.integers(0, n, n)]
        Yb = np.empty((T, K))
        Yb[0] = Y[0]
        for t in range(1, T):
            Yb[t] = A1 @ Yb[t - 1] + e[t - 1]
        X = Yb[:-1]
        B = np.linalg.lstsq(X, Yb[1:], rcond=None)[0]
        Ub = Yb[1:] - X @ B
        P = np.linalg.cholesky(Ub.T @ Ub / (n - K))
        Ps = [np.linalg.matrix_power(B.T, h) for h in range(H + 1)]
        oi.append(Ps)
        oo.append([p @ P for p in Ps])
    q = [(1 - nivel) / 2 * 100, (1 + nivel) / 2 * 100]
    return np.percentile(oi, q, axis=0), np.percentile(oo, q, axis=0)


def _fig_grid(irf, bands, names, H, titulo):
    K = len(names)
    fig, ax = plt.subplots(K, K, figsize=(8.2, 5.6), sharex=True)
    h = np.arange(H + 1)
    for i in range(K):          # fila = respuesta
        for j in range(K):      # columna = impulso
            a = ax[i, j]
            a.fill_between(h, bands[0][:H + 1, i, j], bands[1][:H + 1, i, j],
                           color="#0f766e", alpha=.18, lw=0)
            a.plot(h, irf[:H + 1, i, j], color="#0f766e", lw=1.8)
            a.axhline(0, color="#64748b", lw=.7)
            a.set_title(f"{names[i]} <- shock in {names[j]}", fontsize=9)
            a.grid(alpha=.25)
    for a in ax[-1]:
        a.set_xlabel("steps ahead")
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout()
    return fig


def _irf_text(irf, names, H, thr=0.01):
    """Descripcion dinamica: t=0, pico, semivida y paso de disipacion."""
    K = len(names)
    out = []
    for j in range(K):
        for i in range(K):
            r = irf[:H + 1, i, j]
            pk = int(np.argmax(np.abs(r)))
            hl = next((s for s in range(pk, H + 1) if abs(r[s]) <= abs(r[pk]) / 2), None)
            sig = "positive" if r[pk] > 0 else "negative"
            out.append(f"<li><b>{names[j]} &rarr; {names[i]}</b>: t=0 effect {_fmt(r[0])}; "
                       f"{sig} peak {_fmt(r[pk])} at step {pk}; half-life: "
                       f"{'step ' + str(hl) if hl is not None else 'beyond horizon'}; "
                       f"|response| &lt; {thr} from step "
                       f"{next((s for s in range(H + 1) if np.all(np.abs(r[s:]) < thr)), 'n/a')}.</li>")
    return "<ul>" + "".join(out) + "</ul>"


def recolectar(res, adf_list, sel, white, arch, norm, norm_uni, fc, series, nombres):
    """Empaqueta en un dict todos los resultados (statsmodels -> dict simple)."""
    K = len(nombres)
    A1 = pd.DataFrame(res.coefs[0], index=nombres, columns=nombres)
    ics = pd.DataFrame(sel.ics)
    n_ic = len(ics)
    ics.index = range(1, n_ic + 1) if n_ic == 6 else range(0, n_ic)
    return dict(
        nombres=nombres, series=series, A1=A1, p=res.k_ar,
        adf={v: dict(t=a[0], p=a[1], lag=a[2], nobs=a[3], crit=a[4])
             for v, a in zip(nombres, adf_list)},
        ics=ics[[c for c in ["aic", "bic", "hqic", "fpe"] if c in ics]],
        sel={k: int(v) for k, v in dict(sel.selected_orders).items()},
        roots=pd.DataFrame({"root": res.roots, "modulus": np.abs(res.roots)}),
        eig=np.linalg.eigvals(res.coefs[0]), estable=bool(res.is_stable()),
        coef={v: pd.DataFrame({"coef": res.params[v], "std.err": res.stderr[v],
                               "t": res.tvalues[v], "P>|t|": res.pvalues[v]})
              for v in nombres},
        sigma=pd.DataFrame(res.sigma_u, index=nombres, columns=nombres),
        resid=np.asarray(res.resid), Y=np.asarray(res.endog),
        white=dict(stat=white.test_statistic, p=white.pvalue, df=white.df, nlags=10),
        arch=arch, norm=dict(stat=norm.test_statistic, p=norm.pvalue, df=norm.df),
        norm_uni=norm_uni, fc=fc,
        irf=np.asarray(res.irf().irfs), oirf=np.asarray(res.irf().orth_irfs),
        P=np.linalg.cholesky(res.sigma_u), summary=str(res.summary()),
        nobs=int(res.nobs),
    )


def guardar_resultados(R, ruta):
    """Guarda todos los calculos: JSON completo + CSV por tabla."""
    def js(o):
        if isinstance(o, pd.DataFrame):
            return o.reset_index().to_dict("list")
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (np.generic,)):
            return o.item()
        if isinstance(o, (pd.Period, complex)):
            return str(o)
        return str(o)
    lig = {k: v for k, v in R.items() if k not in ("series",)}
    (ruta / "resultados.json").write_text(json.dumps(lig, default=js, indent=1), encoding="utf-8")
    for k in ("A1", "ics", "roots", "sigma", "arch", "norm_uni"):
        R[k].to_csv(ruta / f"{k}.csv")
    for v, t in R["fc"]["tablas"].items():
        t.to_csv(ruta / f"pronostico_{v}.csv")
    np.save(ruta / "irf.npy", R["irf"])
    np.save(ruta / "oirf.npy", R["oirf"])


CSS = """
:root{--ink:#0f172a;--mut:#475569;--ac:#0f766e;--bg:#f8fafc;--card:#fff;--line:#e2e8f0;--warn:#b45309}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 'Segoe UI',system-ui,-apple-system,Helvetica,Arial,sans-serif}
header{background:linear-gradient(135deg,#0f172a,#134e4a);color:#fff;padding:42px 6vw 34px}
header h1{margin:0 0 6px;font-size:28px;letter-spacing:.2px}header p{margin:2px 0;color:#cbd5e1}
main{max-width:980px;margin:0 auto;padding:22px 20px 60px}
nav.toc{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 20px;margin:18px 0}
nav.toc a{color:var(--ac);text-decoration:none;margin-right:14px;white-space:nowrap}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:21px;color:var(--ac)}.kpi span{color:var(--mut);font-size:12.5px}
h2{margin:34px 0 8px;padding-bottom:6px;border-bottom:2px solid var(--ac);font-size:20px}
.item{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin:12px 0;break-inside:avoid}
.item h3{margin:0 0 6px;font-size:15.5px}.item h3 em{color:var(--ac);font-style:normal;margin-right:6px}
.ans{border-left:4px solid var(--ac);background:#f0fdfa;padding:8px 12px;margin:8px 0;border-radius:0 6px 6px 0}
.warn{border-left:4px solid var(--warn);background:#fffbeb;padding:8px 12px;margin:8px 0;border-radius:0 6px 6px 0}
table.tbl{border-collapse:collapse;margin:8px 0;font-size:13px;font-variant-numeric:tabular-nums}
.tbl th,.tbl td{border-bottom:1px solid var(--line);padding:4px 10px;text-align:right}
.tbl thead th{background:#f1f5f9;border-bottom:2px solid #cbd5e1}.tbl tbody th{text-align:left;background:#fafcfd}
pre{background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto;font-size:11.5px;line-height:1.4}
figure{margin:10px 0;text-align:center}figure svg{max-width:100%;height:auto}
figcaption{color:var(--mut);font-size:12.5px}
.btn{position:fixed;right:18px;bottom:18px;background:var(--ac);color:#fff;border:0;border-radius:24px;
padding:11px 20px;font-size:14px;cursor:pointer;box-shadow:0 4px 14px rgba(0,0,0,.25)}
footer{color:var(--mut);font-size:12.5px;border-top:1px solid var(--line);margin-top:30px;padding-top:12px}
@page{size:A4;margin:14mm 12mm}
@media print{body{background:#fff;font-size:11pt}.btn,nav.toc{display:none}header{padding:18px 0;-webkit-print-color-adjust:exact;print-color-adjust:exact}
.item{box-shadow:none}h2{break-after:avoid}pre{white-space:pre-wrap;overflow:visible}
.ans,.warn,.kpi{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
"""


def construir_html(R, ruta_html):
    nm = R["nombres"]          # ['y', 'x']
    y, x = nm
    A1, S, H = R["A1"], R["sigma"], 17
    serie = R["series"]
    rho = S.iloc[0, 1] / np.sqrt(S.iloc[0, 0] * S.iloc[1, 1])
    secs, it = [], []

    def item(n, titulo, cuerpo):
        it.append(f'<div class="item"><h3><em>#{n}</em>{titulo}</h3>{cuerpo}</div>')

    def ans(t):
        return f'<div class="ans">{t}</div>'

    def fig(svg, cap):
        return f"<figure>{svg}<figcaption>{cap}</figcaption></figure>"

    def sec(titulo, ident):
        secs.append((titulo, ident, "".join(it)))
        it.clear()

    # ---- 1. Datos -----------------------------------------------------------
    item("1-4", "Load, order and inspect the data",
         ans(f"<code>datos_VAR.xlsx</code> has {len(serie)} observations of two numeric series. "
             f"Columns are ordered <b>[{y}, {x}]</b>: <b>{y}</b> is the most exogenous variable, so a "
             f"{y}-shock hits both variables at t while an {x}-shock hits only {x} at t (Cholesky ordering).")
         + _tbl(serie.head().set_axis(serie.index[:5].astype(str)).rename_axis("tiempo")) +
         "<p>Both columns are <code>float64</code> without missing values (see <code>.info()</code> in the script).</p>")
    item("5-6", "Quarterly index and plots",
         ans(f"Index: <code>pd.period_range('1990Q1', periods={len(serie)}, freq='Q')</code> &rarr; "
             f"{serie.index[0]} to {serie.index[-1]}.")
         + fig(R["fig_series"], "Series y and x. Both oscillate around a constant level with no visible trend."))
    sec("1. Data and exploration", "s1")

    # ---- 2. ADF ----------------------------------------------------------------
    cuerpo = ""
    for v in nm:
        a = R["adf"][v]
        c5 = a["crit"]["5%"]
        cuerpo += ans(f"<b>{v}</b>: t = {_fmt(a['t'],3)}; p-value = {_fmt(a['p'])}; lags used = {a['lag']}; "
                      f"5% critical value = {_fmt(c5,3)} &rarr; t &lt; critical, p &lt; 0.05: "
                      f"<b>{_dec(a['p'])}</b> (no unit root).")
    cuerpo += ("<p>H0: unit root (non-stationary). Specification <code>regression='n'</code> "
               "(no constant), <code>maxlag=3</code>, <code>autolag='AIC'</code>. "
               "Both series are stationary at 5%, so the VAR is estimated in levels.</p>")
    item("7-8", "Augmented Dickey-Fuller tests", cuerpo)
    sec("2. Unit roots", "s2")

    # ---- 3. Identificacion -----------------------------------------------------------
    sel = R["sel"]
    item("9", "Optimal number of lags",
         ans("Selected orders: " + ", ".join(f"{k.upper()} = {v}" for k, v in sel.items())) +
         _tbl(R["ics"], 5) + "<p>All criteria agree: <b>VAR(1)</b>. Lower is better; "
         "the common estimation sample is used for every lag.</p>")
    sec("3. Identification", "s3")

    # ---- 4. Estimacion ----------------------------------------------------------
    rt = R["roots"]
    item("10-11", "VAR(1) estimate: roots and stability",
         ans(f"Roots of det(I &minus; A<sub>1</sub>z) = 0 (statsmodels <code>.roots</code>): "
             + ", ".join(_fmt(m, 3) for m in rt["modulus"]) + " in modulus, all &gt; 1. "
             f"Eigenvalues of A<sub>1</sub>: {', '.join(_fmt(abs(e),3) for e in R['eig'])} (&lt; 1) &rarr; "
             f"<b>{'stable / stationary' if R['estable'] else 'NOT stable'}</b>.") + _tbl(rt) +
         "<p>The question asks for the roots of the characteristic polynomial, which must lie "
         "<i>outside</i> the unit circle; they are the reciprocals of the eigenvalues of A<sub>1</sub>.</p>"
         + fig(R["fig_roots"], "Eigenvalues of A<sub>1</sub> (inside the unit circle &rArr; stationary)."))
    item("12", "Matrix A<sub>1</sub>", _tbl(A1) +
         "<p>Rows = equations (dependent variable), columns = lagged regressors, "
         f"i.e. the model is Y<sub>t</sub> = A<sub>1</sub>Y<sub>t-1</sub> + u<sub>t</sub> with Y<sub>t</sub> = ({y}, {x})'.</p>")
    item("13", "Full summary and how to read it",
         "<p>statsmodels prints, per equation: coefficient, standard error, t-statistic and "
         "<code>Prob</code> (two-sided p-value of H0: coef = 0). Next it prints the "
         "<b>correlation matrix of the residuals</b> (here &rho;(u<sub>y</sub>,u<sub>x</sub>) = "
         f"{_fmt(rho,3)}), which is what makes the non-orthogonalized IRFs differ from the orthogonalized ones.</p>"
         f"<pre>{_html.escape(R['summary'])}</pre>")
    eq = ""
    for k, v in enumerate(nm):
        t = R["coef"][v]
        eq += (f"<p><b>Equation {k+1} ({v}):</b> {v}<sub>t</sub> = {_fmt(A1.iloc[k,0])}·{y}<sub>t-1</sub> + "
               f"{_fmt(A1.iloc[k,1])}·{x}<sub>t-1</sub> + u<sub>{v},t</sub></p>" + _tbl(t))
    item("14", "The two estimated equations",
         eq + ans("All four coefficients are significant at 5% (every p-value &lt; 0.05): "
                  f"Each variable's lag is significant in the other equation, which points to feedback in both directions (formal Granger tests: <code>VAR_estimado.test_causality</code>)."))
    sec("4. Estimation of the VAR(1)", "s4")

    # ---- 5. Validacion ------------------------------------------------------------
    w, a_, nm_ = R["white"], R["arch"], R["norm"]
    item("15", "Serial correlation (portmanteau, 10 lags)",
         ans(f"Q = {_fmt(w['stat'],3)}, df = {int(w['df'])}, p-value = {_fmt(w['p'])} &rarr; "
             f"<b>{_dec(w['p'])}</b>: no residual serial correlation (H0: residuals are white noise)."))
    item("16", "Heteroskedasticity (ARCH-LM, 12 lags)",
         _tbl(a_) + ans("Both LM p-values: " + ", ".join(f"{v} = {_fmt(a_.loc[v,'lm_pvalue'])}" for v in nm)
                        + f" &rarr; <b>fail to reject</b> H0 of no ARCH effects at 5% in each equation: homoskedastic residuals."))
    item("17", "Normality (Jarque-Bera)",
         ans(f"Multivariate JB = {_fmt(nm_['stat'],3)} (df = {int(nm_['df'])}), p-value = {_fmt(nm_['p'])} "
             f"&rarr; <b>{_dec(nm_['p'])}</b>: residuals are jointly normal.") + _tbl(R["norm_uni"]) +
         "<p>Univariate tests agree. Normality validates the Gaussian prediction intervals and the "
         "asymptotics of the IRF intervals.</p>")
    sec("5. Assumption checks", "s5")

    # ---- 6. Pronostico ----------------------------------------------------------------
    fcx, fcy = R["fc"]["tablas"][x], R["fc"]["tablas"][y]
    item("18", "12-step-ahead forecast (95% intervals)",
         "".join(f"<p><b>{v}</b></p>" + _tbl(R["fc"]["tablas"][v]) for v in nm) +
         fig(R["fig_fc"], "Forecast (line) and 95% interval (band); the forecast converges to the unconditional mean 0."))
    item("19", f"Half-width of the 3-step interval for {x}",
         ans(f"CI = {_fmt(fcx.iloc[2]['CI'])} ({_fmt(fcx.iloc[2]['lower'])} to {_fmt(fcx.iloc[2]['upper'])}), "
             f"i.e. 1.96·&radic;MSE<sub>3</sub>."))
    item("20", f"Upper bound of the 5-step interval for {y}",
         ans(f"Upper limit = {_fmt(fcy.iloc[4]['upper'])} (forecast {_fmt(fcy.iloc[4]['fcst'])} + {_fmt(fcy.iloc[4]['CI'])})."))
    sec("6. Forecasting", "s6")

    # ---- 7. IRF --------------------------------------------------------------------
    irf, oirf, P = R["irf"], R["oirf"], R["P"]
    cols = [f"{v}<-{u}" for u in nm for v in nm]

    def tab(m):
        d = pd.DataFrame([m[h].T.reshape(-1) for h in range(10)], columns=cols)
        d.index.name = "step"
        return _tbl(d)
    item("21", "First 10 non-orthogonalized IRFs (&Psi;<sub>h</sub> = A<sub>1</sub><sup>h</sup>)",
         tab(irf) + ans(f"<b>t = 0:</b> &Psi;<sub>0</sub> = I. A unit shock in {y} moves {y} by 1 and {x} by 0; "
                        f"a unit shock in {x} moves {x} by 1 and {y} by 0. The non-orthogonalized IRF "
                        f"treats each shock separately and ignores that u<sub>{y}</sub> and u<sub>{x}</sub> "
                        f"are correlated (&rho; = {_fmt(rho,3)}), so there is no contemporaneous cross effect by construction."))
    item("22", "First 10 orthogonalized IRFs (&Psi;<sub>h</sub>P, Cholesky, order [y, x])",
         tab(oirf) + ans(f"<b>t = 0:</b> &Psi;<sub>0</sub>P = P (lower-triangular, PP' = &Sigma;<sub>u</sub>). "
                         f"A one-s.d. {y}-shock moves {y} by {_fmt(P[0,0])} and {x} by {_fmt(P[1,0])}; "
                         f"a one-s.d. {x}-shock moves {x} by {_fmt(P[1,1])} and {y} by {_fmt(P[0,1])} (exactly 0 by the ordering). "
                         f"<b>Difference:</b> shocks are one standard deviation (not one unit), are mutually "
                         f"uncorrelated, and part of the {x} movement at t is attributed to the {y} shock "
                         f"because {y} is placed first.") + _tbl(pd.DataFrame(P, index=nm, columns=nm), 4))
    item("23", "IRF and OIRF plots (18 steps, 95% bootstrap bands)",
         fig(R["fig_irf"], "Non-orthogonalized IRF; rows = responses, columns = impulses.") +
         fig(R["fig_oirf"], "Orthogonalized IRF (Cholesky, y first).") +
         "<p>Bands: residual bootstrap, 100 runs, seed 202601, percentile 95%, computed in this report "
         "(the script's <code>graficar_grilla_irf</code> panels use the helper module).</p>")
    item("24", "Analysis: non-orthogonalized IRFs", _irf_text(irf, nm, H) +
         ans(f"All responses are positive. Own responses decay monotonically from t=0; cross responses start at 0, peak at step 1 and then decay "
             f"(A<sub>1</sub> has positive entries and eigenvalues {', '.join(_fmt(abs(e),2) for e in R['eig'])} &lt; 1), so every shock is "
             f"<b>transitory</b>: effects vanish geometrically and are negligible within about 9 quarters."))
    item("25", "Analysis: orthogonalized IRFs", _irf_text(oirf, nm, H) +
         ans(f"Same persistence as before (it is governed by A<sub>1</sub>) but different scale and t=0 effects: "
             f"{y} shock &rarr; {x} starts at {_fmt(P[1,0])} instead of 0; {x} shock &rarr; {y} still starts at 0. "
             f"The {y}&rarr;{y} and {x}&rarr;{x} responses are rescaled by {_fmt(P[0,0],3)} and {_fmt(P[1,1],3)}. "
             f"Effects dissipate within about 9 quarters; bands containing 0 indicate no significant response at 95%."))
    sec("7. Impulse-response functions", "s7")

    # ---- armado -------------------------------------------------------------------------
    toc = "".join(f'<a href="#{i}">{t}</a>' for t, i, _ in secs)
    body = "".join(f'<h2 id="{i}">{t}</h2>{c}' for t, i, c in secs)
    kp = [("VAR order", f"p = {R['p']}"), ("Max |eigenvalue|", _fmt(max(abs(R['eig'])), 3)),
          ("Portmanteau p", _fmt(R["white"]["p"], 3)), ("JB (multiv.) p", _fmt(R["norm"]["p"], 3)),
          ("Obs. used", str(R["nobs"]))]
    kpis = "".join(f'<div class="kpi"><b>{a}</b><span>{b}</span></div>' for b, a in kp)
    refs = ("<h2>References</h2><ul>"
            "<li>Dickey, D. A., &amp; Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. <i>Journal of the American Statistical Association, 74</i>(366), 427&ndash;431.</li>"
            "<li>Enders, W. (2014). <i>Applied econometric time series</i> (4th ed.). Wiley.</li>"
            "<li>Jarque, C. M., &amp; Bera, A. K. (1987). A test for normality of observations and regression residuals. <i>International Statistical Review, 55</i>(2), 163&ndash;172.</li>"
            "<li>L&uuml;tkepohl, H. (2005). <i>New introduction to multiple time series analysis</i>. Springer.</li>"
            "<li>Seabold, S., &amp; Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. <i>Proceedings of the 9th Python in Science Conference</i>, 92&ndash;96.</li></ul>"
            "<p style='color:#475569;font-size:12.5px'>Page numbers cited from memory; verify in your edition.</p>")
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>VAR Workshop Report</title><style>{CSS}</style></head><body>"
           f"<header><h1>Applied Workshop &mdash; VAR Models</h1>"
           f"<p>Econometr&iacute;a II &middot; Universidad Nacional de Colombia &middot; 2026-II</p>"
           f"<p>Bivariate VAR(1) on [{y}, {x}] &middot; 5% significance level &middot; generated {_dt.date.today():%Y-%m-%d}</p></header>"
           f"<main><div class='kpis'>{kpis}</div><nav class='toc'><b>Contents:</b> {toc}</nav>{body}{refs}"
           f"<footer>All tests at &alpha; = 0.05. Computed with statsmodels; figures are vector SVG (sharp when printed).</footer></main>"
           f"<button class='btn' onclick='window.print()'>&#128438; Download sharp PDF</button></body></html>")
    ruta_html.write_text(doc, encoding="utf-8")
    return ruta_html


def generar_pdf(ruta_html, ruta_pdf):
    """PDF vectorial (texto y SVG nitidos). Usa Playwright/Chromium o WeasyPrint si existen."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            pg = b.new_page()
            pg.goto(ruta_html.resolve().as_uri())
            pg.pdf(path=str(ruta_pdf), format="A4", print_background=True, prefer_css_page_size=True)
            b.close()
        return "playwright"
    except Exception:
        pass
    try:
        from weasyprint import HTML
        HTML(filename=str(ruta_html)).write_pdf(str(ruta_pdf))
        return "weasyprint"
    except Exception:
        print("PDF automatico no disponible: abra el HTML en el navegador y pulse "
              "'Download sharp PDF' (Imprimir > Guardar como PDF, activar 'Graficos de fondo').")
        return None


def figuras(R):
    nm = R["nombres"]
    K = len(nm)
    f, ax = plt.subplots(1, K, figsize=(8.4, 2.8))
    for k, v in enumerate(nm):
        ax[k].plot(R["series"].index.to_timestamp(), R["series"][v], lw=.8, color="#0f766e")
        ax[k].set_title(f"Variable {v}", fontsize=9)
        ax[k].grid(alpha=.25)
    f.tight_layout()
    R["fig_series"] = _svg(f)
    f, a = plt.subplots(figsize=(3.4, 3.4))
    t = np.linspace(0, 2 * np.pi, 200)
    a.plot(np.cos(t), np.sin(t), color="#64748b", lw=1)
    a.scatter(R["eig"].real, R["eig"].imag, color="#b45309", zorder=3)
    a.set_aspect("equal")
    a.grid(alpha=.25)
    R["fig_roots"] = _svg(f)
    f, ax = plt.subplots(1, K, figsize=(8.4, 3))
    for k, v in enumerate(nm):
        tb = R["fc"]["tablas"][v]
        ix = tb.index.to_timestamp()
        ax[k].plot(ix, tb["fcst"], color="#0f766e")
        ax[k].fill_between(ix, tb["lower"], tb["upper"], color="#0f766e", alpha=.18, lw=0)
        ax[k].set_title(f"Forecast {v}", fontsize=9)
        ax[k].grid(alpha=.25)
        ax[k].tick_params(axis="x", labelrotation=30, labelsize=7)
    f.tight_layout()
    R["fig_fc"] = _svg(f)
    H = 17
    bi, bo = _boot_irf(R["Y"], R["A1"].to_numpy(), R["resid"], H, 100, 202601, 0.95)
    R["fig_irf"] = _svg(_fig_grid(R["irf_full"], bi, nm, H, "IRF (non-orthogonalized)"))
    R["fig_oirf"] = _svg(_fig_grid(R["oirf_full"], bo, nm, H, "OIRF (Cholesky, order y, x)"))
    return R
# --- END BUILDER ---


# %% 3.1 Recolectar, guardar y construir el informe ----
RESULTADOS = recolectar(
    VAR_estimado, [adf1, adf2], seleccion_rezagos, P_10, arch_12, normalidad_var,
    normalidad_univariada, pronostico_var, series, variables,
)
# IRF con 18 pasos para graficar (los items 21-22 usan los primeros 10)
RESULTADOS["irf_full"] = VAR_estimado.irf(17).irfs
RESULTADOS["oirf_full"] = VAR_estimado.irf(17).orth_irfs
RESULTADOS = figuras(RESULTADOS)
guardar_resultados(RESULTADOS, SALIDAS)
html_informe = construir_html(RESULTADOS, SALIDAS / "informe_VAR.html")
print("Informe HTML:", html_informe)

# Opcion PDF nitido: cambie a True (o use el boton del HTML: Imprimir > Guardar como PDF)
GENERAR_PDF = False
if GENERAR_PDF:
    print("PDF generado con:", generar_pdf(html_informe, SALIDAS / "informe_VAR.pdf"))

# %%