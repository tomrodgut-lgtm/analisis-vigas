#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISIS DE VIGAS — App web con Streamlit
(unidades: kN, m, GPa; sección en cm)
=========================================================================
Conversión a app web de tu programa de consola. Las fórmulas y algoritmos
son EXACTAMENTE los de tu código: solo se reemplazaron los input() por
campos de formulario, los print() se capturan para mostrar el procedimiento
real, y matplotlib se cambió por Plotly.

CONVENCIONES
  x → derecha desde el extremo izquierdo. Cargas (w, P) + hacia ABAJO.
  Momentos aplicados y reacciones de momento + ANTIHORARIOS (↺).
  Reacciones verticales + hacia ARRIBA. M(+) tracciona la fibra inferior.
  y (flecha) + hacia arriba; θ (giro de la sección) + antihorario.

Ejecutar:  streamlit run app.py
"""
import io
import inspect
import contextlib
import numpy as np
import pandas as pd
import sympy as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

SUP = {1: "E", 2: "A", 3: "L"}
NOM = {"E": "Empotrado", "A": "Apoyo simple", "L": "Libre"}
_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

# Paleta: azul marino, blanco, gris claro y azul acero (acento)
COL = dict(navy="#0A1F44", steel="#4682B4", gris="#7A8CA3", claro="#E8EDF3",
           rojo="#C0392B", cobre="#C96F3B")
FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"


# ════════════════════════════════════════════════════════════════════
# UTILIDADES  (idénticas a tu código; solo se quitó pedir() porque ya
# no hay input() — los datos llegan desde el formulario)
# ════════════════════════════════════════════════════════════════════

def titulo(txt):
    print("\n" + "═" * 80 + f"\n  {txt}\n" + "═" * 80)


def sub(txt):
    print("\n── " + txt + " " + "─" * max(0, 74 - len(txt)))


def sup(n):
    return str(n).translate(_SUP)


def pm(M, ancho=12, pref="    "):
    M = np.atleast_2d(np.asarray(M, float))
    for fila in M:
        print(pref + "[ " + " ".join(f"{v:{ancho}.5g}" for v in fila) + " ]")


def caja(filas, pref="    "):
    """Matriz con corchetes grandes (⎡ ⎢ ⎣) a partir de textos."""
    n, m = len(filas), len(filas[0])
    w = [max(len(f[j]) for f in filas) for j in range(m)]
    for i, f in enumerate(filas):
        if n == 1:
            l, r = "[", "]"
        else:
            l, r = ("⎡", "⎤") if i == 0 else (("⎣", "⎦") if i == n - 1 else ("⎢", "⎥"))
        print(pref + l + " " + "  ".join(c.center(w[j]) for j, c in enumerate(f)) + " " + r)


def simb(e):
    return str(e).replace("**", "^").replace("^2", "²").replace("^3", "³") \
        .replace("^4", "⁴").replace("^5", "⁵").replace("^6", "⁶").replace("*", "·")


def gauss_pasos(A, b, nombres, pref="    "):
    """Eliminación de Gauss mostrando cada operación entre filas."""
    n = len(b)
    M = np.hstack([np.array(A, float), np.array(b, float).reshape(-1, 1)])

    def show():
        for fila in M:
            print(pref + "[ " + " ".join(f"{v:11.5g}" for v in fila[:n]) +
                  " | " + f"{fila[n]:11.5g} ]")

    print("  Matriz aumentada [A | b]:")
    show()
    for k in range(n):
        p = k + int(np.argmax(np.abs(M[k:, k])))
        if abs(M[p, k]) < 1e-14:
            raise ValueError("Sistema singular: la estructura es inestable.")
        if p != k:
            M[[k, p]] = M[[p, k]]
            print(f"\n  Pivoteo: intercambio F{k+1} ↔ F{p+1}")
        ops = False
        for i in range(k + 1, n):
            if abs(M[i, k]) < 1e-14:
                continue
            m = M[i, k] / M[k, k]
            print(f"  F{i+1} ← F{i+1} − ({m:.6g})·F{k+1}   [multiplicador m = {M[i,k]:.5g}/{M[k,k]:.5g}]")
            M[i, :] -= m * M[k, :]
            M[i, k] = 0.0
            ops = True
        if ops:
            print(f"  Tras eliminar la columna {k+1}:")
            show()
    print("\n  Sustitución hacia atrás (de la última fila a la primera):")
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = M[i, n]
        txt = f"{M[i, n]:.6g}"
        for j in range(i + 1, n):
            txt += f" − ({M[i, j]:.6g})({x[j]:.6g})"
            s -= M[i, j] * x[j]
        x[i] = s / M[i, i]
        print(f"    {nombres[i]} = [ {txt} ] / {M[i, i]:.6g} = {x[i]:.6g}")
    return x


# ── Funciones de singularidad: término = (coef, a, n) ≡ coef·<x−a>^n ──
def integ(terms):
    return [(c / (n + 1), a, n + 1) for c, a, n in terms]


def fmt(terms):
    partes = []
    for c, a, n in terms:
        c = sp.expand(c)
        if c.free_symbols:
            s = str(sp.N(c, 5)).replace("*", "·")
            if " " in s:
                signo, mag = "+", "(" + s + ")"
            else:
                signo, mag = ("-", s[1:]) if s.startswith("-") else ("+", s)
        else:
            val = float(c)
            if abs(val) < 1e-12:
                continue
            signo = "-" if val < 0 else "+"
            mag = f"{abs(val):.5g}"
        aa = float(a)
        if aa == 0:
            var = "" if n == 0 else ("x" if n == 1 else "x" + sup(n))
        else:
            var = f"⟨x−{aa:.5g}⟩{sup(n)}"
        pieza = var if (mag == "1" and var) else mag + ("·" + var if var else "")
        partes.append((signo, pieza))
    if not partes:
        return "0"
    s = ("−" if partes[0][0] == "-" else "") + partes[0][1]
    for sg, pc in partes[1:]:
        s += f" {'−' if sg == '-' else '+'} {pc}"
    return s


def mostrar_int(terms, pref="    "):
    """Imprime la integración término a término: ∫ c·⟨x−a⟩ⁿ dx = c/(n+1)·⟨x−a⟩ⁿ⁺¹"""
    for c, a, n in terms:
        if sp.expand(c) == 0:
            continue
        orig = fmt([(c, a, n)])
        nuevo = fmt(integ([(c, a, n)]))
        cs = fmt([(c, 0, 0)])
        var = "x" if float(a) == 0 else f"⟨x−{float(a):.5g}⟩"
        print(f"{pref}∫ ({orig}) dx = ({cs})/{n+1} · {var}{sup(n+1)} = {nuevo}")


def ev_sym(terms, xv):
    return sum((c * (xv - a) ** n for c, a, n in terms if xv >= a), sp.Integer(0))


def num_terms(terms):
    return [(float(c), float(a), n) for c, a, n in terms if abs(float(c)) > 1e-14]


def ev_num(terms, x):
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    for c, a, n in terms:
        out += np.where(x >= a - 1e-12, c * (x - a) ** n, 0.0)
    return out


# ════════════════════════════════════════════════════════════════════
# 1) SECCIÓN: PRIMER MOMENTO DE ÁREA Q(y) Y SU DIAGRAMA
# ════════════════════════════════════════════════════════════════════
def seccion(op, v):
    """Igual que tu seccion(), pero los valores vienen del formulario (dict v)."""
    titulo("PASO 0 · SECCIÓN: PRIMER MOMENTO DE ÁREA (Q), CENTROIDE E INERCIA")
    rects = []
    if op == 1:
        b = v["b"]
        h = v["h"]
        rects = [(b, h, 0.0)]
    elif op in (2, 3):
        bf = v["bf"]
        tf = v["tf"]
        hw = v["hw"]
        tw = v["tw"]
        if op == 2:
            rects = [(tw, hw, 0.0), (bf, tf, hw)]
        else:
            rects = [(bf, tf, 0.0), (tw, hw, tf), (bf, tf, tf + hw)]
    elif op == 4:
        rects = [(float(b), float(h), float(y0)) for b, h, y0 in v["rects"]]
    else:
        A = v["A"]
        I = v["I"]
        return dict(A=A * 1e-4, I=I * 1e-8, geom=False)

    sub("1) DEFINICIONES (de dónde salen las fórmulas)")
    print("  Primer momento de área (momento estático) de un área A respecto a un eje:")
    print("      Q = ∫ y dA   → para figuras compuestas:  Q = Σ Aᵢ·ȳᵢ")
    print("  El centroide es el punto donde Q = 0:  ∫ (y − ȳ) dA = 0  ⇒  ȳ = ∫y dA / ∫dA = Q_base / A")
    print("  Inercia: I = ∫ (y−ȳ)² dA . Con y−ȳ = d + η (d = distancia entre ejes, η medida desde el")
    print("  centroide propio): ∫(d+η)² dA = ∫η² dA + 2d∫η dA + d²A = Io + A·d²  porque ∫η dA = 0")
    print("  (Teorema de Steiner). Para un rectángulo: Io = ∫ η² b dη (de −h/2 a h/2) = b·h³/12.")
    print("  Esfuerzo cortante: del equilibrio de una rebanada dx, dF = (dM/I)·Q ⇒ τ = V·Q/(I·b),")
    print("  donde Q(y) = momento estático, respecto al eje neutro, del área ubicada ENCIMA del nivel y.")

    sub("2) Dividir en rectángulos (y medida desde la base)")
    SA = SAy = 0.0
    for i, (b, h, y0) in enumerate(rects, 1):
        Ai, yi = b * h, y0 + h / 2
        SA += Ai
        SAy += Ai * yi
        print(f"  Rect {i}: A{i} = b·h = {b:g}·{h:g} = {Ai:.4f} cm² | ȳ{i} = y0 + h/2 = {y0:g} + {h:g}/2"
              f" = {yi:.4f} cm | A{i}·ȳ{i} = {Ai*yi:.4f} cm³")
    yc = SAy / SA
    print(f"\n  Q_base = Σ Aᵢ·ȳᵢ = {SAy:.4f} cm³    ΣA = {SA:.4f} cm²")
    print(f"  ȳ = Q_base / ΣA = {SAy:.4f} / {SA:.4f} = {yc:.4f} cm (desde la base)")

    sub("3) Inercia centroidal  I = Σ (b·h³/12 + A·d²)")
    I = 0.0
    for i, (b, h, y0) in enumerate(rects, 1):
        Ai, yi = b * h, y0 + h / 2
        Io, di = b * h ** 3 / 12, yi - yc
        I += Io + Ai * di ** 2
        print(f"  Rect {i}: Io = {b:g}·{h:g}³/12 = {Io:.4f} | d = {yi:.4f} − {yc:.4f} = {di:.4f}"
              f" | A·d² = {Ai:.4f}·({di:.4f})² = {Ai*di**2:.4f}  →  {Io + Ai*di**2:.4f} cm⁴")
    print(f"  I = {I:.4f} cm⁴")

    H = max(y0 + h for _, h, y0 in rects)

    def Q_en(y):
        Q, piezas = 0.0, []
        for i, (b, h, y0) in enumerate(rects, 1):
            lo, hi = max(y0, y), y0 + h
            if hi > lo + 1e-12:
                Ap, yp = b * (hi - lo), (lo + hi) / 2 - yc
                Q += Ap * yp
                piezas.append((i, b, hi - lo, Ap, yp))
        return Q, piezas

    def b_en(y):
        return sum(b for b, h, y0 in rects if y0 < y < y0 + h)

    sub("4) Q(y) = A'(y)·(ȳ'(y) − ȳ)   [área por encima del nivel y, respecto al eje neutro]")
    print("  Se evalúa en cada nivel donde cambia la sección y en el eje neutro. En cada rectángulo:")
    print("  A' = b·(y_sup − y_inf) ,  ȳ' − ȳ = (y_inf + y_sup)/2 − ȳ ,  Q = Σ A'·(ȳ' − ȳ)")
    niveles = sorted(set([0.0, H, yc] + [y0 for _, _, y0 in rects] + [y0 + h for _, h, y0 in rects]))
    for y in niveles:
        Q, pz = Q_en(y)
        det = " + ".join(f"[{b:g}·{dh:.3f}={Ap:.3f}]·({yp:+.3f})" for _, b, dh, Ap, yp in pz) or "0"
        tag = "  ← EJE NEUTRO (Q máximo)" if abs(y - yc) < 1e-9 else ""
        bd, bu = b_en(y - 1e-7), b_en(y + 1e-7)
        bs = f"b={bu:g}" if abs(bd - bu) < 1e-9 else f"b={bd:g} (abajo) / {bu:g} (arriba)"
        print(f"  y = {y:8.3f} cm: Q = {det} = {Q:9.4f} cm³   {bs}{tag}")
    Qna = Q_en(yc)[0]
    bw = b_en(yc)
    if bw == 0:
        bw = max(b_en(yc - 1e-7), b_en(yc + 1e-7))
    print(f"\n  Q_EN = {Qna:.4f} cm³ | ancho en el eje neutro b = {bw:g} cm")
    return dict(A=SA * 1e-4, I=I * 1e-8, geom=True, c=max(yc, H - yc) * 1e-2,
                Q=Qna * 1e-6, bna=bw * 1e-2, rects=rects, yc=yc, H=H, Q_en=Q_en, b_en=b_en,
                niveles=niveles)


def _grid_y(sec):
    H = sec["H"]
    pts = list(np.linspace(1e-6, H - 1e-6, 800))
    for _, h, y0 in sec["rects"]:
        for yb in (y0, y0 + h):
            for dy in (-1e-6, 1e-6):
                if 1e-6 <= yb + dy <= H - 1e-6:
                    pts.append(yb + dy)
    return np.array(sorted(set(pts)))


# ── Gráficos de la sección (Plotly en lugar de matplotlib) ─────────
def _rgba(hexcol, a):
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def _layout(fig, height):
    fig.update_layout(template="plotly_white", height=height, showlegend=False,
                      font=dict(family=FONT, size=13, color=COL["navy"]),
                      margin=dict(l=60, r=30, t=80, b=55),
                      paper_bgcolor="white", plot_bgcolor="white",
                      hoverlabel=dict(font_family=FONT, bgcolor="white"))
    fig.update_annotations(font=dict(family=FONT, size=13, color=COL["navy"]))
    return fig


def _dibujar_seccion(fig, sec):
    rects, yc = sec["rects"], sec["yc"]
    bmax = max(b for b, _, _ in rects)
    cols = ["#9DB9D6", "#C5D3E3", "#7FA0C4", "#B7C4D6", "#D9E2EC"]
    for i, (b, h, y0) in enumerate(rects):
        fig.add_trace(go.Scatter(
            x=[-b / 2, b / 2, b / 2, -b / 2, -b / 2], y=[y0, y0, y0 + h, y0 + h, y0],
            mode="lines", fill="toself", fillcolor=cols[i % 5],
            line=dict(color=COL["navy"], width=1.5), hoveron="fills", hoverinfo="text",
            text=f"<b>Rect {i+1}</b><br>b = {b:g} cm · h = {h:g} cm<br>"
                 f"y: {y0:g} → {y0+h:g} cm<br>A = {b*h:.2f} cm²"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[0], y=[y0 + h / 2], mode="text", text=[f"{i+1}"],
                                 hoverinfo="skip"), row=1, col=1)
    fig.add_annotation(x=-bmax * 0.72, y=yc, text=f"ȳ = {yc:.2f} cm", showarrow=False,
                       xanchor="left", yanchor="bottom", font=dict(color=COL["rojo"]),
                       row=1, col=1)
    fig.update_xaxes(range=[-bmax * 0.75, bmax * 0.75], row=1, col=1)


def fig_Q(sec):
    ys = _grid_y(sec)
    Q = np.array([sec["Q_en"](y)[0] for y in ys])
    bb = np.array([sec["b_en"](y) for y in ys])
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=("Sección (cm)", "Ancho b(y) [cm]",
                                        "<b>Primer momento de área Q(y) [cm³]</b>"))
    _dibujar_seccion(fig, sec)
    fig.add_trace(go.Scatter(x=bb, y=ys, mode="lines", line=dict(color=COL["gris"], width=2.5),
                             fill="tozerox", fillcolor=_rgba(COL["gris"], 0.25),
                             hovertemplate="b = %{x:.3f} cm<br>y = %{y:.3f} cm<extra></extra>"),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=Q, y=ys, mode="lines", line=dict(color=COL["steel"], width=3),
                             fill="tozerox", fillcolor=_rgba(COL["steel"], 0.25),
                             hovertemplate="Q = %{x:.3f} cm³<br>y = %{y:.3f} cm<extra></extra>"),
                  row=1, col=3)
    px, py, pt = [], [], []
    for y in sec["niveles"]:
        q = sec["Q_en"](min(max(y, 1e-6), sec["H"] - 1e-6))[0]
        px.append(q)
        py.append(y)
        pt.append(f"{q:.1f}" if q > 1e-3 * max(abs(Q.max()), 1) else "")
    fig.add_trace(go.Scatter(x=px, y=py, mode="markers+text", text=pt, textposition="top right",
                             marker=dict(color=COL["steel"], size=8, line=dict(color="white", width=1)),
                             textfont=dict(color=COL["navy"], size=12), cliponaxis=False,
                             hovertemplate="Q = %{x:.3f} cm³<br>y = %{y:.3f} cm<extra>nivel</extra>"),
                  row=1, col=3)
    fig.add_hline(y=sec["yc"], line=dict(color=COL["rojo"], width=1.2, dash="dash"),
                  row="all", col="all")
    for y in sec["niveles"]:
        fig.add_hline(y=y, line=dict(color="#B8C4D2", width=0.8, dash="dot"), row="all", col="all")
    fig.update_yaxes(range=[0, sec["H"]], title_text="y desde la base [cm]", row=1, col=1)
    fig.update_xaxes(range=[0, max(abs(Q.max()), 1) * 1.18], title_text="Q(y) [cm³]", row=1, col=3)
    fig.update_xaxes(title_text="b(y) [cm]", row=1, col=2)
    fig.update_xaxes(showgrid=True, gridcolor="#EEF1F5")
    fig.update_yaxes(showgrid=True, gridcolor="#EEF1F5")
    return _layout(fig, 560)


def fig_esfuerzos(sec, res):
    ys = _grid_y(sec)
    I_cm = sec["I"] * 1e8
    Q = np.array([sec["Q_en"](y)[0] for y in ys])
    bb = np.array([sec["b_en"](y) for y in ys])
    sig = -res["Mmax"] * 100 * (ys - sec["yc"]) / I_cm * 10      # MPa
    tau = abs(res["Vmax"]) * Q / (I_cm * bb) * 10                # MPa
    fig = make_subplots(
        rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.05,
        subplot_titles=("Sección (cm)",
                        f"σ(y) = −M·(y−ȳ)/I<br><sup>M = {res['Mmax']:.2f} kN·m  [MPa]</sup>",
                        f"τ(y) = V·Q(y)/(I·b(y))<br><sup>|V| = {abs(res['Vmax']):.2f} kN  [MPa]</sup>"))
    _dibujar_seccion(fig, sec)
    for c, v, col, nm in ((2, sig, COL["navy"], "σ"), (3, tau, COL["steel"], "τ")):
        fig.add_trace(go.Scatter(x=v, y=ys, mode="lines", line=dict(color=col, width=3),
                                 fill="tozerox", fillcolor=_rgba(col, 0.22),
                                 hovertemplate=f"{nm} = %{{x:.4f}} MPa<br>y = %{{y:.3f}} cm<extra></extra>"),
                      row=1, col=c)
        i = int(np.argmax(np.abs(v)))
        fig.add_trace(go.Scatter(x=[v[i]], y=[ys[i]], mode="markers+text", text=[f"{v[i]:.3f}"],
                                 textposition="top right", cliponaxis=False,
                                 marker=dict(color=col, size=9, line=dict(color="white", width=1)),
                                 textfont=dict(color=col, size=13),
                                 hovertemplate=f"{nm}máx = %{{x:.4f}} MPa<br>y = %{{y:.3f}} cm<extra></extra>"),
                      row=1, col=c)
        fig.add_vline(x=0, line=dict(color=COL["navy"], width=0.8), row=1, col=c)
        fig.add_hline(y=sec["yc"], line=dict(color=COL["rojo"], width=0.9, dash="dash"), row=1, col=c)
        fig.update_xaxes(title_text=f"{nm} [MPa]", row=1, col=c)
    fig.add_hline(y=sec["yc"], line=dict(color=COL["rojo"], width=1.2, dash="dash"), row=1, col=1)
    fig.update_yaxes(range=[0, sec["H"]], title_text="y desde la base [cm]", row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#EEF1F5")
    fig.update_yaxes(showgrid=True, gridcolor="#EEF1F5")
    _layout(fig, 560)
    return fig, np.max(np.abs(sig)), np.max(tau)


def esfuerzos(sec, res):
    """Bloque 'ESFUERZOS EN LA SECCIÓN' de tu main(): mismos cálculos y mismos print()."""
    sub("ESFUERZOS EN LA SECCIÓN (con |M|máx y |V|máx)")
    print("  Flexión: σ(y) = −M·(y−ȳ)/I     Cortante: τ(y) = V·Q(y)/(I·b(y))")
    sig = abs(res["Mmax"]) * sec["c"] / sec["I"] / 1000
    tau = abs(res["Vmax"]) * sec["Q"] / (sec["I"] * sec["bna"]) / 1000
    print(f"  σ_máx = |M|·c / I = {abs(res['Mmax']):.4f}·{sec['c']:.4f} / {sec['I']:.4e} = {sig:.4f} MPa")
    print(f"  τ en el eje neutro = |V|·Q_EN / (I·b) = {abs(res['Vmax']):.4f}·{sec['Q']:.4e} / "
          f"({sec['I']:.4e}·{sec['bna']:.4f}) = {tau:.4f} MPa")
    fig, smax, tmax = fig_esfuerzos(sec, res)
    print(f"  Máximos del diagrama: |σ| = {smax:.4f} MPa , τ = {tmax:.4f} MPa")
    return dict(fig=fig, sig=sig, tau=tau, smax=smax, tmax=tmax)


# ════════════════════════════════════════════════════════════════════
# 2) DATOS DE LA VIGA
# ════════════════════════════════════════════════════════════════════
def datos_viga(sec, v):
    """Igual que tu datos_viga(), pero los valores vienen del formulario (dict v)."""
    titulo("DATOS DE LA VIGA")
    L = v["L"]
    E = v["E"]
    EI = E * 1e6 * sec["I"]
    print(f"  EI = E·I = {E:g}·10⁶ kN/m² · {sec['I']:.6e} m⁴ = {EI:.5g} kN·m²")
    izq, der = v["izq"], v["der"]
    print(f"  → Izquierdo: {NOM[izq]} | Derecho: {NOM[der]}")
    M0 = v["M0"]
    ML = v["ML"]
    w_i = v["w_i"]
    w_f = v["w_f"]
    P = v["P"]
    a = 0.0
    if P != 0:
        a = v["a"]
    w_pp = 0.0
    if v["pp"]:
        g = v["gamma"]
        w_pp = g * sec["A"]
        print(f"  w_pp = γ·A = {g:g}·{sec['A']:.6f} = {w_pp:.5g} kN/m")
    beta = 0.0
    if v["cort"]:
        nu = v["nu"]
        kap = v["kap"]
        G = E / (2 * (1 + nu))
        As = kap * sec["A"]
        beta = EI / (G * 1e6 * As)
        print(f"  G = E/(2(1+ν)) = {G:.4f} GPa ; As = κ·A = {As:.6f} m²")
        print(f"  β = EI/(G·As) = {beta:.6g} m²  →  Φ = 12β/L² = {12*beta/L**2:.6g} (para toda la viga)")
    return dict(L=L, E=E, EI=EI, izq=izq, der=der, M0=M0, ML=ML,
                w1=w_i + w_pp, w2=w_f + w_pp, P=P, a=a, w_pp=w_pp, beta=beta)


# ════════════════════════════════════════════════════════════════════
# 3) DOBLE INTEGRACIÓN  (idéntica a tu código)
# ════════════════════════════════════════════════════════════════════
def doble_integracion(d):
    titulo("MÉTODO DE DOBLE INTEGRACIÓN — PASO A PASO")
    Rt = lambda v: sp.Rational(f"{float(v):.12g}")
    L, M0, ML = Rt(d["L"]), Rt(d["M0"]), Rt(d["ML"])
    w1, w2, P, a = Rt(d["w1"]), Rt(d["w2"]), Rt(d["P"]), Rt(d["a"])
    k = (w2 - w1) / L
    EI, beta = d["EI"], d["beta"]
    RA, MA, RB, MB, C1, C2 = sp.symbols("R_A M_A R_B M_B C_1 C_2")
    izq, der = d["izq"], d["der"]
    Z = sp.Integer(0)
    RA_, MA_, RB_, MB_ = Z, Z, Z, Z
    unk = []
    if izq in "EA":
        RA_ = RA
        unk.append(RA)
    if izq == "E":
        MA_ = MA
        unk.append(MA)
    if der in "EA":
        RB_ = RB
        unk.append(RB)
    if der == "E":
        MB_ = MB
        unk.append(MB)
    unk += [C1, C2]

    sub("PASO 1 · Cargas y ecuaciones de EQUILIBRIO GLOBAL")
    print("  Origen: la viga completa está en equilibrio ⇒ ΣFy = 0 y ΣM = 0.")
    print(f"  w(x) = w₁ + k·x  con w₁ = {float(w1):.5g} kN/m , k = (w₂ − w₁)/L = "
          f"({float(w2):.5g} − {float(w1):.5g})/{float(L):g} = {float(k):.5g} kN/m²")
    Wt = w1 * L + k * L ** 2 / 2
    Mw = w1 * L ** 2 / 2 + k * L ** 3 / 3
    print(f"  Resultante: W = ∫₀ᴸ w dx = w₁L + kL²/2 = {float(Wt):.5g} kN")
    print(f"  Cada franja dW = w dx está a la distancia x del origen ⇒ Mw = ∫₀ᴸ w·x dx = w₁L²/2 + kL³/3"
          f" = {float(Mw):.5g} kN·m")
    if P != 0:
        print(f"  Carga puntual P = {float(P):g} kN en x = a = {float(a):g} m")
    print(f"  Momentos aplicados: M₀ = {float(M0):g} (izq) , M_L = {float(ML):g} kN·m (der)")
    print("  ΣFy = 0 :  R_A + R_B − W − P = 0")
    print("  ΣM₍ₓ₌₀₎ = 0 (↺ +):  M₀ + M_A + M_L + M_B + R_B·L − P·a − Mw = 0   (brazo de R_B = L, de P = a)")
    eqs = [("ΣFy = 0", RA_ + RB_ - Wt - P),
           ("ΣM(x=0) = 0", M0 + MA_ + ML + MB_ + RB_ * L - P * a - Mw)]
    print("  Sustituyendo (solo quedan las reacciones que existen según los apoyos):")
    for i, (nm, e) in enumerate(eqs, 1):
        print(f"    ({i}) {nm:<14} → {simb(sp.N(sp.expand(e), 6))} = 0")

    sub("PASO 2 · CORTANTE V(x)")
    print("  Origen: equilibrio vertical de un diferencial dx de viga ⇒ dV/dx = −w(x).")
    print("  Integrando desde x=0:  V(x) = R_A − ∫₀ˣ w(ξ)dξ − P·⟨x−a⟩⁰")
    print("                              = R_A − w₁·x − k·x²/2 − P·⟨x−a⟩⁰")
    print("  ⟨x−a⟩⁰ vale 0 si x<a y 1 si x≥a (la carga puntual produce un SALTO en V).")
    Mt = [(RA_, 0, 1), (-(M0 + MA_), 0, 0)]
    if w1 != 0:
        Mt.append((-w1 / 2, 0, 2))
    if k != 0:
        Mt.append((-k / 6, 0, 3))
    if P != 0:
        Mt.append((-P, a, 1))
    Vt = [(c * n, aa, n - 1) for c, aa, n in Mt if n >= 1]
    print(f"  V(x) = {fmt(Vt)}")

    sub("PASO 3 · MOMENTO FLECTOR M(x)")
    print("  Origen: equilibrio de momentos del diferencial dx ⇒ dM/dx = V(x) → M(x) = ∫V dx + C")
    print("  Se integra término a término con ∫⟨x−a⟩ⁿ dx = ⟨x−a⟩ⁿ⁺¹/(n+1):")
    mostrar_int(Vt)
    print("  Constante C: cuerpo libre de un tramo infinitesimal en x=0⁺:  M(0⁺) + M₀ + M_A = 0")
    print("  ⇒ C = M(0⁺) = −(M₀ + M_A)")
    print(f"  M(x) = {fmt(Mt)}")

    sub("PASO 4 · ECUACIÓN DE LA ELÁSTICA  EI·y″ = M(x)")
    print("  Origen (flexión pura): una fibra a distancia y del eje neutro se deforma ε = −κ·y")
    print("  (κ = curvatura ≈ y″). Ley de Hooke σ = E·ε. El momento resultante M = −∫σ·y dA = E·κ·∫y² dA = E·I·κ")
    print("  ⇒ κ = y″ = M/(EI)  ⇒  EI·y″ = M(x)")
    print(f"  EI·y″ = {fmt(Mt)}")

    th_t = integ(Mt) + [(C1, 0, 0)]
    sub("PASO 5 · 1ª INTEGRACIÓN → PENDIENTE  EI·θ = ∫M dx + C₁")
    print("  θ(x) = giro de la sección ≈ y′(x):")
    mostrar_int(Mt)
    print(f"  EI·θ(x) = {fmt(th_t)}")

    if beta != 0:
        y_t = integ(th_t) + [(-Rt(beta) * c, aa, n) for c, aa, n in integ(Vt)] + [(C2, 0, 0)]
    else:
        y_t = integ(th_t) + [(C2, 0, 0)]
    sub("PASO 6 · 2ª INTEGRACIÓN → FLECHA  EI·y = ∫EI·θ dx + C₁x + C₂")
    mostrar_int(th_t[:-1])
    print("  La constante C₁ integrada da C₁·x ; se agrega C₂.")
    if beta != 0:
        print(f"  Con cortante (Timoshenko): y′ = θ − V/(G·As) ⇒ EI·y = ∫EI·θ dx − β·∫V dx + C₂ ,"
              f" β = EI/(G·As) = {beta:.6g} m²")
        print("  término de cortante −β·∫V dx :")
        mostrar_int(Vt)
    print(f"  EI·y(x) = {fmt(y_t)}")

    sub("PASO 7 · CONDICIONES DE FRONTERA")
    print("  Origen: un apoyo impide el desplazamiento vertical (y=0); un empotre impide además el giro (θ=0).")
    bcs = []
    if izq == "E":
        bcs += [("x=0: y=0 (empotre)", ev_sym(y_t, 0)), ("x=0: θ=0 (empotre)", ev_sym(th_t, 0))]
    elif izq == "A":
        bcs += [("x=0: y=0 (apoyo)", ev_sym(y_t, 0))]
    if der == "E":
        bcs += [("x=L: y=0 (empotre)", ev_sym(y_t, L)), ("x=L: θ=0 (empotre)", ev_sym(th_t, L))]
    elif der == "A":
        bcs += [("x=L: y=0 (apoyo)", ev_sym(y_t, L))]
    print("  Se sustituye x en EI·y(x) o EI·θ(x) y se iguala a cero:")
    for nm, e in bcs:
        print(f"    {nm:<22} → {simb(sp.N(sp.expand(e), 6))} = 0")
    alleq = eqs + bcs
    nombres_u = [str(u) for u in unk]
    print(f"\n  Incógnitas: {', '.join(nombres_u)} ({len(unk)}) | Ecuaciones: {len(alleq)}"
          " (2 de equilibrio + condiciones de frontera)")

    sub("PASO 8 · SISTEMA LINEAL A·u = b y solución (Gauss)")
    A, b = sp.linear_eq_to_matrix([e for _, e in alleq], unk)
    An = np.array(A.tolist(), float)
    bn = np.array(b.tolist(), float).ravel()
    print(f"  Todas las ecuaciones son lineales en u = [{', '.join(nombres_u)}]ᵀ")
    sol = gauss_pasos(An, bn, nombres_u)
    S = {u: sp.Float(v, 15) for u, v in zip(unk, sol)}
    print("\n  RESULTADOS:")
    for u, v in zip(unk, sol):
        unidad = {"R": "kN", "M": "kN·m"}.get(str(u)[0], "")
        print(f"    {str(u):<5} = {v:.6g} {unidad}")
    rea = {nm: 0.0 for nm in ("RA", "MA", "RB", "MB")}
    for u, v in zip(unk, sol):
        key = str(u).replace("_", "")
        if key in rea:
            rea[key] = float(v)

    fin = lambda t: [(c.subs(S), aa, n) for c, aa, n in t]
    Vf, Mf, thf, yf = fin(Vt), fin(Mt), fin(th_t), fin(y_t)
    sub("PASO 9 · ECUACIONES FINALES")
    print(f"  V(x)    = {fmt(Vf)}   [kN]")
    print(f"  M(x)    = {fmt(Mf)}   [kN·m]")
    print(f"  EI·θ(x) = {fmt(thf)}")
    print(f"  EI·y(x) = {fmt(yf)}")
    print(f"  con EI = {EI:.6g} kN·m²  →  θ = (EI·θ)/EI [rad] , y = (EI·y)/EI [m]")

    res = dict(V=num_terms(Vf), M=num_terms(Mf), th=num_terms(thf), y=num_terms(yf),
               EI=EI, rea=rea)
    Lf = float(L)
    xs = np.linspace(0, Lf, 4001)
    V, M = ev_num(res["V"], xs), ev_num(res["M"], xs)
    y = ev_num(res["y"], xs) / EI
    puntos = sorted(set([0, Lf / 4, Lf / 2, 3 * Lf / 4, Lf] + ([float(a)] if P != 0 else [])))
    sub("PASO 10 · TABLA DE VALORES")
    print(f"  {'x [m]':>8} {'V [kN]':>12} {'M [kN·m]':>12} {'θ [rad]':>14} {'y [mm]':>12}")
    for xv in puntos:
        xa = np.array([min(max(xv, 1e-9 * Lf), Lf - 1e-9 * Lf)])
        print(f"  {xv:8.3f} {ev_num(res['V'], xa)[0]:12.4f} {ev_num(res['M'], xa)[0]:12.4f} "
              f"{ev_num(res['th'], xa)[0]/EI:14.6e} {ev_num(res['y'], xa)[0]/EI*1000:12.4f}")
    iM, iy, iV = np.argmax(np.abs(M)), np.argmax(np.abs(y)), np.argmax(np.abs(V))
    res.update(Vmax=V[iV], Mmax=M[iM], xM=xs[iM], ymax=y[iy], xy=xs[iy])
    print(f"\n  |V|máx = {V[iV]:.4f} kN     (x={xs[iV]:.3f} m)")
    print(f"  |M|máx = {M[iM]:.4f} kN·m   (x={xs[iM]:.3f} m)")
    print(f"  |y|máx = {y[iy]*1000:.4f} mm    (x={xs[iy]:.3f} m)")
    return res


# ════════════════════════════════════════════════════════════════════
# 4) DIAGRAMAS DE LA VIGA (Plotly en lugar de matplotlib)
# ════════════════════════════════════════════════════════════════════
def esquema(d):
    L = d["L"]
    fig = go.Figure()
    fig.update_xaxes(range=[-0.1 * L, 1.1 * L], visible=False)
    fig.update_yaxes(range=[-2.4 if d["P"] < 0 else -1.4, 2.7], visible=False)
    fig.add_trace(go.Scatter(x=[0, L], y=[0, 0], mode="lines",
                             line=dict(color=COL["navy"], width=9),
                             hovertemplate=f"Viga · L = {L:g} m<extra></extra>"))
    for x, t, lado in ((0, d["izq"], "i"), (L, d["der"], "d")):
        if t == "A":
            fig.add_trace(go.Scatter(
                x=[x - 0.025 * L, x + 0.025 * L, x, x - 0.025 * L], y=[-0.5, -0.5, -0.05, -0.5],
                mode="lines", fill="toself", fillcolor="#9AA8B8",
                line=dict(color=COL["navy"], width=1.5), hoveron="fills", hoverinfo="text",
                text="Apoyo simple"))
        elif t == "E":
            x0 = x - 0.03 * L if lado == "i" else x
            fig.add_trace(go.Scatter(
                x=[x0, x0 + 0.03 * L, x0 + 0.03 * L, x0, x0], y=[-0.8, -0.8, 0.8, 0.8, -0.8],
                mode="lines", fill="toself",
                fillpattern=dict(shape="/", bgcolor=COL["claro"], fgcolor=COL["navy"]),
                line=dict(color=COL["navy"], width=1.5), hoveron="fills", hoverinfo="text",
                text="Empotramiento"))
    w = lambda x: d["w1"] + (d["w2"] - d["w1"]) * x / L
    wm = max(abs(d["w1"]), abs(d["w2"]))
    if wm > 0:
        xs = np.linspace(0, L, 11)
        alt = [np.sign(w(x)) * (0.3 + 0.8 * abs(w(x)) / wm) for x in xs]
        for x, h in zip(xs, alt):
            if h != 0:
                fig.add_annotation(x=x, y=0.1 * np.sign(h), ax=x, ay=h, xref="x", yref="y",
                                   axref="x", ayref="y", showarrow=True, text="", arrowhead=2,
                                   arrowsize=1, arrowwidth=1.4, arrowcolor=COL["steel"])
        fig.add_trace(go.Scatter(x=xs, y=alt, mode="lines", line=dict(color=COL["steel"], width=1.5),
                                 hoverinfo="skip"))
        fig.add_annotation(x=L / 2, y=max(alt + [0]) + 0.15, showarrow=False, yanchor="bottom",
                           text=f"w = {d['w1']:.4g} → {d['w2']:.4g} kN/m"
                                + (" (incl. peso propio)" if d["w_pp"] else ""),
                           font=dict(color=COL["steel"]))
    if d["P"] != 0:
        s = np.sign(d["P"])
        fig.add_annotation(x=d["a"], y=0.1 * s, ax=d["a"], ay=1.9 * s, xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True, text="", arrowhead=2,
                           arrowsize=1.2, arrowwidth=3, arrowcolor=COL["cobre"])
        fig.add_annotation(x=d["a"], y=2.0 * s + (0.05 if s > 0 else -0.25), showarrow=False,
                           text=f"P = {d['P']:g} kN", yanchor="bottom" if s > 0 else "top",
                           font=dict(color=COL["cobre"], size=14))
    for x, m, ha in ((0, d["M0"], "left"), (L, d["ML"], "right")):
        if m != 0:
            fig.add_annotation(x=x, y=-0.9, showarrow=False, xanchor=ha, yanchor="top",
                               text=f"<b>M = {abs(m):g} kN·m {'↺' if m > 0 else '↻'}</b>",
                               font=dict(color="#2A7F62", size=14))
    fig.update_layout(title=dict(text="Viga y cargas  (+: cargas ↓, momentos ↺)", x=0.01, font=dict(size=15)))
    return _layout(fig, 330)


def graficar(d, res):
    L = d["L"]
    eps = 1e-9 * L
    pts = set(np.linspace(0, L, 1601))
    if d["P"] != 0:
        pts |= {max(d["a"] - eps, 0), min(d["a"] + eps, L)}
    xs = np.array(sorted(pts))
    V, M = ev_num(res["V"], xs), ev_num(res["M"], xs)
    y = ev_num(res["y"], xs) / res["EI"] * 1000
    xv = np.concatenate([[0], xs, [L]])
    Vv = np.concatenate([[0], V, [0]])
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Diagrama de CORTANTE V(x)",
                                        "Diagrama de MOMENTO FLECTOR M(x)  (+ = tracción abajo)",
                                        "Elástica: FLECHA y(x)  (+ = hacia arriba)"))
    for r, (xx, yy, col, nom, uni) in enumerate((
            (xv, Vv, COL["steel"], "V", "kN"),
            (xs, M, COL["navy"], "M", "kN·m"),
            (xs, y, COL["cobre"], "y", "mm")), 1):
        fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name=nom, line=dict(color=col, width=2.6),
                                 fill="tozeroy", fillcolor=_rgba(col, 0.18),
                                 hovertemplate=f"<b>{nom}</b> = %{{y:.5g}} {uni}<extra></extra>"),
                      row=r, col=1)
        i = int(np.argmax(np.abs(yy)))
        fig.add_trace(go.Scatter(x=[xx[i]], y=[yy[i]], mode="markers", hoverinfo="skip",
                                 marker=dict(color=col, size=9, line=dict(color="white", width=1.5))),
                      row=r, col=1)
        fig.add_annotation(x=xx[i], y=yy[i], text=f"<b>{yy[i]:.4g} {uni}</b>", showarrow=True,
                           arrowcolor=col, arrowwidth=1, ax=34, ay=-30 if yy[i] >= 0 else 30,
                           font=dict(color=col, size=13), bgcolor="white", bordercolor=col,
                           borderwidth=1, borderpad=3, row=r, col=1)
        fig.add_hline(y=0, line=dict(color=COL["navy"], width=1), row=r, col=1)
        fig.update_yaxes(title_text=uni, row=r, col=1)
    if d["P"] != 0:
        fig.add_vline(x=d["a"], line=dict(color=COL["cobre"], width=1, dash="dot"), row="all", col="all")
    fig.update_xaxes(range=[-0.1 * L, 1.1 * L], hoverformat=".3f", showspikes=True,
                     spikemode="across", spikesnap="cursor", spikethickness=1,
                     spikecolor=COL["gris"], spikedash="dot")
    fig.update_xaxes(title_text="x [m]", row=3, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#EEF1F5")
    fig.update_yaxes(showgrid=True, gridcolor="#EEF1F5", zeroline=False)
    fig.update_layout(hovermode="x unified")
    return _layout(fig, 860)


# ════════════════════════════════════════════════════════════════════
# 5) MÉTODO DE RIGIDEZ MATRICIAL  (idéntico a tu código)
# ════════════════════════════════════════════════════════════════════
def deducir_ke():
    """Deduce k = ∫ EI·N″ᵢ·N″ⱼ dx con las funciones de forma de Hermite (simbólico)."""
    x, L, EI = sp.symbols("x L EI", positive=True)
    xi = x / L
    N = [1 - 3 * xi ** 2 + 2 * xi ** 3, L * (xi - 2 * xi ** 2 + xi ** 3),
         3 * xi ** 2 - 2 * xi ** 3, L * (-xi ** 2 + xi ** 3)]
    nom = ["N₁", "N₂", "N₃", "N₄"]
    print("  Se aproxima la elástica del elemento con un polinomio cúbico (solución exacta de EI·y⁗=0):")
    print("     y(x) = N₁·v_i + N₂·θ_i + N₃·v_j + N₄·θ_j ,   ξ = x/L")
    print("  Las N valen 1 en su GDL y 0 en los otros tres (condiciones nodales):")
    for n_, f in zip(nom, N):
        print(f"     {n_}(x) = {simb(sp.expand(f))}")
    Npp = [sp.simplify(sp.diff(f, x, 2)) for f in N]
    print("  Curvatura y″ = Σ Nᵢ″·dᵢ :")
    for n_, f in zip(nom, Npp):
        print(f"     {n_}″(x) = {simb(sp.factor(f))}")
    print("  Energía de deformación U = ½∫EI·(y″)² dx = ½ dᵀ·k·d  ⇒  kᵢⱼ = ∫₀ᴸ EI·Nᵢ″·Nⱼ″ dx")
    k11 = sp.integrate(EI * Npp[0] ** 2, (x, 0, L))
    print(f"  Ejemplo k₁₁ = ∫₀ᴸ EI·({simb(sp.factor(Npp[0]))})² dx = {simb(sp.simplify(k11))}")
    K = sp.Matrix(4, 4, lambda i, j: sp.simplify(sp.integrate(EI * Npp[i] * Npp[j], (x, 0, L)) * L ** 3 / EI))
    print("  Resolviendo las 16 integrales:  k = (EI/L³)·")
    caja([[simb(K[i, j]) for j in range(4)] for i in range(4)])


def empotramiento(wa, wb, Le, beta):
    """Fuerzas de empotramiento perfecto con la misma doble integración (con cortante si β≠0)."""
    k = (wb - wa) / Le
    A = np.array([[Le ** 2 / 2, -Le], [Le ** 3 / 6 - beta * Le, -Le ** 2 / 2]])
    b = np.array([wa * Le ** 3 / 6 + k * Le ** 4 / 24,
                  wa * Le ** 4 / 24 + k * Le ** 5 / 120 - beta * (wa * Le ** 2 / 2 + k * Le ** 3 / 6)])
    V1, M1 = np.linalg.solve(A, b)
    W = wa * Le + k * Le ** 2 / 2
    Mw = wa * Le ** 2 / 2 + k * Le ** 3 / 3
    V2 = W - V1
    M2 = Mw - M1 - V2 * Le
    return A, b, np.array([V1, M1, V2, M2])


def rigidez(d, res):
    titulo("MÉTODO DE RIGIDEZ MATRICIAL — PASO A PASO")
    L, EI, beta = d["L"], res["EI"], d["beta"]
    w1, w2, P, a = d["w1"], d["w2"], d["P"], d["a"]
    k = (w2 - w1) / L
    nodos = {0.0, float(L)}
    if P != 0 and 0 < a < L:
        nodos.add(float(a))
    nodos = sorted(nodos)
    if len(nodos) == 2:
        nodos.insert(1, L / 2)
    nn, ne = len(nodos), len(nodos) - 1
    nd = 2 * nn
    dof = lambda i: f"{'v' if i % 2 == 0 else 'θ'}{i // 2 + 1}"

    sub("PASO 1 · DISCRETIZACIÓN (2 GDL por nodo: v = desplazamiento ↑, θ = giro ↺)")
    print("  Se colocan nodos en los extremos, en la carga puntual (para aplicarla como fuerza nodal)")
    print("  y, si hace falta, a mitad de claro. Fuerzas nodales: [V_i, M_i, V_j, M_j]; desplazamientos [v_i, θ_i, v_j, θ_j].")
    for i, x in enumerate(nodos, 1):
        print(f"    Nodo {i}: x = {x:.4f} m → GDL: v{i} (#{2*i-1}), θ{i} (#{2*i})")
    for e in range(ne):
        print(f"    Elemento {e+1}: nodos {e+1}–{e+2}, Le = {nodos[e+1]-nodos[e]:.4f} m")

    sub("PASO 2 · DEDUCCIÓN DE LA MATRIZ DE RIGIDEZ DEL ELEMENTO")
    deducir_ke()
    print("\n  Con deformación por cortante (viga de Timoshenko) se obtiene la forma general:")
    print("  k = [ EI / ((1+Φ)·L³) ] ·")
    caja([["12", "6L", "−12", "6L"],
          ["6L", "(4+Φ)L²", "−6L", "(2−Φ)L²"],
          ["−12", "−6L", "12", "−6L"],
          ["6L", "(2−Φ)L²", "−6L", "(4+Φ)L²"]])
    print("  con Φ = 12·EI/(G·As·L²) = 12β/L²   (β = EI/(G·As))")
    print("  Origen de Φ: la flecha de un empotrado-guiado bajo V es V·L³/(12EI) + V·L/(G·As) = [V·L³/(12EI)]·(1+Φ),")
    print("  por eso la rigidez 12EI/L³ se divide entre (1+Φ). Si Φ = 0 (sin cortante) queda la matriz de arriba.")
    print(f"  Aquí β = {beta:.6g} m² → " +
          ("se usa Φₑ = 12β/Lₑ² en cada elemento." if beta else "Φ = 0 (Euler-Bernoulli)."))

    sub("PASO 3 · MATRIZ NUMÉRICA DE CADA ELEMENTO")
    K = np.zeros((nd, nd))
    ke_list = []
    for e in range(ne):
        Le = nodos[e + 1] - nodos[e]
        Ph = 12 * beta / Le ** 2
        f = EI / ((1 + Ph) * Le ** 3)
        base = np.array([[12, 6 * Le, -12, 6 * Le],
                         [6 * Le, (4 + Ph) * Le ** 2, -6 * Le, (2 - Ph) * Le ** 2],
                         [-12, -6 * Le, 12, -6 * Le],
                         [6 * Le, (2 - Ph) * Le ** 2, -6 * Le, (4 + Ph) * Le ** 2]])
        ke = f * base
        ke_list.append(ke)
        print(f"\n  Elemento {e+1}: Le = {Le:.4f} m , Φ = 12·{beta:.5g}/{Le:.4f}² = {Ph:.6g}")
        print(f"    EI/((1+Φ)Le³) = {EI:.5g}/((1+{Ph:.5g})·{Le:.4f}³) = {f:.6g}")
        print("    matriz base (coeficientes con L y Φ sustituidos):")
        pm(base)
        print("    ke = [EI/((1+Φ)Le³)] · base =")
        pm(ke)

    sub("PASO 4 · FUERZAS DE EMPOTRAMIENTO PERFECTO (carga distribuida)")
    print("  Origen: se bloquean los nodos del elemento (v=θ=0 en ambos extremos) y se resuelve con la")
    print("  misma doble integración (x local 0…Le):")
    print("    M(x) = V₁x − M₁ − wa·x²/2 − k·x³/6 ,  V(x) = V₁ − wa·x − k·x²/2 ,  k = (wb−wa)/Le")
    print("    EI·θ = ∫M dx , EI·y = ∫EI·θ dx − β∫V dx   (θ(0)=y(0)=0 ⇒ C₁=C₂=0 ; β=0 sin cortante)")
    print("    θ(Le)=0 : (Le²/2)·V₁ − Le·M₁ = wa·Le³/6 + k·Le⁴/24")
    print("    y(Le)=0 : (Le³/6 − β·Le)·V₁ − (Le²/2)·M₁ = wa·Le⁴/24 + k·Le⁵/120 − β(wa·Le²/2 + k·Le³/6)")
    print("    luego V₂ = W − V₁ , M₂ = Mw − M₁ − V₂·Le   (equilibrio del elemento)")
    FEF = []
    Feq = np.zeros(nd)
    for e in range(ne):
        xi, xj = nodos[e], nodos[e + 1]
        Le = xj - xi
        wa, wb = w1 + k * xi, w1 + k * xj
        A2, b2, fef = empotramiento(wa, wb, Le, beta)
        FEF.append(fef)
        print(f"\n  Elemento {e+1}: wa = {wa:.5g}, wb = {wb:.5g} kN/m, Le = {Le:.4f} m")
        print(f"    [{A2[0,0]:10.5g} {A2[0,1]:10.5g}] [V₁]   [{b2[0]:10.5g}]")
        print(f"    [{A2[1,0]:10.5g} {A2[1,1]:10.5g}] [M₁] = [{b2[1]:10.5g}]")
        print(f"    ⇒ V₁ = {fef[0]:.5g} , M₁ = {fef[1]:.5g} , V₂ = {fef[2]:.5g} , M₂ = {fef[3]:.5g}")
        Feq[2 * e:2 * e + 4] += -fef
    print("\n  Las cargas nodales EQUIVALENTES son −FEF (se invierte el sentido para que, sumadas a las")
    print("  fuerzas del elemento bloqueado, reproduzcan la carga real). Ensambladas:")
    pm(Feq.reshape(-1, 1))

    sub("PASO 5 · ENSAMBLE DE LA MATRIZ GLOBAL K")
    print("  Compatibilidad: los GDL del nodo compartido son los mismos para ambos elementos;")
    print("  equilibrio: las fuerzas internas de los elementos en cada nodo se suman ⇒ K = Σ ke (en sus GDL).")
    for e, ke in enumerate(ke_list):
        K[2 * e:2 * e + 4, 2 * e:2 * e + 4] += ke
        print(f"    Elemento {e+1} se suma en los GDL {[dof(i) for i in range(2*e, 2*e+4)]}")
    print(f"\n  K global ({nd}×{nd}) =")
    pm(K, 11)

    sub("PASO 6 · VECTOR DE CARGAS NODALES  F = F_equivalentes + F_nodales")
    Fn = np.zeros(nd)
    if P != 0:
        j = int(np.argmin([abs(x - a) for x in nodos]))
        Fn[2 * j] += -P
        print(f"  P = {P:g} kN ↓ en el nodo {j+1} → F[{dof(2*j)}] += {-P:g}  (positivo = ↑)")
    Fn[1] += d["M0"]
    Fn[-1] += d["ML"]
    if d["M0"] or d["ML"]:
        print(f"  Momentos aplicados (↺ +): F[θ1] += {d['M0']:g} , F[θ{nn}] += {d['ML']:g}")
    F = Feq + Fn
    print("  F =")
    pm(F.reshape(-1, 1))

    sub("PASO 7 · CONDICIONES DE APOYO")
    rest = []
    if d["izq"] in "EA":
        rest.append(0)
    if d["izq"] == "E":
        rest.append(1)
    if d["der"] in "EA":
        rest.append(nd - 2)
    if d["der"] == "E":
        rest.append(nd - 1)
    libres = [i for i in range(nd) if i not in rest]
    print("  Ecuación de equilibrio de toda la estructura: K·d = F + R   (R = reacciones, solo en GDL restringidos)")
    print(f"  GDL restringidos (=0): {[dof(i) for i in rest]}")
    print(f"  GDL libres           : {[dof(i) for i in libres]}")
    dd = np.zeros(nd)
    if libres:
        Kff = K[np.ix_(libres, libres)]
        Ff = F[libres]
        sub("PASO 8 · SISTEMA REDUCIDO  Kff·d_f = F_f  (se eliminan filas y columnas restringidas)")
        print("  En los GDL libres R = 0, y los desplazamientos restringidos son 0, así que se queda solo:")
        print("  Kff =")
        pm(Kff, 11)
        print("  F_f =")
        pm(Ff.reshape(-1, 1))
        sub("PASO 9 · RESOLUCIÓN POR ELIMINACIÓN DE GAUSS")
        dd[libres] = gauss_pasos(Kff, Ff, [dof(i) for i in libres])
    else:
        print("  Todos los GDL están restringidos → d = 0")
    print("\n  Vector de desplazamientos d:")
    for i in range(nd):
        u = "m" if i % 2 == 0 else "rad"
        extra = f"  ({dd[i]*1000:.5g} mm)" if i % 2 == 0 else ""
        print(f"    {dof(i):<4} = {dd[i]:.6e} {u}{extra}")

    sub("PASO 10 · REACCIONES  R = K·d − F  (filas de los GDL restringidos)")
    rea = {}
    for r in rest:
        Kd = float(K[r] @ dd)
        R = Kd - F[r]
        nm = {0: "RA", 1: "MA", nd - 2: "RB", nd - 1: "MB"}[r]
        rea[nm] = R
        unidad = "kN" if r % 2 == 0 else "kN·m"
        print(f"  {nm} ({dof(r)}): K[{dof(r)},:]·d = {Kd:.6g} ; F[{dof(r)}] = {F[r]:.6g}"
              f"  →  R = {Kd:.6g} − ({F[r]:.6g}) = {R:.6g} {unidad}")

    sub("PASO 11 · FUERZAS EN LOS EXTREMOS DE CADA ELEMENTO  f = ke·d_e + FEF")
    print("  Se recupera la fuerza real en el elemento: la que produce el desplazamiento + la del elemento bloqueado.")
    print("  Convención de viga: V_ini = f₁ , M_ini = −f₂ , V_fin = −f₃ , M_fin = f₄")
    Vdi = lambda x: ev_num(res["V"], np.array([x]))[0]
    Mdi = lambda x: ev_num(res["M"], np.array([x]))[0]
    eps = 1e-9 * L
    for e in range(ne):
        de = dd[2 * e:2 * e + 4]
        fe = ke_list[e] @ de + FEF[e]
        xi, xj = nodos[e], nodos[e + 1]
        print(f"  Elem {e+1}  f = [{fe[0]:.4f}, {fe[1]:.4f}, {fe[2]:.4f}, {fe[3]:.4f}]")
        print(f"    Rigidez : V ini={fe[0]:9.4f}  M ini={-fe[1]:9.4f}  V fin={-fe[2]:9.4f}  M fin={fe[3]:9.4f}")
        print(f"    Integr. : V ini={Vdi(xi+eps):9.4f}  M ini={Mdi(xi):9.4f}  "
              f"V fin={Vdi(xj-eps):9.4f}  M fin={Mdi(xj):9.4f}")

    sub("RESUMEN COMPARATIVO (doble integración vs rigidez)")
    print(f"  {'Magnitud':<18}{'Doble integr.':>16}{'Rigidez':>16}{'Diferencia':>14}")
    for nm in ("RA", "MA", "RB", "MB"):
        if nm in rea:
            print(f"  {nm:<18}{res['rea'][nm]:16.6f}{rea[nm]:16.6f}{abs(res['rea'][nm]-rea[nm]):14.2e}")
    for i, x in enumerate(nodos):
        xx = np.array([x])
        yd = ev_num(res["y"], xx)[0] / EI * 1000
        print(f"  {'y(x=%.3f) [mm]' % x:<18}{yd:16.6f}{dd[2*i]*1000:16.6f}{abs(yd - dd[2*i]*1000):14.2e}")
        thd = ev_num(res["th"], xx)[0] / EI
        print(f"  {'θ(x=%.3f) [rad]' % x:<18}{thd:16.8f}{dd[2*i+1]:16.8f}{abs(thd - dd[2*i+1]):14.2e}")
    return dict(rea=rea, dd=dd, nodos=nodos)   # (añadido) para mostrar tablas en la app


# ════════════════════════════════════════════════════════════════════
# ORQUESTACIÓN: ejecuta tus funciones capturando sus print()
# ════════════════════════════════════════════════════════════════════
def capturar(fn, *a, **k):
    """Ejecuta fn y devuelve (resultado, texto impreso con print)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **k)
    return r, buf.getvalue()


def ejecutar(op, sv, v):
    sec, t_sec = capturar(seccion, op, sv)
    d, t_dat = capturar(datos_viga, sec, v)
    res, t_di = capturar(doble_integracion, d)
    esf, t_esf = (None, "")
    if sec["geom"]:
        esf, t_esf = capturar(esfuerzos, sec, res)
    rig, t_rig = capturar(rigidez, d, res)
    return dict(sec=sec, d=d, res=res, rig=rig, esf=esf,
                fig_esq=esquema(d), fig_diag=graficar(d, res),
                fig_Q=fig_Q(sec) if sec["geom"] else None,
                txt=dict(sec=t_sec, dat=t_dat, di=t_di, esf=t_esf, rig=t_rig))


def secciones(texto):
    """Divide la salida capturada en (título, cuerpo) usando los encabezados '── ' de sub()."""
    bloques, tit, cuerpo = [], None, []
    for ln in texto.split("\n"):
        if ln.startswith("── "):
            bloques.append((tit, cuerpo))
            tit, cuerpo = ln.strip("─ ").strip(), []
        else:
            cuerpo.append(ln)
    bloques.append((tit, cuerpo))
    out = []
    for tit, cuerpo in bloques:
        if tit is None:
            lineas = [l for l in cuerpo if l.strip() and not l.startswith("═")]
            if len(lineas) > 1:
                out.append((lineas[0].strip(), "\n".join(lineas[1:])))
        else:
            out.append((tit, "\n".join(cuerpo).strip("\n")))
    return out


def mostrar_proc(textos, clave, archivo):
    """textos: lista de salidas capturadas; cada paso se muestra en un st.expander monoespaciado."""
    c1, c2 = st.columns([3, 2])
    abrir = c1.checkbox("Expandir todos los pasos", key=f"exp_{clave}")
    c2.download_button("Descargar procedimiento (.txt)", "\n".join(textos), file_name=archivo,
                       key=f"dl_{clave}")
    bloques = [b for t in textos for b in secciones(t)]
    for i, (tit, cuerpo) in enumerate(bloques):
        with st.expander(tit, expanded=abrir or i == 0):
            st.code(cuerpo, language="text")


_PC = "width" in inspect.signature(st.plotly_chart).parameters


def mostrar_fig(fig):
    cfg = dict(displaylogo=False, modeBarButtonsToRemove=["lasso2d", "select2d"])
    if _PC:
        st.plotly_chart(fig, width="stretch", config=cfg)
    else:
        st.plotly_chart(fig, use_container_width=True, config=cfg)


def tabla(df, fmt="%.6g"):
    cfg = {c: st.column_config.NumberColumn(format=fmt)
           for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    st.dataframe(df, hide_index=True, column_config=cfg)


# ════════════════════════════════════════════════════════════════════
# INTERFAZ STREAMLIT
# ════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Análisis de vigas", page_icon="📐", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root { --navy:#0A1F44; --steel:#4682B4; --gris:#F1F4F8; --borde:#D9E1EA; }
html, body, .stApp, [class*="css"] { font-family:'Inter','Segoe UI',Helvetica,Arial,sans-serif; }
.block-container { padding-top:2.2rem; max-width:1320px; }
header[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer { visibility:hidden; }
h1,h2,h3,h4 { color:var(--navy); letter-spacing:-0.01em; font-weight:650; }
section[data-testid="stSidebar"] { background:var(--gris); border-right:1px solid var(--borde); }
section[data-testid="stSidebar"] [data-testid="stExpander"] { background:#fff; }
.side-title { font-size:.78rem; text-transform:uppercase; letter-spacing:.12em; font-weight:600;
              color:var(--steel); margin:.2rem 0 .6rem; }
.hero { background:linear-gradient(135deg,#0A1F44 0%,#16396B 100%); color:#fff;
        padding:1.5rem 1.9rem; border-radius:14px; margin-bottom:.9rem; }
.hero .t { font-size:1.75rem; font-weight:700; letter-spacing:-0.02em; margin:0; }
.hero .s { color:#B9CCE3; margin:.3rem 0 0; font-size:.95rem; }
.aviso { border-left:4px solid var(--steel); background:#EAF1F8; color:var(--navy);
         padding:.55rem .95rem; border-radius:6px; font-size:.88rem; margin-bottom:1.1rem; }
section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] { width:100%; }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
    background:var(--navy); border:1px solid var(--navy); color:#fff; font-weight:600;
    letter-spacing:.08em; padding:.85rem 1rem; font-size:1.05rem; border-radius:10px; }
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
    background:var(--steel); border-color:var(--steel); color:#fff; }
.stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid var(--borde); }
.stTabs [data-baseweb="tab"] { height:46px; padding:0 16px; font-weight:500; color:#4A5D78; }
.stTabs [aria-selected="true"] { color:var(--navy); font-weight:600; }
.stTabs [data-baseweb="tab-highlight"] { background-color:var(--steel); }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--borde); border-radius:12px;
                           padding:.8rem 1rem; box-shadow:0 1px 2px rgba(10,31,68,.05); }
[data-testid="stMetricLabel"] { color:#5B6E8A; }
[data-testid="stMetricValue"] { color:var(--navy); font-weight:650; }
[data-testid="stExpander"] { border:1px solid var(--borde); border-radius:10px; background:#fff; }
[data-testid="stExpander"] summary { font-weight:500; color:var(--navy); }
[data-testid="stCode"] pre, [data-testid="stCode"] code, .stCodeBlock pre, .stCodeBlock code {
    font-family:'JetBrains Mono','DejaVu Sans Mono',Consolas,'Courier New',monospace !important;
    font-size:.80rem !important; line-height:1.45 !important; }
.pie { color:#6B7C93; font-size:.8rem; text-align:center; margin-top:2rem; }
</style>
""", unsafe_allow_html=True)

AVISO = "Herramienta educativa, no certificada para diseño"
SEC_OPC = {"Rectangular": 1, "T": 2, "I (doble T)": 3, "Compuesta por rectángulos": 4,
           "Conozco A e I (sin Q)": 5}
APO = {v: k for k, v in NOM.items()}          # nombre → letra
errs = []

# ─────────────────────────  BARRA LATERAL  ─────────────────────────
with st.sidebar:
    st.markdown('<div class="side-title">Datos de entrada</div>', unsafe_allow_html=True)

    with st.expander("1 · Sección transversal", expanded=True):
        tipo = st.selectbox("Tipo de sección", list(SEC_OPC), index=0)
        op = SEC_OPC[tipo]
        sv = {}
        c1, c2 = st.columns(2)
        if op == 1:
            sv["b"] = c1.number_input("Base b (cm)", value=20.0, min_value=0.0, step=1.0, format="%g")
            sv["h"] = c2.number_input("Altura h (cm)", value=40.0, min_value=0.0, step=1.0, format="%g")
            ok = sv["b"] > 0 and sv["h"] > 0
        elif op in (2, 3):
            sv["bf"] = c1.number_input("Ancho ala bf (cm)", value=30.0, min_value=0.0, step=1.0, format="%g")
            sv["tf"] = c2.number_input("Espesor ala tf (cm)", value=5.0, min_value=0.0, step=0.5, format="%g")
            sv["hw"] = c1.number_input("Altura alma hw (cm)", value=30.0, min_value=0.0, step=1.0, format="%g")
            sv["tw"] = c2.number_input("Espesor alma tw (cm)", value=8.0, min_value=0.0, step=0.5, format="%g")
            ok = all(sv[k] > 0 for k in ("bf", "tf", "hw", "tw"))
        elif op == 4:
            st.caption("Una fila por rectángulo. y0 = borde inferior medido desde la base (cm).")
            df0 = pd.DataFrame({"b (cm)": [8.0, 30.0], "h (cm)": [30.0, 5.0], "y0 (cm)": [0.0, 30.0]})
            ed = st.data_editor(df0, num_rows="dynamic", hide_index=True, key="rects_ed",
                                column_config={c: st.column_config.NumberColumn(c, min_value=0.0, format="%g")
                                               for c in df0.columns})
            ed = ed.dropna(subset=["b (cm)", "h (cm)"]).fillna({"y0 (cm)": 0.0})
            sv["rects"] = [tuple(float(x) for x in r) for r in ed.to_numpy()]
            ok = len(sv["rects"]) > 0 and all(b > 0 and h > 0 for b, h, _ in sv["rects"])
        else:
            sv["A"] = c1.number_input("Área A (cm²)", value=800.0, min_value=0.0, step=10.0, format="%g")
            sv["I"] = c2.number_input("Inercia I (cm⁴)", value=106666.67, min_value=0.0, step=100.0, format="%g")
            ok = sv["A"] > 0 and sv["I"] > 0
        if not ok:
            errs.append("Las dimensiones de la sección deben ser positivas.")

    with st.expander("2 · Viga y apoyos", expanded=True):
        c1, c2 = st.columns(2)
        L = c1.number_input("Longitud L (m)", value=6.0, min_value=0.01, step=0.5, format="%g")
        E = c2.number_input("Módulo E (GPa)", value=200.0, min_value=0.001, step=5.0, format="%g",
                            help="Acero ≈ 200 · concreto ≈ 25 · madera ≈ 10")
        n_apo = list(APO)
        izq_n = c1.selectbox("Apoyo izquierdo", n_apo, index=1)
        der_n = c2.selectbox("Apoyo derecho", n_apo, index=1)
        izq, der = APO[izq_n], APO[der_n]
        estable = izq == "E" or der == "E" or (izq == "A" and der == "A")
        if not estable:
            st.warning("Combinación inestable (mecanismo). Elige otra, p. ej. simple–simple, "
                       "empotrado–libre, empotrado–empotrado…")

    with st.expander("3 · Cargas", expanded=True):
        st.caption("w y P positivas ↓ · momentos positivos antihorarios ↺")
        c1, c2 = st.columns(2)
        M0 = c1.number_input("Momento izq. (kN·m)", value=0.0, step=1.0, format="%g")
        ML = c2.number_input("Momento der. (kN·m)", value=0.0, step=1.0, format="%g")
        w_i = c1.number_input("w en x=0 (kN/m)", value=0.0, step=1.0, format="%g")
        unif = st.checkbox("Carga uniforme (w final = w inicial)", value=True)
        if unif:
            w_f = w_i
        else:
            w_f = c2.number_input("w en x=L (kN/m)", value=float(w_i), step=1.0, format="%g")
        P = c1.number_input("Carga puntual P (kN)", value=0.0, step=1.0, format="%g")
        a = 0.0
        if P != 0:
            a = c2.number_input("Posición de P (m)", value=float(L) / 2, min_value=0.0, max_value=float(L),
                                step=0.25, format="%g", key=f"a_{L:g}",
                                help="Distancia desde el extremo izquierdo. Por defecto L/2.")
        pp = st.checkbox("Incluir peso propio", value=False)
        gamma = 24.0
        if pp:
            gamma = st.number_input("γ (kN/m³)", value=24.0, min_value=0.0, step=1.0, format="%g",
                                    help="Concreto ≈ 24 · acero ≈ 78.5 · madera ≈ 6")

    with st.expander("4 · Deformación por cortante (opcional)", expanded=False):
        cort = st.checkbox("Incluir (viga de Timoshenko, parámetro Φ)", value=False)
        nu, kap = 0.3, 5 / 6
        if cort:
            c1, c2 = st.columns(2)
            nu = c1.number_input("Poisson ν", value=0.3, min_value=-0.99, max_value=0.5, step=0.05, format="%g")
            kap = c2.number_input("Factor κ", value=5 / 6, min_value=0.01, step=0.01, format="%.4f",
                                  help="Rectángulo: 5/6 ≈ 0.833")

    v = dict(L=L, E=E, izq=izq, der=der, M0=M0, ML=ML, w_i=w_i, w_f=w_f, P=P, a=a,
             pp=pp, gamma=gamma, cort=cort, nu=nu, kap=kap)
    calc = st.button("CALCULAR", type="primary", disabled=bool(errs) or not estable, key="calc")
    st.markdown(f'<div class="pie" style="margin-top:.8rem">{AVISO}</div>', unsafe_allow_html=True)

# ─────────────────────────  CABECERA  ─────────────────────────
st.markdown("""
<div class="hero">
  <div class="t">Análisis de vigas</div>
  <div class="s">Doble integración · Rigidez matricial · Propiedades de sección &nbsp;|&nbsp; kN · m · GPa · sección en cm</div>
</div>
""", unsafe_allow_html=True)
st.markdown(f'<div class="aviso"><b>{AVISO}.</b></div>', unsafe_allow_html=True)

if not estable:
    st.warning("**Apoyos inestables.** Con esa combinación la viga es un mecanismo y no se puede "
               "calcular. Cambia los apoyos en la barra lateral (simple–simple, empotrado–libre, "
               "empotrado–empotrado, …).")
    st.stop()
if errs:
    for e in errs:
        st.error(e)
    st.stop()

params = dict(op=op, sv=sv, v=v)
if calc or "out" not in st.session_state:
    with st.spinner("Calculando (doble integración y rigidez matricial)…"):
        try:
            st.session_state["out"] = ejecutar(op, sv, v)
            st.session_state["params"] = params
        except Exception as e:
            st.error(f"No se pudo completar el cálculo: {e}")
            st.stop()
out = st.session_state["out"]
if st.session_state["params"] != params:
    st.info("Los datos de la barra lateral cambiaron. Pulsa **CALCULAR** para actualizar los resultados.")

sec, d, res, rig, esf, txt = (out[k] for k in ("sec", "d", "res", "rig", "esf", "txt"))
Lf, EI = float(d["L"]), res["EI"]

t_res, t_dia, t_di, t_rig, t_sec = st.tabs(["Resultados", "Diagramas",
                                            "Procedimiento: doble integración",
                                            "Procedimiento: rigidez", "Sección y Q"])

# ── Resultados ──────────────────────────────────────────────────────
with t_res:
    xs_ = np.linspace(0, Lf, 4001)
    V_, M_ = ev_num(res["V"], xs_), ev_num(res["M"], xs_)
    xV = xs_[np.argmax(np.abs(V_))]
    cols = st.columns(5 if sec["geom"] else 3)
    cols[0].metric("Cortante máx. |V|", f"{res['Vmax']:.4f} kN")
    cols[0].caption(f"en x = {xV:.3f} m")
    cols[1].metric("Momento máx. |M|", f"{res['Mmax']:.4f} kN·m")
    cols[1].caption(f"en x = {res['xM']:.3f} m")
    cols[2].metric("Flecha máx. |y|", f"{res['ymax']*1000:.4f} mm")
    cols[2].caption(f"en x = {res['xy']:.3f} m")
    if sec["geom"]:
        cols[3].metric("σ máx. (flexión)", f"{esf['sig']:.4f} MPa")
        cols[3].caption("|M|·c / I")
        cols[4].metric("τ en eje neutro", f"{esf['tau']:.4f} MPa")
        cols[4].caption("|V|·Q_EN / (I·b)")
    st.caption("Convenciones: cargas ↓ (+) · momentos y reacciones de momento ↺ (+) · reacciones "
               "verticales ↑ (+) · M(+) tracciona la fibra inferior · y ↑ (+) · θ ↺ (+).")

    st.markdown("#### Datos y propiedades")
    filas = [("Longitud L", f"{Lf:g}", "m"), ("Módulo E", f"{d['E']:g}", "GPa"),
             ("Apoyos", f"{NOM[d['izq']]} (izq.) / {NOM[d['der']]} (der.)", "—"),
             ("Área A", f"{sec['A']*1e4:.6g}", "cm²"), ("Inercia I", f"{sec['I']*1e8:.6g}", "cm⁴"),
             ("Rigidez EI", f"{EI:.6g}", "kN·m²"),
             ("Carga distribuida w(0) → w(L)", f"{d['w1']:.5g} → {d['w2']:.5g}", "kN/m")]
    if d["P"] != 0:
        filas.append(("Carga puntual P en x = a", f"{d['P']:g} en {d['a']:g}", "kN · m"))
    if d["M0"] or d["ML"]:
        filas.append(("Momentos aplicados (izq. / der.)", f"{d['M0']:g} / {d['ML']:g}", "kN·m"))
    if d["w_pp"]:
        filas.append(("Peso propio w_pp = γ·A", f"{d['w_pp']:.5g}", "kN/m"))
    if d["beta"]:
        filas.append(("β = EI/(G·As)", f"{d['beta']:.6g}", "m²"))
        filas.append(("Φ = 12β/L² (toda la viga)", f"{12*d['beta']/Lf**2:.6g}", "—"))
    if sec["geom"]:
        filas += [("Centroide ȳ (desde la base)", f"{sec['yc']:.4f}", "cm"),
                  ("Altura total H", f"{sec['H']:.4f}", "cm"),
                  ("Distancia a fibra extrema c", f"{sec['c']*100:.4f}", "cm"),
                  ("Q en el eje neutro", f"{sec['Q']*1e6:.4f}", "cm³"),
                  ("Ancho en el eje neutro b", f"{sec['bna']*100:.4f}", "cm")]
    st.dataframe(pd.DataFrame(filas, columns=["Magnitud", "Valor", "Unidad"]), hide_index=True)

    st.markdown("#### Reacciones en los apoyos")
    nom_r = {"RA": "R_A · fuerza vertical izquierda (↑ +)", "MA": "M_A · momento izquierdo (↺ +)",
             "RB": "R_B · fuerza vertical derecha (↑ +)", "MB": "M_B · momento derecho (↺ +)"}
    filas = [(nom_r[k], res["rea"][k], rig["rea"][k], abs(res["rea"][k] - rig["rea"][k]),
              "kN" if k[0] == "R" else "kN·m") for k in ("RA", "MA", "RB", "MB") if k in rig["rea"]]
    tabla(pd.DataFrame(filas, columns=["Reacción", "Doble integración", "Rigidez", "|Diferencia|", "Unidad"]),
          "%.6g")

    st.markdown("#### Valores en puntos clave (doble integración)")
    puntos = sorted(set([0, Lf / 4, Lf / 2, 3 * Lf / 4, Lf] + ([float(d["a"])] if d["P"] != 0 else [])))
    filas = []
    for xv in puntos:
        xa = np.array([min(max(xv, 1e-9 * Lf), Lf - 1e-9 * Lf)])
        filas.append((xv, ev_num(res["V"], xa)[0], ev_num(res["M"], xa)[0],
                      ev_num(res["th"], xa)[0] / EI, ev_num(res["y"], xa)[0] / EI * 1000))
    tabla(pd.DataFrame(filas, columns=["x [m]", "V [kN]", "M [kN·m]", "θ [rad]", "y [mm]"]))

    st.markdown("#### Comparación en los nodos: doble integración vs rigidez")
    filas = []
    for i, x in enumerate(rig["nodos"]):
        xx = np.array([x])
        yd = ev_num(res["y"], xx)[0] / EI * 1000
        thd = ev_num(res["th"], xx)[0] / EI
        yr, thr = rig["dd"][2 * i] * 1000, rig["dd"][2 * i + 1]
        filas.append((x, yd, yr, abs(yd - yr), thd, thr, abs(thd - thr)))
    tabla(pd.DataFrame(filas, columns=["x [m]", "y integr. [mm]", "y rigidez [mm]", "|Δy| [mm]",
                                       "θ integr. [rad]", "θ rigidez [rad]", "|Δθ| [rad]"]))

    st.markdown("#### Máximos")
    filas = [("|V| máx", res["Vmax"], "kN", xV), ("|M| máx", res["Mmax"], "kN·m", res["xM"]),
             ("|y| máx", res["ymax"] * 1000, "mm", res["xy"])]
    tabla(pd.DataFrame(filas, columns=["Magnitud", "Valor", "Unidad", "en x [m]"]))
    if sec["geom"]:
        tabla(pd.DataFrame([("σ máx = |M|·c/I", esf["sig"], "MPa"),
                            ("τ en el eje neutro = |V|·Q_EN/(I·b)", esf["tau"], "MPa"),
                            ("|σ| máx del diagrama", esf["smax"], "MPa"),
                            ("τ máx del diagrama", esf["tmax"], "MPa")],
                           columns=["Esfuerzo", "Valor", "Unidad"]))

# ── Diagramas ───────────────────────────────────────────────────────
with t_dia:
    st.caption("Pasa el cursor para ver los valores · arrastra para hacer zoom · doble clic para restablecer.")
    mostrar_fig(out["fig_esq"])
    mostrar_fig(out["fig_diag"])

# ── Procedimiento: doble integración ───────────────────────────────
with t_di:
    st.caption("Salida real del procedimiento paso a paso (datos de la viga + doble integración).")
    mostrar_proc([txt["dat"], txt["di"]], "di", "procedimiento_doble_integracion.txt")

# ── Procedimiento: rigidez ─────────────────────────────────────────
with t_rig:
    st.caption("Salida real del método de rigidez matricial, paso a paso.")
    mostrar_proc([txt["rig"]], "rig", "procedimiento_rigidez.txt")

# ── Sección y Q ────────────────────────────────────────────────────
with t_sec:
    if sec["geom"]:
        st.markdown("#### Sección, ancho b(y) y primer momento de área Q(y)")
        mostrar_fig(out["fig_Q"])
        st.markdown("#### Esfuerzos en la sección: σ(y) y τ(y)")
        mostrar_fig(esf["fig"])
        st.markdown("#### Procedimiento")
        mostrar_proc([txt["sec"], txt["esf"]], "sec", "procedimiento_seccion.txt")
    else:
        st.info("Elegiste «Conozco A e I»: no se calcula Q(y) ni los diagramas de σ(y) y τ(y). "
                "Selecciona una sección geométrica en la barra lateral para verlos.")

st.markdown(f'<div class="pie">{AVISO}.</div>', unsafe_allow_html=True)
