#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISIS DE VIGAS (unidades: kN, m, GPa; sección en cm)
========================================================
1) Sección: primer momento de área Q(y) con su DIAGRAMA, centroide e inercia.
2) Doble integración paso a paso (V, M, θ, y) mostrando de dónde sale cada ecuación.
3) Diagramas de V, M, flecha y de esfuerzos σ(y), τ(y) en la sección.
4) Rigidez matricial paso a paso: deducción de k con funciones de forma,
   matriz con parámetro Φ (cortante, opcional), ensamble, Gauss y reacciones.

CONVENCIONES
  x → derecha desde el extremo izquierdo. Cargas (w, P) + hacia ABAJO.
  Momentos aplicados y reacciones de momento + ANTIHORARIOS (↺).
  Reacciones verticales + hacia ARRIBA. M(+) tracciona la fibra inferior.
  y (flecha) + hacia arriba; θ (giro de la sección) + antihorario.

Requiere: pip install numpy sympy matplotlib
"""
import sys
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SUP = {1: "E", 2: "A", 3: "L"}
NOM = {"E": "Empotrado", "A": "Apoyo simple", "L": "Libre"}
_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


# ════════════════════════════════════════════════════════════════════
# UTILIDADES
# ════════════════════════════════════════════════════════════════════
def pedir(texto, defecto=None, tipo=float):
    while True:
        extra = f" [{defecto:g}]" if isinstance(defecto, (int, float)) else ""
        s = input(f"{texto}{extra}: ").strip().replace(",", ".")
        if s == "" and defecto is not None:
            return defecto
        try:
            return tipo(s)
        except ValueError:
            print("  ⚠ Valor no válido, intenta de nuevo.")


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
def seccion():
    titulo("PASO 0 · SECCIÓN: PRIMER MOMENTO DE ÁREA (Q), CENTROIDE E INERCIA")
    print("  1) Rectangular\n  2) T\n  3) I (doble T)\n"
          "  4) Compuesta por rectángulos\n  5) Ya conozco A e I (no se calcula Q)")
    op = pedir("Tipo de sección", 1, int)
    rects = []
    if op == 1:
        b = pedir("  Base b (cm)", 20.0)
        h = pedir("  Altura h (cm)", 40.0)
        rects = [(b, h, 0.0)]
    elif op in (2, 3):
        bf = pedir("  Ancho del ala bf (cm)", 30.0)
        tf = pedir("  Espesor del ala tf (cm)", 5.0)
        hw = pedir("  Altura libre del alma hw (cm)", 30.0)
        tw = pedir("  Espesor del alma tw (cm)", 8.0)
        if op == 2:
            rects = [(tw, hw, 0.0), (bf, tf, hw)]
        else:
            rects = [(bf, tf, 0.0), (tw, hw, tf), (bf, tf, tf + hw)]
    elif op == 4:
        n = pedir("  Número de rectángulos", 2, int)
        for i in range(1, n + 1):
            b = pedir(f"  Rect {i}: base b (cm)")
            h = pedir(f"  Rect {i}: altura h (cm)")
            y0 = pedir(f"  Rect {i}: y de su borde inferior medido desde la base (cm)", 0.0)
            rects.append((b, h, y0))
    else:
        A = pedir("  Área A (cm²)")
        I = pedir("  Inercia I (cm⁴)")
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


def _dibujar_seccion(ax, sec):
    rects, yc = sec["rects"], sec["yc"]
    bmax = max(b for b, _, _ in rects)
    cols = ["#9ecae1", "#a1d99b", "#fdd0a2", "#bcbddc", "#fcbba1"]
    for i, (b, h, y0) in enumerate(rects):
        ax.add_patch(Rectangle((-b / 2, y0), b, h, fc=cols[i % 5], ec="k"))
        ax.text(0, y0 + h / 2, f"{i+1}", ha="center", va="center")
    ax.axhline(yc, color="r", ls="--", lw=1.2)
    ax.text(-bmax * 0.6, yc, f" ȳ = {yc:.2f} cm", color="r", va="bottom")
    ax.set_xlim(-bmax * 0.75, bmax * 0.75)
    ax.set_ylim(0, sec["H"])
    ax.set_title("Sección (cm)")
    ax.set_ylabel("y desde la base [cm]")


def fig_Q(sec):
    ys = _grid_y(sec)
    Q = np.array([sec["Q_en"](y)[0] for y in ys])
    bb = np.array([sec["b_en"](y) for y in ys])
    fig, ax = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
    _dibujar_seccion(ax[0], sec)
    ax[1].plot(bb, ys, "k-", lw=2)
    ax[1].fill_betweenx(ys, 0, bb, color="gray", alpha=0.25)
    ax[1].set_title("Ancho b(y) [cm]")
    ax[2].plot(Q, ys, color="tab:purple", lw=2.5)
    ax[2].fill_betweenx(ys, 0, Q, color="tab:purple", alpha=0.25)
    ax[2].set_title("Primer momento de área Q(y) [cm³]", weight="bold")
    for y in sec["niveles"]:
        q = sec["Q_en"](min(max(y, 1e-6), sec["H"] - 1e-6))[0]
        for a in ax:
            a.axhline(y, color="gray", ls=":", lw=0.7)
        ax[2].plot([q], [y], "o", color="tab:purple")
        if q > 1e-3 * max(abs(Q.max()), 1):
            ax[2].annotate(f"{q:.1f}", (q, y), textcoords="offset points", xytext=(6, 4),
                           color="tab:purple", weight="bold")
    for a in ax[1:]:
        a.grid(alpha=0.3)
    ax[2].set_xlabel("Q(y) = área sobre el nivel y × distancia al eje neutro")
    plt.tight_layout()
    plt.savefig("seccion_Q.png", dpi=150)
    print("  📈 Diagrama de Q(y) guardado en 'seccion_Q.png'")


def fig_esfuerzos(sec, res):
    ys = _grid_y(sec)
    I_cm = sec["I"] * 1e8
    Q = np.array([sec["Q_en"](y)[0] for y in ys])
    bb = np.array([sec["b_en"](y) for y in ys])
    sig = -res["Mmax"] * 100 * (ys - sec["yc"]) / I_cm * 10      # MPa
    tau = abs(res["Vmax"]) * Q / (I_cm * bb) * 10                # MPa
    fig, ax = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
    _dibujar_seccion(ax[0], sec)
    for a, v, col, tit in (
            (ax[1], sig, "tab:green", f"σ(y) = −M·(y−ȳ)/I   con M = {res['Mmax']:.2f} kN·m   [MPa]"),
            (ax[2], tau, "tab:blue", f"τ(y) = V·Q(y)/(I·b(y))   con |V| = {abs(res['Vmax']):.2f} kN   [MPa]")):
        a.plot(v, ys, color=col, lw=2.5)
        a.fill_betweenx(ys, 0, v, color=col, alpha=0.25)
        a.axvline(0, color="k", lw=0.8)
        a.axhline(sec["yc"], color="r", ls="--", lw=0.8)
        a.set_title(tit, fontsize=9)
        a.grid(alpha=0.3)
        i = int(np.argmax(np.abs(v)))
        a.annotate(f"{v[i]:.3f}", (v[i], ys[i]), textcoords="offset points", xytext=(6, 6),
                   color=col, weight="bold")
    plt.tight_layout()
    plt.savefig("seccion_esfuerzos.png", dpi=150)
    print("  📈 Esfuerzos en la sección guardados en 'seccion_esfuerzos.png'")
    return np.max(np.abs(sig)), np.max(tau)


# ════════════════════════════════════════════════════════════════════
# 2) DATOS DE LA VIGA
# ════════════════════════════════════════════════════════════════════
def datos_viga(sec):
    titulo("DATOS DE LA VIGA")
    L = pedir("Longitud de la viga L (m)", 6.0)
    E = pedir("Módulo de elasticidad E (GPa)  [acero≈200, concreto≈25, madera≈10]", 200.0)
    EI = E * 1e6 * sec["I"]
    print(f"  EI = E·I = {E:g}·10⁶ kN/m² · {sec['I']:.6e} m⁴ = {EI:.5g} kN·m²")
    print("\nTipos de apoyo: 1) Empotrado   2) Apoyo simple (articulado/rodillo)   3) Libre")
    while True:
        izq = SUP.get(pedir("Apoyo IZQUIERDO (1/2/3)", 2, int))
        der = SUP.get(pedir("Apoyo DERECHO   (1/2/3)", 2, int))
        if izq and der and (izq == "E" or der == "E" or (izq == "A" and der == "A")):
            break
        print("  ⚠ Combinación inestable (mecanismo). Elige otra, p.ej. simple-simple,"
              " empotrado-libre, empotrado-empotrado...")
    print(f"  → Izquierdo: {NOM[izq]} | Derecho: {NOM[der]}")
    print("\nCargas (w, P positivas ↓ ; momentos positivos antihorarios ↺)")
    M0 = pedir("Momento aplicado en el extremo IZQUIERDO (kN·m)", 0.0)
    ML = pedir("Momento aplicado en el extremo DERECHO (kN·m)", 0.0)
    w_i = pedir("Carga distribuida en x=0 (kN/m)", 0.0)
    w_f = pedir("Carga distribuida en x=L (kN/m)  [igual a la inicial = uniforme]", w_i)
    P = pedir("Carga puntual P (kN)", 0.0)
    a = 0.0
    if P != 0:
        while True:
            a = pedir("  Distancia de P desde el extremo izquierdo (m)", L / 2)
            if 0 <= a <= L:
                break
            print("  ⚠ Debe estar entre 0 y L.")
    w_pp = 0.0
    if input("\n¿Incluir peso propio? (s/n) [n]: ").strip().lower().startswith("s"):
        g = pedir("  Peso específico γ (kN/m³) [concreto≈24, acero≈78.5, madera≈6]", 24.0)
        w_pp = g * sec["A"]
        print(f"  w_pp = γ·A = {g:g}·{sec['A']:.6f} = {w_pp:.5g} kN/m")
    beta = 0.0
    if input("\n¿Incluir deformación por cortante (viga de Timoshenko, parámetro Φ)? (s/n) [n]: "
             ).strip().lower().startswith("s"):
        nu = pedir("  Coeficiente de Poisson ν", 0.3)
        kap = pedir("  Factor de cortante κ [rectángulo 5/6≈0.833]", 5 / 6)
        G = E / (2 * (1 + nu))
        As = kap * sec["A"]
        beta = EI / (G * 1e6 * As)
        print(f"  G = E/(2(1+ν)) = {G:.4f} GPa ; As = κ·A = {As:.6f} m²")
        print(f"  β = EI/(G·As) = {beta:.6g} m²  →  Φ = 12β/L² = {12*beta/L**2:.6g} (para toda la viga)")
    return dict(L=L, E=E, EI=EI, izq=izq, der=der, M0=M0, ML=ML,
                w1=w_i + w_pp, w2=w_f + w_pp, P=P, a=a, w_pp=w_pp, beta=beta)


# ════════════════════════════════════════════════════════════════════
# 3) DOBLE INTEGRACIÓN
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
# 4) DIAGRAMAS DE LA VIGA
# ════════════════════════════════════════════════════════════════════
def esquema(ax, d):
    L = d["L"]
    ax.set_xlim(-0.1 * L, 1.1 * L)
    ax.set_ylim(-1.2, 2.4)
    ax.axis("off")
    ax.plot([0, L], [0, 0], "k-", lw=6, solid_capstyle="butt")
    for x, t, lado in ((0, d["izq"], "i"), (L, d["der"], "d")):
        if t == "A":
            ax.fill([x - 0.025 * L, x + 0.025 * L, x], [-0.5, -0.5, -0.05], "gray", ec="k")
        elif t == "E":
            x0 = x - 0.03 * L if lado == "i" else x
            ax.add_patch(Rectangle((x0, -0.8), 0.03 * L, 1.6, hatch="///", fc="lightgray", ec="k"))
    w = lambda x: d["w1"] + (d["w2"] - d["w1"]) * x / L
    wm = max(abs(d["w1"]), abs(d["w2"]))
    if wm > 0:
        xs = np.linspace(0, L, 11)
        alt = [np.sign(w(x)) * (0.3 + 0.8 * abs(w(x)) / wm) for x in xs]
        for x, h in zip(xs, alt):
            if h != 0:
                ax.annotate("", xy=(x, 0.1 * np.sign(h)), xytext=(x, h),
                            arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=1.3))
        ax.plot(xs, alt, color="tab:blue", lw=1.3)
        ax.text(L / 2, max(alt + [0]) + 0.15,
                f"w = {d['w1']:.4g} → {d['w2']:.4g} kN/m" + (" (incl. peso propio)" if d["w_pp"] else ""),
                ha="center", color="tab:blue")
    if d["P"] != 0:
        s = np.sign(d["P"])
        ax.annotate("", xy=(d["a"], 0.1 * s), xytext=(d["a"], 1.9 * s),
                    arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=2.5))
        ax.text(d["a"], 2.0 * s + (0.05 if s > 0 else -0.25), f"P = {d['P']:g} kN",
                ha="center", color="tab:red")
    for x, m, ha in ((0, d["M0"], "left"), (L, d["ML"], "right")):
        if m != 0:
            ax.text(x, -0.9, f"M = {abs(m):g} kN·m {'↺' if m > 0 else '↻'}", ha=ha,
                    color="tab:green", fontsize=11, weight="bold")


def graficar(d, res):
    L = d["L"]
    eps = 1e-9 * L
    pts = set(np.linspace(0, L, 1601))
    if d["P"] != 0:
        pts |= {max(d["a"] - eps, 0), min(d["a"] + eps, L)}
    xs = np.array(sorted(pts))
    V, M = ev_num(res["V"], xs), ev_num(res["M"], xs)
    y = ev_num(res["y"], xs) / res["EI"] * 1000
    fig, ax = plt.subplots(4, 1, figsize=(11, 13), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.1, 1, 1, 1]))
    esquema(ax[0], d)
    ax[0].set_title("Viga y cargas  (+: cargas ↓, momentos ↺)")
    xv = np.concatenate([[0], xs, [L]])
    Vv = np.concatenate([[0], V, [0]])
    for a_, xx, yy, col, tit, uni in (
            (ax[1], xv, Vv, "tab:blue", "Diagrama de CORTANTE V(x)", "kN"),
            (ax[2], xs, M, "tab:green", "Diagrama de MOMENTO FLECTOR M(x)  (+ = tracción abajo)", "kN·m"),
            (ax[3], xs, y, "tab:red", "Elástica: FLECHA y(x)  (+ = hacia arriba)", "mm")):
        a_.axhline(0, color="k", lw=0.8)
        a_.plot(xx, yy, color=col, lw=2)
        a_.fill_between(xx, yy, 0, color=col, alpha=0.2)
        a_.set_title(tit)
        a_.set_ylabel(uni)
        a_.grid(alpha=0.3)
        i = int(np.argmax(np.abs(yy)))
        a_.annotate(f"{yy[i]:.4g} {uni}", (xx[i], yy[i]), textcoords="offset points",
                    xytext=(8, 8 if yy[i] >= 0 else -16), color=col, weight="bold")
    ax[3].set_xlabel("x [m]")
    ax[3].set_xlim(-0.1 * L, 1.1 * L)
    plt.tight_layout()
    plt.savefig("viga_diagramas.png", dpi=150)
    print("  📈 Diagramas de la viga guardados en 'viga_diagramas.png'")


# ════════════════════════════════════════════════════════════════════
# 5) MÉTODO DE RIGIDEZ MATRICIAL
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


# ════════════════════════════════════════════════════════════════════
def main():
    print("╔" + "═" * 78 + "╗")
    print("║" + "ANÁLISIS DE VIGAS: DOBLE INTEGRACIÓN + RIGIDEZ MATRICIAL".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print("Unidades: kN, m, GPa (sección en cm). Enter = valor por defecto.")
    sec = seccion()
    if sec["geom"]:
        fig_Q(sec)
    d = datos_viga(sec)
    res = doble_integracion(d)
    if sec["geom"]:
        sub("ESFUERZOS EN LA SECCIÓN (con |M|máx y |V|máx)")
        print("  Flexión: σ(y) = −M·(y−ȳ)/I     Cortante: τ(y) = V·Q(y)/(I·b(y))")
        sig = abs(res["Mmax"]) * sec["c"] / sec["I"] / 1000
        tau = abs(res["Vmax"]) * sec["Q"] / (sec["I"] * sec["bna"]) / 1000
        print(f"  σ_máx = |M|·c / I = {abs(res['Mmax']):.4f}·{sec['c']:.4f} / {sec['I']:.4e} = {sig:.4f} MPa")
        print(f"  τ en el eje neutro = |V|·Q_EN / (I·b) = {abs(res['Vmax']):.4f}·{sec['Q']:.4e} / "
              f"({sec['I']:.4e}·{sec['bna']:.4f}) = {tau:.4f} MPa")
        smax, tmax = fig_esfuerzos(sec, res)
        print(f"  Máximos del diagrama: |σ| = {smax:.4f} MPa , τ = {tmax:.4f} MPa")
    graficar(d, res)
    print("\n  (Cierra las ventanas de las figuras para continuar con el método de rigidez.)")
    plt.show()
    rigidez(d, res)
    print("\n✔ Listo.")


if __name__ == "__main__":
    main()
