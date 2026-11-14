"""
MTL 712 / MTL 7012 - Lab Sheet II
==================================
Numerical solution of ODEs from scratch:
    1. Classical RK4 for a system of two equations
    2. A higher-order IVP reduced to a first-order system
    3. Lotka-Volterra predator-prey system (RK4 vs Heun / improved Euler)
    4. Milne's predictor-corrector method
    5. Comparative study (RK4 vs Milne)

All ODE solvers are written generically: they take f(x, y) -> array and work
for a system of any size. No built-in ODE solver (solve_ivp, odeint, ...) is
used anywhere. numpy/matplotlib/pandas are used only for arithmetic, tables
and plotting, as permitted by the instructions.

Running this file regenerates every table (printed to console + saved as
.csv) and every plot (saved as .png) inside the "outputs" folder.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)


# ======================================================================
# 0. GENERIC SOLVERS
# ======================================================================

def rk4_system(f, y0, x0, xf, h):
    """
    Classical fourth-order Runge-Kutta method for y' = f(x, y), y(x0) = y0.

    f  : callable f(x, y) -> np.ndarray, same shape as y
    y0 : array-like, initial condition (any number of equations)

    Returns
    -------
    x : ndarray, shape (N+1,)
    Y : ndarray, shape (N+1, n_eq)
    """
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    n_eq = y0.size
    N = int(round((xf - x0) / h))
    x = x0 + h * np.arange(N + 1)
    Y = np.zeros((N + 1, n_eq))
    Y[0] = y0
    for n in range(N):
        xn, yn = x[n], Y[n]
        k1 = f(xn, yn)
        k2 = f(xn + h / 2, yn + h / 2 * k1)
        k3 = f(xn + h / 2, yn + h / 2 * k2)
        k4 = f(xn + h, yn + h * k3)
        Y[n + 1] = yn + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x, Y


def heun_system(f, y0, x0, xf, h):
    """
    Improved Euler / Heun's method (predictor-corrector, single corrector
    pass), as used in Lab Sheet-I.
    """
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    n_eq = y0.size
    N = int(round((xf - x0) / h))
    x = x0 + h * np.arange(N + 1)
    Y = np.zeros((N + 1, n_eq))
    Y[0] = y0
    for n in range(N):
        xn, yn = x[n], Y[n]
        k1 = f(xn, yn)
        y_pred = yn + h * k1
        k2 = f(xn + h, y_pred)
        Y[n + 1] = yn + (h / 2) * (k1 + k2)
    return x, Y


def milne_pc(f, y0, x0, xf, h, tol=1e-10, max_iter=500):
    """
    Milne's predictor-corrector method.

        Predictor: y_{n+1}^(0) = y_{n-3} + (4h/3)(2f_n - f_{n-1} + 2f_{n-2})
        Corrector: y_{n+1}^(k+1) = y_{n-1} + (h/3)(f_{n-1} + 4f_n + f_{n+1}^(k))

    Starting values y0, y1, y2, y3 are generated with classical RK4 using the
    same step size h. The corrector is iterated until the max-norm change
    between successive iterates is below `tol`.

    Returns a dict with:
        x        : mesh points
        Y        : corrected solution at every mesh point (incl. RK4 starters)
        Ypred    : predicted value at each point (NaN for the first 4 starters)
        n_iters  : number of corrector iterations used at each point
        err_est  : Milne's error estimate  -(1/29)(y_{n+1} - y_{n+1}^(0))
    """
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    n_eq = y0.size
    N = int(round((xf - x0) / h))
    if N < 3:
        raise ValueError("Need at least 3 steps (4 mesh points) to start Milne's method.")

    x = x0 + h * np.arange(N + 1)
    Y = np.zeros((N + 1, n_eq))
    Ypred = np.full((N + 1, n_eq), np.nan)
    n_iters = np.zeros(N + 1, dtype=int)
    err_est = np.zeros((N + 1, n_eq))

    # --- starting values via RK4 ---
    _, Y_start = rk4_system(f, y0, x0, x0 + 3 * h, h)
    Y[0:4] = Y_start
    F = [f(x[i], Y[i]) for i in range(4)]  # f0, f1, f2, f3

    # --- march forward with predictor-corrector ---
    for n in range(3, N):
        f_nm2, f_nm1, f_n = F[n - 2], F[n - 1], F[n]

        y_pred = Y[n - 3] + (4 * h / 3) * (2 * f_n - f_nm1 + 2 * f_nm2)
        Ypred[n + 1] = y_pred

        y_corr = y_pred.copy()
        k_used = 0
        for k in range(1, max_iter + 1):
            f_np1 = f(x[n + 1], y_corr)
            y_new = Y[n - 1] + (h / 3) * (f_nm1 + 4 * f_n + f_np1)
            k_used = k
            if np.max(np.abs(y_new - y_corr)) < tol:
                y_corr = y_new
                break
            y_corr = y_new

        Y[n + 1] = y_corr
        n_iters[n + 1] = k_used
        err_est[n + 1] = -(1.0 / 29.0) * (Y[n + 1] - Ypred[n + 1])
        F.append(f(x[n + 1], Y[n + 1]))

    return {"x": x, "Y": Y, "Ypred": Ypred, "n_iters": n_iters, "err_est": err_est}


def save_table(df, name):
    path = os.path.join(OUTDIR, name + ".csv")
    df.to_csv(path, index=False)
    print(f"\n--- {name} ---")
    print(df.to_string(index=False))
    print(f"[saved table -> {path}]")


def save_fig(fig, name):
    path = os.path.join(OUTDIR, name + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved figure -> {path}]")


# ======================================================================
# SECTION 1: RK4 for a system of two equations
# ======================================================================

def section1():
    print("\n" + "=" * 70)
    print("SECTION 1: RK4 for the linear system y1' = -2y1+y2, y2' = y1-2y2")
    print("=" * 70)

    def f(x, y):
        y1, y2 = y
        return np.array([-2 * y1 + y2, y1 - 2 * y2])

    def exact(x):
        y1e = np.exp(-x) + np.exp(-3 * x)
        y2e = np.exp(-x) - np.exp(-3 * x)
        return y1e, y2e

    y0 = [2.0, 0.0]
    results = {}
    for h in (0.1, 0.05):
        x, Y = rk4_system(f, y0, 0.0, 1.0, h)
        y1e, y2e = exact(x)
        err1 = np.abs(Y[:, 0] - y1e)
        err2 = np.abs(Y[:, 1] - y2e)
        results[h] = (x, Y, y1e, y2e, err1, err2)

        df = pd.DataFrame({
            "x_n": x,
            "y1_numeric": Y[:, 0],
            "y1_exact": y1e,
            "|err_y1|": err1,
            "y2_numeric": Y[:, 1],
            "y2_exact": y2e,
            "|err_y2|": err2,
        })
        save_table(df, f"section1_h{h}")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, Y[:, 0], "o-", label="y1 numeric (RK4)")
        ax.plot(x, y1e, "--", label="y1 exact")
        ax.plot(x, Y[:, 1], "s-", label="y2 numeric (RK4)")
        ax.plot(x, y2e, "--", label="y2 exact")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Section 1: RK4 vs exact, h = {h}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_fig(fig, f"section1_h{h}_plot")

    # Order of accuracy check: halve h, error should drop by ~2^4 = 16
    max_err_01 = max(results[0.1][4].max(), results[0.1][5].max())
    max_err_005 = max(results[0.05][4].max(), results[0.05][5].max())
    ratio = max_err_01 / max_err_005 if max_err_005 > 0 else np.nan
    print(f"\nMax error at h=0.1   : {max_err_01:.3e}")
    print(f"Max error at h=0.05  : {max_err_005:.3e}")
    print(f"Error ratio (h -> h/2): {ratio:.2f}  (expect ~16 for a 4th-order method)")


# ======================================================================
# SECTION 2: Higher-order IVP  y'' + 2y' + 5y = 0
# ======================================================================

def section2():
    print("\n" + "=" * 70)
    print("SECTION 2: y'' + 2y' + 5y = 0  ->  system  y1'=y2, y2'=-2y2-5y1")
    print("=" * 70)

    def f(x, y):
        y1, y2 = y  # y1 = y, y2 = y'
        return np.array([y2, -2 * y2 - 5 * y1])

    def exact(x):
        return np.exp(-x) * (np.cos(2 * x) + 0.5 * np.sin(2 * x))

    y0 = [1.0, 0.0]

    # Fine grid (h=0.1, or finer) used ONLY to draw a smooth exact curve.
    # This matters for h=0.4, where only 5 mesh points would otherwise make
    # the exact solution look like a jagged polyline instead of a smooth
    # curve. The error TABLE still compares numeric vs. exact at the actual
    # mesh points of each run.
    h_exact_plot = 0.1
    x_fine = np.arange(0.0, 2.0 + h_exact_plot / 2, h_exact_plot)
    ye_fine = exact(x_fine)

    for h in (0.1, 0.4):
        x, Y = rk4_system(f, y0, 0.0, 2.0, h)
        ye = exact(x)
        err = np.abs(Y[:, 0] - ye)

        df = pd.DataFrame({
            "x_n": x, "y_numeric": Y[:, 0], "yprime_numeric": Y[:, 1],
            "y_exact": ye, "|error|": err,
        })
        save_table(df, f"section2_h{h}")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, Y[:, 0], "o-", ms=4, label=f"numeric (RK4, h={h})")
        ax.plot(x_fine, ye_fine, "--", label=f"exact (plotted at h={h_exact_plot})")
        ax.set_xlabel("x"); ax.set_ylabel("y(x)")
        ax.set_title(f"Section 2: y(x), h = {h}")
        ax.legend(); ax.grid(True, alpha=0.3)
        save_fig(fig, f"section2_h{h}_solution")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(Y[:, 0], Y[:, 1])
        ax.set_xlabel("y"); ax.set_ylabel("y'")
        ax.set_title(f"Section 2: phase plane (y, y'), h = {h}")
        ax.grid(True, alpha=0.3)
        save_fig(fig, f"section2_h{h}_phase")

        print(f"h={h}: max |error| = {err.max():.3e}, "
              f"max |y_numeric| = {np.max(np.abs(Y[:,0])):.3e}")


# ======================================================================
# SECTION 3: Lotka-Volterra predator-prey system
# ======================================================================

def section3():
    print("\n" + "=" * 70)
    print("SECTION 3: Lotka-Volterra predator-prey system")
    print("=" * 70)

    a, b, c, d = 1.5, 1.0, 3.0, 1.0

    def f(x, y):
        y1, y2 = y
        return np.array([a * y1 - b * y1 * y2, -c * y2 + d * y1 * y2])

    def V(y1, y2):
        return d * y1 - c * np.log(y1) + b * y2 - a * np.log(y2)

    y0 = [10.0, 5.0]
    h = 0.05
    V0 = V(*y0)

    x_rk4, Y_rk4 = rk4_system(f, y0, 0.0, 15.0, h)
    x_heun, Y_heun = heun_system(f, y0, 0.0, 15.0, h)

    V_rk4 = V(Y_rk4[:, 0], Y_rk4[:, 1])
    V_heun = V(Y_heun[:, 0], Y_heun[:, 1])
    dV_rk4 = np.abs(V_rk4 - V0)
    dV_heun = np.abs(V_heun - V0)

    df = pd.DataFrame({
        "x_n": x_rk4,
        "y1_RK4": Y_rk4[:, 0], "y2_RK4": Y_rk4[:, 1], "|V-V0|_RK4": dV_rk4,
        "y1_Heun": Y_heun[:, 0], "y2_Heun": Y_heun[:, 1], "|V-V0|_Heun": dV_heun,
    })
    save_table(df.iloc[::20], "section3_summary_every20th")  # full table is long; sample every 20th row

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_rk4, Y_rk4[:, 0], label="y1 (prey), RK4")
    ax.plot(x_rk4, Y_rk4[:, 1], label="y2 (predator), RK4")
    ax.set_xlabel("x"); ax.set_ylabel("population")
    ax.set_title("Section 3: y1(x), y2(x) - RK4")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_fig(fig, "section3_populations_vs_x")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(Y_rk4[:, 0], Y_rk4[:, 1])
    ax.set_xlabel("y1 (prey)"); ax.set_ylabel("y2 (predator)")
    ax.set_title("Section 3: phase plane trajectory (RK4)")
    ax.grid(True, alpha=0.3)
    save_fig(fig, "section3_phase_plane")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_rk4, dV_rk4, label="RK4")
    ax.plot(x_heun, dV_heun, label="Heun (improved Euler)")
    ax.set_xlabel("x"); ax.set_ylabel("|V - V0|")
    ax.set_yscale("log")
    ax.set_title("Section 3: growth of |V(y1,y2) - V(y1(0),y2(0))|")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    save_fig(fig, "section3_V_error")

    print(f"Final |V-V0| at x=15:  RK4 = {dV_rk4[-1]:.3e},  Heun = {dV_heun[-1]:.3e}")


# ======================================================================
# SECTION 4: Milne's predictor-corrector method
# ======================================================================

def _milne_table(f, y0, x0, xf, h, exact_fn, comp=0):
    res = milne_pc(f, y0, x0, xf, h)
    x, Y, Ypred, n_iters, err_est = (res["x"], res["Y"], res["Ypred"],
                                      res["n_iters"], res["err_est"])
    ye = exact_fn(x)
    actual_err = np.abs(Y[:, comp] - ye)
    df = pd.DataFrame({
        "x_n": x,
        "y_predicted": Ypred[:, comp],
        "y_corrected": Y[:, comp],
        "corrector_iters": n_iters,
        "Milne_error_estimate": err_est[:, comp],
        "actual_error": actual_err,
    })
    return x, Y, ye, actual_err, err_est, df


def section4():
    print("\n" + "=" * 70)
    print("SECTION 4: Milne's predictor-corrector method")
    print("=" * 70)

    # --- 4a: y' = 1 + y^2, y(0) = 0, exact = tan(x) ---
    def f_a(x, y):
        return np.array([1.0 + y[0] ** 2])

    x, Y, ye, actual_err, err_est, df = _milne_table(
        f_a, [0.0], 0.0, 1.0, 0.1, np.tan)
    save_table(df, "section4a_milne_tan")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, Y[:, 0], "o-", label="Milne numeric")
    ax.plot(x, ye, "--", label="exact = tan(x)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Section 4a: y' = 1+y^2, Milne vs exact")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_fig(fig, "section4a_solution")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, np.abs(err_est[:, 0]), "o-", label="Milne error estimate")
    ax.plot(x, actual_err, "s--", label="actual error")
    ax.set_xlabel("x"); ax.set_ylabel("error")
    ax.set_yscale("log")
    ax.set_title("Section 4a: Milne error estimate vs actual error")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    save_fig(fig, "section4a_error_estimate")

    # --- 4b: y' = x + y, y(0) = 1, exact = 2e^x - x - 1, compare with Heun ---
    def f_b(x, y):
        return np.array([x + y[0]])

    def exact_b(x):
        return 2 * np.exp(x) - x - 1

    x, Y, ye, actual_err, err_est, df = _milne_table(
        f_b, [1.0], 0.0, 1.0, 0.1, exact_b)
    save_table(df, "section4b_milne_xplusy")

    x_h, Y_h = heun_system(f_b, [1.0], 0.0, 1.0, 0.1)
    err_heun = np.abs(Y_h[:, 0] - exact_b(x_h))

    df_cmp = pd.DataFrame({
        "x_n": x,
        "Milne_error": actual_err,
        "Heun_error": err_heun,
    })
    save_table(df_cmp, "section4b_milne_vs_heun")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(x, actual_err + 1e-20, "o-", label="Milne |error|")
    ax.semilogy(x_h, err_heun + 1e-20, "^-", label="Heun |error|")
    ax.set_xlabel("x"); ax.set_ylabel("absolute error (log scale)")
    ax.set_title("Section 4b: y' = x+y,  Milne vs Heun error comparison")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    save_fig(fig, "section4b_error_comparison")

    print(f"Max |Milne error| = {actual_err.max():.3e}, "
          f"max |Heun error| = {err_heun.max():.3e}")


# ======================================================================
# SECTION 5: Comparative study
# ======================================================================

def section5():
    print("\n" + "=" * 70)
    print("SECTION 5: Comparative study")
    print("=" * 70)

    # --- 5a: y' = y(1-y), y(0)=0.5, RK4 vs Milne, h = 0.25 and 0.125 ---
    def f_a(x, y):
        return np.array([y[0] * (1.0 - y[0])])

    for h in (0.25, 0.125):
        x_rk4, Y_rk4 = rk4_system(f_a, [0.5], 0.0, 1.0, h)
        milne_res = milne_pc(f_a, [0.5], 0.0, 1.0, h)
        x_m, Y_m = milne_res["x"], milne_res["Y"]

        df = pd.DataFrame({
            "x_n": x_rk4,
            "y_RK4": Y_rk4[:, 0],
            "y_Milne": Y_m[:, 0],
            "|RK4-Milne|": np.abs(Y_rk4[:, 0] - Y_m[:, 0]),
        })
        save_table(df, f"section5a_h{h}")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x_rk4, Y_rk4[:, 0], "o-", label="RK4")
        ax.plot(x_m, Y_m[:, 0], "s--", label="Milne")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(f"Section 5a: y'=y(1-y), h={h}")
        ax.legend(); ax.grid(True, alpha=0.3)
        save_fig(fig, f"section5a_h{h}_plot")

    print("Note: comparison against the five Lab Sheet-I methods (Euler, "
          "Heun, Midpoint, etc.) requires re-running those methods from "
          "Lab Sheet-I on this same IVP; plug your Lab Sheet-I functions "
          "in here and add their columns to the section5a table above.")

    # --- 5b: y' = -y, y(0)=1, 0<=x<=20, RK4 vs Milne (h=0.1, then 0.05, 0.025) ---
    def f_b(x, y):
        return np.array([-y[0]])

    def exact_b(x):
        return np.exp(-x)

    h = 0.1
    x_rk4, Y_rk4 = rk4_system(f_b, [1.0], 0.0, 20.0, h)
    milne_res = milne_pc(f_b, [1.0], 0.0, 20.0, h)
    x_m, Y_m = milne_res["x"], milne_res["Y"]

    err_rk4 = np.abs(Y_rk4[:, 0] - exact_b(x_rk4))
    err_milne = np.abs(Y_m[:, 0] - exact_b(x_m))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(x_rk4, err_rk4 + 1e-20, label="RK4, h=0.1")
    ax.semilogy(x_m, err_milne + 1e-20, label="Milne, h=0.1")
    ax.set_xlabel("x"); ax.set_ylabel("absolute error (log scale)")
    ax.set_title("Section 5b: y'=-y,  absolute error, RK4 vs Milne")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    save_fig(fig, "section5b_error_h0.1")

    print(f"h=0.1  max |error| RK4={err_rk4.max():.3e}, "
          f"Milne={err_milne.max():.3e} (at x=20: {err_milne[-1]:.3e})")

    fig, ax = plt.subplots(figsize=(8, 5))
    for h2 in (0.1, 0.05, 0.025):
        milne_res_h = milne_pc(f_b, [1.0], 0.0, 20.0, h2)
        x_mh, Y_mh = milne_res_h["x"], milne_res_h["Y"]
        err_mh = np.abs(Y_mh[:, 0] - exact_b(x_mh))
        ax.semilogy(x_mh, err_mh + 1e-20, label=f"Milne, h={h2}")
        print(f"Milne h={h2}: max |error| = {err_mh.max():.3e} "
              f"(error at x=20: {err_mh[-1]:.3e})")
    ax.set_xlabel("x"); ax.set_ylabel("absolute error (log scale)")
    ax.set_title("Section 5b: Milne's method error for y'=-y at decreasing h")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    save_fig(fig, "section5b_milne_h_refinement")


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    section1()
    section2()
    section3()
    section4()
    section5()
    print(f"\nAll tables (.csv) and figures (.png) written to ./{OUTDIR}/")