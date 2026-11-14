import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("./outputs", exist_ok=True)

def f(x, y):
    return y * (1 - y)

def exact(x):
    return np.exp(x) / (1 + np.exp(x))

def euler_method(f, x0, y0, x_end, h):
    N = int(round((x_end - x0) / h))
    x = x0 + h * np.arange(N + 1)
    y = np.zeros(N + 1)
    y[0] = y0
    for n in range(N):
        y[n+1] = y[n] + h * f(x[n], y[n])
    return x, y

def backward_euler(f, x0, y0, x_end, h):
    N = int(round((x_end - x0) / h))
    x = x0 + h * np.arange(N + 1)
    y = np.zeros(N + 1)
    y[0] = y0

    for n in range(N):
        # h*y_{n+1}^2 + (1-h)y_{n+1} - y_n = 0
        a = h
        b = 1 - h
        c = -y[n]
        disc = b**2 - 4*a*c
        y[n+1] = (-b + np.sqrt(disc)) / (2*a)

    return x, y

def improved_euler(f, x0, y0, x_end, h):
    N = int(round((x_end - x0) / h))
    x = x0 + h * np.arange(N + 1)
    y = np.zeros(N + 1)
    y[0] = y0

    for n in range(N):
        k1 = f(x[n], y[n])
        y_pred = y[n] + h*k1
        k2 = f(x[n] + h, y_pred)
        y[n+1] = y[n] + h/2*(k1 + k2)

    return x, y

def modified_euler(f, x0, y0, x_end, h):
    N = int(round((x_end - x0) / h))
    x = x0 + h * np.arange(N + 1)
    y = np.zeros(N + 1)
    y[0] = y0

    for n in range(N):
        k1 = f(x[n], y[n])
        y_mid = y[n] + h/2*k1
        k2 = f(x[n] + h/2, y_mid)
        y[n+1] = y[n] + h*k2

    return x, y

def rk4(f, x0, y0, x_end, h):
    N = int(round((x_end - x0) / h))
    x = x0 + h * np.arange(N + 1)
    y = np.zeros(N + 1)
    y[0] = y0

    for n in range(N):
        k1 = f(x[n], y[n])
        k2 = f(x[n] + h/2, y[n] + h*k1/2)
        k3 = f(x[n] + h/2, y[n] + h*k2/2)
        k4 = f(x[n] + h, y[n] + h*k3)
        y[n+1] = y[n] + h/6*(k1 + 2*k2 + 2*k3 + k4)

    return x, y

def milne(f, x0, y0, x_end, h):
    N = int(round((x_end - x0) / h))
    x = x0 + h * np.arange(N + 1)
    y = np.zeros(N + 1)
    y[0] = y0

    # RK4 starting values
    for n in range(3):
        k1 = f(x[n], y[n])
        k2 = f(x[n] + h/2, y[n] + h*k1/2)
        k3 = f(x[n] + h/2, y[n] + h*k2/2)
        k4 = f(x[n] + h, y[n] + h*k3)
        y[n+1] = y[n] + h/6*(k1 + 2*k2 + 2*k3 + k4)

    # Milne predictor-corrector
    for n in range(3, N):
        f_n = f(x[n], y[n])
        f_nm1 = f(x[n-1], y[n-1])
        f_nm2 = f(x[n-2], y[n-2])

        y_pred = y[n-3] + 4*h/3*(2*f_n - f_nm1 + 2*f_nm2)
        y_corr = y_pred

        for _ in range(100):
            f_np1 = f(x[n+1], y_corr)
            y_new = y[n-1] + h/3*(f_nm1 + 4*f_n + f_np1)

            if abs(y_new - y_corr) < 1e-10:
                break

            y_corr = y_new

        y[n+1] = y_new

    return x, y


# Generate solution + absolute-error plots for both requested step sizes
saved_files = []

for h in [0.25, 0.125]:

    x, y_euler = euler_method(f, 0, 0.5, 1, h)
    _, y_backward = backward_euler(f, 0, 0.5, 1, h)
    _, y_improved = improved_euler(f, 0, 0.5, 1, h)
    _, y_modified = modified_euler(f, 0, 0.5, 1, h)
    _, y_rk4 = rk4(f, 0, 0.5, 1, h)
    _, y_milne = milne(f, 0, 0.5, 1, h)

    y_ex = exact(x)

    solutions = {
        "Euler": y_euler,
        "Backward Euler": y_backward,
        "Improved Euler / Heun": y_improved,
        "Modified Euler / Midpoint": y_modified,
        "RK4": y_rk4,
        "Milne": y_milne,
    }

    # --------------------------------------------------------
    # Absolute error comparison
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))

    for name, y_num in solutions.items():
        error = np.abs(y_num - y_ex)
        plt.semilogy(
            x,
            error + 1e-16,
            marker="o",
            label=name
        )

    plt.xlabel("x")
    plt.ylabel("Absolute Error")
    plt.title(f"Part 5(a): Absolute Error Comparison (h = {h})")
    plt.legend()
    plt.grid(True, which="both")

    filename = f"./outputs/part5a_absolute_error_h{h}.png"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()

    saved_files.append(filename)

    # --------------------------------------------------------
    # Also print maximum absolute errors
    # --------------------------------------------------------
    print(f"\nh = {h}")
    print("-" * 45)

    for name, y_num in solutions.items():
        max_error = np.max(np.abs(y_num - y_ex))
        print(f"{name:<25}: {max_error:.8e}")


saved_files
