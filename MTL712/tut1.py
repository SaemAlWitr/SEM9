import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# NUMERICAL METHODS IMPLEMENTATION
# ==========================================

def euler_method(f, x0, y0, x_end, h):
    n_steps = int(np.round((x_end - x0) / h))
    x = np.linspace(x0, x_end, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0
    for i in range(n_steps):
        y[i+1] = y[i] + h * f(x[i], y[i])
    return x, y

def backward_euler_q2(x0, y0, x_end, h):
    # Explicit formula specific to Q2: y' = -50y
    n_steps = int(np.round((x_end - x0) / h))
    x = np.linspace(x0, x_end, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0
    for i in range(n_steps):
        y[i+1] = y[i] / (1 + 50 * h)
    return x, y

def improved_euler(f, x0, y0, x_end, h):
    n_steps = int(np.round((x_end - x0) / h))
    x = np.linspace(x0, x_end, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0
    for i in range(n_steps):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h, y[i] + h * k1)
        y[i+1] = y[i] + (h / 2) * (k1 + k2)
    return x, y

def modified_euler(f, x0, y0, x_end, h):
    n_steps = int(np.round((x_end - x0) / h))
    x = np.linspace(x0, x_end, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0
    for i in range(n_steps):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h/2, y[i] + (h / 2) * k1)
        y[i+1] = y[i] + h * k2
    return x, y

def rk4(f, x0, y0, x_end, h):
    n_steps = int(np.round((x_end - x0) / h))
    x = np.linspace(x0, x_end, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0
    for i in range(n_steps):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h/2, y[i] + h/2 * k1)
        k3 = f(x[i] + h/2, y[i] + h/2 * k2)
        k4 = f(x[i] + h, y[i] + h * k3)
        y[i+1] = y[i] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return x, y

# ==========================================
# PROBLEM SOLVERS & PLOTTING
# ==========================================

# Q1: Euler's Method
def solve_q1():
    f = lambda x, y: y - x**2 + 1
    exact_f = lambda x: (x + 1)**2 - 0.5 * np.exp(x)
    x_num, y_num = euler_method(f, 0, 0.5, 1, 0.1)
    
    x_exact = np.linspace(0, 1, 100)
    y_exact = exact_f(x_exact)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_exact, y_exact, 'c-', label="Exact Solution")
    plt.plot(x_num, y_num, 'r--', label="Euler Method (h=0.1)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Q1: Euler's Method vs Exact Solution")
    plt.legend()
    plt.grid(True)
    plt.show()

# Q2: Backward Euler's Method
def solve_q2():
    exact_f = lambda x: np.exp(-50 * x)
    x_num, y_num = backward_euler_q2(0, 1, 1, 0.1)
    
    x_exact = np.linspace(0, 1, 200)
    y_exact = exact_f(x_exact)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_exact, y_exact, 'k-', label="Exact Solution")
    plt.plot(x_num, y_num, 'ro--', label="Backward Euler (h=0.1)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Q2: Backward Euler vs Exact Solution")
    plt.legend()
    plt.grid(True)
    plt.show()

# Q3: Improved Euler's Method
def solve_q3():
    f = lambda x, y: x + y
    exact_f = lambda x: 2 * np.exp(x) - x - 1
    x_num, y_num = improved_euler(f, 0, 1, 1, 0.1)
    
    x_exact = np.linspace(0, 1, 100)
    y_exact = exact_f(x_exact)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_exact, y_exact, 'k-', label="Exact Solution")
    plt.plot(x_num, y_num, 'go--', label="Improved Euler (h=0.1)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Q3: Improved Euler Method vs Exact Solution")
    plt.legend()
    plt.grid(True)
    plt.show()

# Q4: Modified Euler's Method
def solve_q4():
    f = lambda x, y: x * y
    exact_f = lambda x: np.exp(x**2 / 2)
    
    x_num1, y_num1 = modified_euler(f, 0, 1, 1, 0.1)
    x_num2, y_num2 = modified_euler(f, 0, 1, 1, 0.05)
    
    x_exact = np.linspace(0, 1, 100)
    y_exact = exact_f(x_exact)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_exact, y_exact, 'k-', label="Exact Solution")
    plt.plot(x_num1, y_num1, 'bo--', label="Modified Euler (h=0.1)")
    plt.plot(x_num2, y_num2, 'ro--', label="Modified Euler (h=0.05)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Q4: Modified Euler Method vs Exact Solution")
    plt.legend()
    plt.grid(True)
    plt.show()

# Q5: Classical Fourth-Order Runge-Kutta
def solve_q5():
    f = lambda x, y: np.cos(x) - y
    exact_f = lambda x: 0.5 * (np.sin(x) + np.cos(x)) + 0.5 * np.exp(-x)
    x_num, y_num = rk4(f, 0, 1, 2, 0.2)
    
    x_exact = np.linspace(0, 2, 100)
    y_exact = exact_f(x_exact)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_exact, y_exact, 'k-', label="Exact Solution")
    plt.plot(x_num, y_num, 'mo--', label="RK4 (h=0.2)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Q5: RK4 Method vs Exact Solution")
    plt.legend()
    plt.grid(True)
    plt.show()

# Q6: Comparative Study
def solve_q6():
    f = lambda x, y: y * (1 - y)
    
    def run_comparison(h):
        print(f"\n--- Comparative Study Table for h = {h} ---")
        x_e, y_e = euler_method(f, 0, 0.5, 1, h)
        _, y_be = backward_euler_q6(f, 0, 0.5, 1, h) 
        _, y_ie = improved_euler(f, 0, 0.5, 1, h)
        _, y_me = modified_euler(f, 0, 0.5, 1, h)
        _, y_rk = rk4(f, 0, 0.5, 1, h)
        
        print(f"{'x_n':<6} | {'Euler':<10} | {'Back Euler':<10} | {'Imp Euler':<10} | {'Mod Euler':<10} | {'RK4':<10}")
        print("-" * 75)
        for i in range(len(x_e)):
            print(f"{x_e[i]:<6.3f} | {y_e[i]:<10.6f} | {y_be[i]:<10.6f} | {y_ie[i]:<10.6f} | {y_me[i]:<10.6f} | {y_rk[i]:<10.6f}")
            
        plt.figure(figsize=(10, 6))
        plt.plot(x_e, y_e, 'o-', label="Euler")
        plt.plot(x_e, y_be, 's-', label="Backward Euler")
        plt.plot(x_e, y_ie, '^-', label="Improved Euler")
        plt.plot(x_e, y_me, 'd-', label="Modified Euler")
        plt.plot(x_e, y_rk, 'x-', label="RK4")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Q6: Method Comparison (h={h})")
        plt.legend()
        plt.grid(True)
        plt.show()
    run_comparison(.25)
    run_comparison(.125)

def backward_euler_q6(f, x0, y0, x_end, h):
    # Newton-Raphson to solve implicit equation for nonlinear Q6: y_{n+1} - y_n - h*y_{n+1}(1 - y_{n+1}) = 0
    n_steps = int(np.round((x_end - x0) / h))
    x = np.linspace(x0, x_end, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0
    for i in range(n_steps):
        y_next = y[i] # Initial guess
        for _ in range(10): # 10 iterations max
            F = y_next - y[i] - h * y_next * (1 - y_next)
            dF = 1 - h * (1 - 2 * y_next)
            y_next = y_next - F / dF
        y[i+1] = y_next
    return x, y

if __name__ == "__main__":
    solve_q1()
    solve_q2()
    solve_q3()
    solve_q4()
    solve_q5()
    solve_q6()