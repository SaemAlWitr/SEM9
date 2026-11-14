from pathlib import Path
import os
import csv
import json
import math

os.environ.setdefault('MPLCONFIGDIR', 'tmp/mtl712-lab3-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'outputs'
METHODS = ['Numerov', 'Stormer 2', 'Stormer 4', 'RK4']


class Result:
    def __init__(self, x, y, evaluations, velocity=None, iterations=None, last_changes=None):
        self.x, self.y, self.evaluations = x, y, evaluations
        self.velocity = velocity
        self.iterations, self.last_changes = iterations, last_changes


class CountedForce:
    """Count actual calls to the scalar acceleration f(x,y)."""
    def __init__(self, f):
        self.f, self.calls = f, 0

    def __call__(self, x, y):
        self.calls += 1
        return self.f(x, y)


def mesh(x0, xf, h):
    if h <= 0 or xf <= x0:
        raise ValueError('Require h > 0 and xf > x0.')
    n = round((xf-x0)/h)
    if not math.isclose(n*h, xf-x0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('The step must divide the interval.')
    return x0 + h*np.arange(n+1)


def rk4_step(f, x, y, v, h):
    k1y, k1v = v, f(x, y)
    k2y, k2v = v+h*k1v/2, f(x+h/2, y+h*k1y/2)
    k3y, k3v = v+h*k2v/2, f(x+h/2, y+h*k2y/2)
    k4y, k4v = v+h*k3v, f(x+h, y+h*k3y)
    return (y+h*(k1y+2*k2y+2*k3y+k4y)/6,
            v+h*(k1v+2*k2v+2*k3v+k4v)/6)


def starting_values(f, x, y0, v0, h, steps):
    y = np.zeros(len(x))
    y[0], v = y0, v0
    for n in range(steps):
        y[n+1], v = rk4_step(f, x[n], y[n], v, h)
    return y


def rk4(f, x0, y0, v0, xf, h):
    x, force = mesh(x0, xf, h), CountedForce(f)
    y, v = np.zeros(len(x)), np.zeros(len(x))
    y[0], v[0] = y0, v0
    for n in range(len(x)-1):
        y[n+1], v[n+1] = rk4_step(force, x[n], y[n], v[n], h)
    return Result(x, y, force.calls, v)


def numerov_harmonic(y0, v0, xf, h):
    """Explicit linear Numerov, with f_n=-y_n.

    Equivalent recurrence: (1+h^2/12)y[n+1] =
    (2-5h^2/6)y[n] - (1+h^2/12)y[n-1].
    """
    x, force = mesh(0, xf, h), CountedForce(lambda x, y: -y)
    y = starting_values(force, x, y0, v0, h, 1)
    previous, current = force(x[0], y[0]), force(x[1], y[1])
    for n in range(1, len(x)-1):
        y[n+1] = (2*y[n]-y[n-1]+h*h*(10*current+previous)/12)/(1+h*h/12)
        if n < len(x)-2:
            previous, current = current, force(x[n+1], y[n+1])
    return Result(x, y, force.calls)


def numerov_nonlinear(f, x0, y0, v0, xf, h, tol=1e-12, max_iter=10000):
    """Fixed-point iteration; stop only when successive values differ < tol."""
    x, force = mesh(x0, xf, h), CountedForce(f)
    y = starting_values(force, x, y0, v0, h, 1)
    iterations, changes = np.zeros(len(x), dtype=int), np.zeros(len(x))
    previous, current = force(x[0], y[0]), force(x[1], y[1])
    for n in range(1, len(x)-1):
        base = 2*y[n]-y[n-1]+h*h*(10*current+previous)/12
        guess = 2*y[n]-y[n-1]+h*h*current
        for k in range(1, max_iter+1):
            updated = base+h*h*force(x[n+1], guess)/12
            if not math.isfinite(updated):
                raise RuntimeError('Nonlinear iteration diverged.')
            change = abs(updated-guess)
            guess = updated
            if change < tol:
                break
        else:
            raise RuntimeError('Nonlinear Numerov failed to converge.')
        y[n+1], iterations[n+1], changes[n+1] = guess, k, change
        if n < len(x)-2:
            previous, current = current, force(x[n+1], y[n+1])
    return Result(x, y, force.calls, iterations=iterations, last_changes=changes)


def stormer(f, x0, y0, v0, xf, h, order):
    """Truncate after 1, 2 or 3 NONZERO backward-difference terms.
    y_n+1 = 2y_n-y_n-1 + S_i
    S2: f_n
    S3: f_n + (f_n-2f[n-1]+f[n-2])/12
    S4: f_n + nabla^2 f_n/12 + nabla^3 f_n/12
    The coefficient of nabla f_n is zero.
    """
    x, force = mesh(x0, xf, h), CountedForce(f)
    starters = order-1
    if len(x)-1 < starters:
        raise ValueError('Not enough mesh points for RK4 starting values.')
    y = starting_values(force, x, y0, v0, h, starters)
    history = np.zeros(len(x))
    # S2 only uses f_1,...,f_(N-1); higher orders need f_0 as well.
    for n in range(0 if order > 2 else 1, starters):
        history[n] = force(x[n], y[n])
    for n in range(starters, len(x)-1):
        history[n] = force(x[n], y[n])
        acceleration = history[n]
        if order >= 3:
            acceleration += (history[n]-2*history[n-1]+history[n-2])/12
        if order == 4:
            acceleration += (history[n]-3*history[n-1]+3*history[n-2]-history[n-3])/12
        y[n+1] = 2*y[n]-y[n-1]+h*h*acceleration
    return Result(x, y, force.calls)


def solve_harmonic(method, y0, v0, xf, h):
    if method == 'Numerov':
        return numerov_harmonic(y0, v0, xf, h)
    if method == 'RK4':
        return rk4(lambda x, y: -y, 0, y0, v0, xf, h)
    return stormer(lambda x, y: -y, 0, y0, v0, xf, h, int(method[-1]))


def pendulum_result(method, h):
    # One extra point supplies a central difference at x=50, too.
    f = lambda x, y: -math.sin(y)
    r = rk4(f, 0, 1, 0, 50+h, h) if method == 'RK4' else stormer(f, 0, 1, 0, 50+h, h, int(method[-1]))
    velocity = np.zeros(len(r.y)-1)
    velocity[1:] = (r.y[2:]-r.y[:-2])/(2*h)
    energy = velocity**2/2-np.cos(r.y[:-1])
    native_energy = None if r.velocity is None else r.velocity[:-1]**2/2-np.cos(r.y[:-1])
    trimmed = Result(r.x[:-1], r.y[:-1], r.evaluations,
                     None if r.velocity is None else r.velocity[:-1])
    return trimmed, velocity, energy, native_energy


def save_table(name, headings, columns):
    with (OUT/name).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(headings)
        writer.writerows(zip(*columns))


def save_rows(name, headings, rows):
    with (OUT/name).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(headings)
        writer.writerows(rows)


def finish_plot(name, xlabel, ylabel, title):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which='both', alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT/name, dpi=180)
    plt.close()


def convergence_rows(results, exact):
    rows, previous = [], None
    for h, r in results:
        error = float(max(abs(r.y-exact(r.x))))
        ratio = None if previous is None else previous/error
        rows.append([h, error, ratio, None if ratio is None else math.log2(ratio)])
        previous = error
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.figsize': (8, 4.6), 'font.size': 10})
    report = {}

    # 1: Linear Numerov, sine solution.
    q1 = []
    for h in [0.1, 0.05, 0.025]:
        r = numerov_harmonic(0, 1, 6, h)
        q1.append((h, r))
        exact, error = np.sin(r.x), r.y-np.sin(r.x)
        save_table(f'q1_h{h:g}.csv', ['x', 'Numerov', 'exact', 'signed_error', 'absolute_error'], [r.x, r.y, exact, error, abs(error)])
        plt.figure()
        plt.plot(r.x, exact, 'k-', label='Exact: sin(x)')
        plt.plot(r.x, r.y, 'o', ms=2.5, label=f'Numerov, h={h:g}')
        finish_plot(f'q1_h{h:g}.png', 'x', 'y', '1. Linear Numerov')
    report['q1'] = convergence_rows(q1, np.sin)
    save_rows('q1_convergence.csv', ['h', 'max_abs_error', 'error_ratio', 'observed_order'], report['q1'])

    # 2: Nonlinear Numerov, rational exact solution.
    q2 = []
    for h in [0.1, 0.05]:
        r = numerov_nonlinear(lambda x, y: 6*y*y, 0, 1, -2, 1, h)
        exact = 1/(1+r.x)**2
        error = r.y-exact
        save_table(f'q2_h{h:g}.csv', ['x', 'Numerov', 'exact', 'iterations', 'signed_error', 'absolute_error', 'last_iterate_change'], [r.x, r.y, exact, r.iterations, error, abs(error), r.last_changes])
        q2.append([h, float(max(abs(error))), int(min(r.iterations[2:])), int(max(r.iterations[2:])), float(np.mean(r.iterations[2:]))])
        plt.figure()
        plt.plot(r.x, exact, 'k-', label='Exact: 1/(1+x)^2')
        plt.plot(r.x, r.y, 'o', ms=3, label=f'Numerov, h={h:g}')
        finish_plot(f'q2_h{h:g}.png', 'x', 'y', '2. Nonlinear Numerov')
    report['q2'] = q2
    save_rows('q2_summary.csv', ['h', 'max_abs_error', 'min_iterations', 'max_iterations', 'mean_iterations'], q2)

    # 3: Störmer convergence study.
    report['q3'] = {}
    plt.figure()
    for order in [2, 3, 4]:
        results = [(h, solve_harmonic(f'Stormer {order}', 0, 1, 6, h)) for h in [0.1, 0.05, 0.025, 0.0125]]
        rows = convergence_rows(results, np.sin)
        report['q3'][str(order)] = rows
        save_rows(f'q3_order{order}_convergence.csv', ['h', 'max_abs_error', 'error_ratio', 'observed_order'], rows)
        for h, r in results:
            save_table(f'q3_order{order}_h{h:g}.csv', ['x', 'y', 'exact', 'absolute_error'], [r.x, r.y, np.sin(r.x), abs(r.y-np.sin(r.x))])
        plt.loglog([row[0] for row in rows], [row[1] for row in rows], 'o-', label=f'Stormer order {order}')
    finish_plot('q3_convergence.png', 'Step size h', 'Maximum absolute error', '3. Stormer convergence')

    # 4: Nonlinear pendulum. Central velocities for every method, including RK4.
    q4, pendulum = [], {}
    for method in ['Stormer 2', 'Stormer 4', 'RK4']:
        r, velocity, energy, native = pendulum_result(method, 0.05)
        pendulum[method] = (r, velocity, energy, native)
        drift = abs(energy+math.cos(1))
        columns, headings = [r.x, r.y, velocity, energy, drift], ['x', 'y', 'central_velocity', 'central_energy', 'abs_energy_error']
        if native is not None:
            headings += ['RK4_velocity', 'RK4_native_energy', 'RK4_native_abs_energy_error']
            columns += [r.velocity, native, abs(native+math.cos(1))]
        save_table(f'q4_{method.replace(" ", "_")}.csv', headings, columns)
        q4.append([method, float(max(drift)), float(drift[-1]), None if native is None else float(max(abs(native+math.cos(1))))])
    report['q4'] = q4
    save_rows('q4_summary.csv', ['method', 'max_central_energy_error', 'final_central_energy_error', 'max_native_RK4_energy_error'], q4)
    for kind in ['solution', 'phase', 'energy']:
        plt.figure()
        for method, (r, velocity, energy, native) in pendulum.items():
            if kind == 'solution':
                plt.plot(r.x, r.y, label=method)
            elif kind == 'phase':
                plt.plot(r.y, velocity, label=method)
            else:
                plt.plot(r.x, abs(energy+math.cos(1)), label=method)
        if kind == 'solution':
            finish_plot('q4_solution.png', 'x', 'y (radians)', '4. Pendulum, h=0.05')
        elif kind == 'phase':
            plt.axis('equal')
            finish_plot('q4_phase.png', 'y (radians)', "Central approximation to y'", '4. Pendulum phase trajectories')
        else:
            finish_plot('q4_energy.png', 'x', '|E(x)-E(0)|', '4. Pendulum energy using central velocities')
    plt.figure()
    r, velocity, energy, native = pendulum['RK4']
    plt.semilogy(r.x[1:], np.maximum(abs(energy[1:]+math.cos(1)), 1e-18), label='RK4, central velocity')
    plt.semilogy(r.x[1:], np.maximum(abs(native[1:]+math.cos(1)), 1e-18), label='RK4, native velocity')
    finish_plot('q4_RK4_energy_diagnostic.png', 'x', '|E(x)-E(0)|', '4. Effect of second-order velocity postprocessing')

    # 5: Same-step comparison and actual acceleration-call counts.
    report['q5'] = []
    for h in [0.1, 0.05]:
        results = {method: solve_harmonic(method, 1, 0, 20, h) for method in METHODS}
        x = results['Numerov'].x
        save_table(f'q5_h{h:g}.csv', ['x', 'exact']+METHODS+[m+'_abs_error' for m in METHODS], [x, np.cos(x)]+[results[m].y for m in METHODS]+[abs(results[m].y-np.cos(x)) for m in METHODS])
        for kind in ['solution', 'error']:
            plt.figure()
            if kind == 'solution':
                plt.plot(x, np.cos(x), 'k--', linewidth=1, label='Exact: cos(x)')
            for method, r in results.items():
                if kind == 'solution':
                    plt.plot(x, r.y, label=method)
                else:
                    # Exclude x=0 and floor zeros only for logarithmic display.
                    plt.semilogy(x[1:], np.maximum(abs(r.y[1:]-np.cos(x[1:])), 1e-18), label=method)
            finish_plot(f'q5_{kind}_h{h:g}.png', 'x', 'y' if kind == 'solution' else 'Absolute error', f'5. Four-method {kind} comparison, h={h:g}')
        for method, r in results.items():
            report['q5'].append([h, method, float(max(abs(r.y-np.cos(x)))), r.evaluations])
    save_rows('q5_summary.csv', ['h', 'method', 'max_abs_error', 'f_evaluations'], report['q5'])
    report['work'] = []
    plt.figure()
    for method in METHODS:
        rows = []
        overhead = {'Numerov': 4, 'Stormer 2': 3, 'Stormer 4': 12}.get(method)
        for budget in [100, 200, 400, 800, 1600]:
            n = budget//4 if method == 'RK4' else budget-overhead
            r = solve_harmonic(method, 1, 0, 20, 20/n)
            row = [budget, method, n, 20/n, r.evaluations, float(max(abs(r.y-np.cos(r.x))))]
            rows.append(row)
            report['work'].append(row)
        plt.loglog([row[4] for row in rows], [row[5] for row in rows], 'o-', label=method)
    finish_plot('q5_equal_work.png', 'Actual evaluations of f (including RK4 starters)', 'Maximum absolute error', '5. Accuracy at equal acceleration-evaluation budgets')
    save_rows('q5_equal_work.csv', ['budget', 'method', 'steps', 'h', 'actual_evaluations', 'max_abs_error'], report['work'])
    (OUT/'report_data.json').write_text(json.dumps(report, indent=2))
    print('Completed all five problems. Tables and plots:', OUT)
    for name in ['q1', 'q2', 'q4', 'q5']:
        print(name, report[name])
    print('Q3 convergence:', report['q3'])


if __name__ == '__main__':
    main()
