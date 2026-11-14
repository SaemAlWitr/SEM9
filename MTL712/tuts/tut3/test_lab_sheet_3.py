"""Independent checks against exact solutions and the prescribed work counts."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

SOURCE = Path(__file__).with_name('lab_sheet_3.py')
spec = importlib.util.spec_from_file_location('lab3', SOURCE) if SOURCE.exists() else None
lab = importlib.util.module_from_spec(spec) if spec else None
if lab:
    spec.loader.exec_module(lab)


class SolverChecks(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(lab, 'The Python assignment solution has not been implemented yet.')

    def test_observed_orders_against_sine(self):
        for method, order in [('Numerov', 4), ('Stormer 2', 2), ('Stormer 3', 3), ('Stormer 4', 4), ('RK4', 4)]:
            errors = []
            for h in [0.05, 0.025]:
                r = lab.solve_harmonic(method, 0.0, 1.0, 6.0, h)
                errors.append(max(abs(r.y - np.sin(r.x))))
            observed = np.log2(errors[0] / errors[1])
            self.assertLess(abs(observed - order), 0.25, (method, observed))

    def test_nonlinear_numerov_residual_and_iteration_stop(self):
        r = lab.numerov_nonlinear(lambda x, y: 6*y*y, 0, 1, -2, 1, 0.1)
        f = 6*r.y*r.y
        residual = r.y[2:] - 2*r.y[1:-1] + r.y[:-2] - 0.1**2/12*(f[2:] + 10*f[1:-1] + f[:-2])
        self.assertLess(max(abs(residual)), 2e-12)
        self.assertTrue(np.all(r.last_changes[2:] < 1e-12))
        self.assertTrue(np.all(r.iterations[2:] > 0))
        self.assertLess(max(abs(r.y - 1/(1+r.x)**2)), 0.001)

    def test_actual_acceleration_evaluation_counts(self):
        expected = {'Numerov': 204, 'Stormer 2': 203, 'Stormer 3': 208, 'Stormer 4': 212, 'RK4': 800}
        for method, count in expected.items():
            r = lab.solve_harmonic(method, 1, 0, 20, 0.1)
            self.assertEqual(r.evaluations, count, method)
            self.assertEqual(len(r.x), 201)
            self.assertAlmostEqual(r.x[-1], 20)

    def test_constant_acceleration_is_exact(self):
        for order in [2, 3, 4]:
            r = lab.stormer(lambda x, y: 2.0, 0, 1, 3, 2, 0.1, order)
            np.testing.assert_allclose(r.y, 1 + 3*r.x + r.x**2, atol=2e-13, rtol=0)

    def test_pendulum_energy_and_terminal_central_difference(self):
        r, velocity, energy, native_energy = lab.pendulum_result('RK4', 0.05)
        self.assertEqual(len(r.x), 1001)
        self.assertAlmostEqual(r.x[-1], 50)
        self.assertEqual(velocity[0], 0)
        self.assertLess(max(abs(native_energy + np.cos(1))), 1e-6)
        self.assertLess(max(abs(energy + np.cos(1))), 0.001)


if __name__ == '__main__':
    unittest.main()
