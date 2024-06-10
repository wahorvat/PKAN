"""Tests for exponet.utils.units."""

from absl.testing import absltest
from exponet.utils import units
import numpy as np


class UnitsTest(absltest.TestCase):

  def test_angstrom2bohr(self):
    self.assertAlmostEqual(units.angstrom2bohr(2), 3.77945225091, places=10)

  def test_angstrom2bohr_numpy(self):
    x = np.random.uniform(size=(3,))
    x1 = units.angstrom2bohr(x)
    x2 = np.array([units.angstrom2bohr(v) for v in x])
    np.testing.assert_allclose(x1, x2)

  def test_bohr2angstrom(self):
    self.assertAlmostEqual(units.bohr2angstrom(2), 1.05835442134, places=10)

  def test_bohr2angstrom_numpy(self):
    x = np.random.uniform(size=(3,))
    x1 = units.bohr2angstrom(x)
    x2 = np.array([units.bohr2angstrom(v) for v in x])
    np.testing.assert_allclose(x1, x2)

  def test_angstrom_bohr_idempotent(self):
    x = np.random.uniform()
    x1 = units.bohr2angstrom(units.angstrom2bohr(x))
    self.assertAlmostEqual(x, x1, places=10)

  def test_bohr_angstrom_idempotent(self):
    x = np.random.uniform()
    x1 = units.angstrom2bohr(units.bohr2angstrom(x))
    self.assertAlmostEqual(x, x1, places=10)

  def test_hartree2kcal(self):
    self.assertAlmostEqual(units.hartree2kcal(2), 1255.018948, places=10)

  def test_hartree2kcal_numpy(self):
    x = np.random.uniform(size=(3,))
    x1 = units.hartree2kcal(x)
    x2 = np.array([units.hartree2kcal(v) for v in x])
    np.testing.assert_allclose(x1, x2)

  def test_kcal2hartree(self):
    self.assertAlmostEqual(units.kcal2hartree(2), 0.00318720287, places=10)

  def test_kcal2hartree_numpy(self):
    x = np.random.uniform(size=(3,))
    x1 = units.kcal2hartree(x)
    x2 = np.array([units.kcal2hartree(v) for v in x])
    np.testing.assert_allclose(x1, x2)

  def test_hartree_kcal_idempotent(self):
    x = np.random.uniform()
    x1 = units.kcal2hartree(units.hartree2kcal(x))
    self.assertAlmostEqual(x, x1, places=10)

  def test_kcal_hartree_idempotent(self):
    x = np.random.uniform()
    x1 = units.hartree2kcal(units.kcal2hartree(x))
    self.assertAlmostEqual(x, x1, places=10)


if __name__ == '__main__':
  absltest.main()
