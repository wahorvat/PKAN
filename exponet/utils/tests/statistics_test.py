"""Tests for exponet.utils.statistics."""

from absl.testing import absltest
from absl.testing import parameterized
from exponet.utils import statistics
import numpy as np
import pandas as pd


class StatisticsTest(parameterized.TestCase):

  @parameterized.parameters(0.1, 0.2, 0.5)
  def test_exponentially_weighted_stats(self, alpha):
    # Generate some data which (loosely) mimics converging simulation.
    n = 10_000
    data = (0.1 * np.random.uniform(size=n) + np.exp(-np.arange(n) / 100) +
            (1 + 0.05 * np.random.normal(size=n)))

    stats = []
    for x in data:
      stat = statistics.exponentialy_weighted_stats(
          alpha=alpha,
          observation=x,
          previous_stats=stats[-1] if stats else None)
      stats.append(stat)

    # Exponentially weighted algorithm is equivalent to that in pandas without
    # bias correction or adjusting for the initial iterations:
    ewm = pd.Series(data).ewm(adjust=False, alpha=alpha)
    expected_mean = ewm.mean()
    expected_variance = ewm.var(bias=True)
    with self.subTest('Check mean'):
      np.testing.assert_allclose([s.mean for s in stats], expected_mean)
    with self.subTest('Check variance'):
      np.testing.assert_allclose([s.variance for s in stats], expected_variance)


if __name__ == '__main__':
  absltest.main()
