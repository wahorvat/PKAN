"""Simple statistics utilities."""

from typing import Generic, Optional, TypeVar, Union

import attr
from jax import numpy as jnp
import numpy as np

T = TypeVar('T', float, np.ndarray, jnp.ndarray)


@attr.s(auto_attribs=True)
class WeightedStats(Generic[T]):
  mean: T
  variance: T


def exponentialy_weighted_stats(
    alpha: Union[float, T],
    observation: T,
    previous_stats: Optional[WeightedStats[T]] = None,
) -> WeightedStats[T]:
  """Returns the exponentially-weighted mean and variance.

  mu_t = alpha mu_{t-1} + (1-alpha) x_t

  and similarly for the variance.

  Args:
    alpha: weighting factor for previous observations.
    observation: new (t-th) value to include in the mean and variance.
    previous_stats: previous value of the mean and variance after (t-1)
      observations. Pass in None to indicate no prior observations have been
      made.
  """
  if previous_stats is None:
    return WeightedStats[T](mean=observation, variance=0.0 * observation)
  else:
    # See Incremental calculation of weighted mean and variance, Tony Finch,
    # https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
    diff = observation - previous_stats.mean
    incr = alpha * diff
    mean = previous_stats.mean + incr
    variance = (1 - alpha) * (previous_stats.variance + diff * incr)
    return WeightedStats[T](mean=mean, variance=variance)
