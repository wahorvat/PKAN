"""Setup for pip package."""

import unittest

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['absl-py',
    'attrs',
    'chex',
    'h5py',
    'folx'
    'jax',
    'jaxlib',
    'kfac-jax'
    'ml-collections',
    'optax',
    'numpy',
    'pandas',
    'pyscf',
    'pyblock',
    'scipy',
    'typing_extensions',                     
]

def ferminet_test_suite():
  test_loader = unittest.TestLoader()
  test_suite = test_loader.discover('exponet/tests', pattern='*_test.py')
  return test_suite


setup(
    name='exponet',
    version='0.1',
    description=('A library to train networks to represent ground '
                 'state wavefunctions of fermionic systems'),
    url='https://github.com/wahorvat/exponet',
    author='Will Horvat',
    # Contained modules and scripts.
    scripts=['bin/exponet'],
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require={'testing': ['pylint', 'pytest', 'pytype']},
    platforms=['any'],
    license='Apache 2.0',
    test_suite='setup.exponet_test_suite',
)
