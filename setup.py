from setuptools import find_packages
from distutils.core import setup

setup(
  name = "diffusion_policy",
  version="1.0.0",
  packages = find_packages(),
  install_requires=[ "diffusers", "dill", "torch", "zarr", "hydra.core"  ],
)
