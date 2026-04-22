from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension("mc_pricer_cpp", ["fast_engine.cpp"]),
]

setup(
    name="mc_pricer_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
