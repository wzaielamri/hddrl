import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="hddrl",
    py_modules=["hddrl"],
    version="1.0",
    description="Hierarchical Decentralized Deep Reinforcement Learning. Backbone code from Malte Schilling",
    author="Wadhah Zai El Amri",
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True
)
