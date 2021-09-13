# pip install -e .

from setuptools import setup, find_packages
from setuptools.extension import Extension
import os

setup(
    name='mri-normalization-tools',
    version='0.1',
    packages=find_packages(),
    url='',
    license='MIT',
    author='ML, Wong',
    description='',
    install_requires=[r.strip() for r in open('requirements.txt').readlines()]
)