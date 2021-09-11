# pip install -e .

from setuptools import setup
from setuptools.extension import Extension
import os

setup(
    name='mri-normalization-tools',
    version='0.1',
    packages=['mnts'],
    url='',
    license='MIT',
    author='ML, Wong',
    description='',
    install_requires=[r.strip() for r in open('requirements.txt').readlines()]
)