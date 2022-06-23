"""Packaging logic for EnAVP"""
import os
import sys
import setuptools

# https://setuptools.readthedocs.io/en/latest/

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setuptools.setup(
    long_description=read('README.md'),
)
