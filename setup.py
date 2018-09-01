from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='findkit',
    version='0.1',
    description='A Python library for content-based information retrieval',
    url='https://github.com/lambdaofgod/findkit',
    author='Jakub Bartczuk',
    py_modules=find_packages(),
    install_requires=requirements
)
