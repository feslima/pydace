from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='pydace',
      version='0.0.1',
      description='Desing and Analysis of Computational Experiments toolbox.',
      author='Felipe Souza Lima',
      author_email='felipe.lima@eq.ufcg.edu.br',
      license='Apache License 2.0',
      packages=find_packages(exclude=['tests']),
      long_description=long_description,
      keywords="surrogate, metamodel",
      setup_requires=['numpy'],
      install_requires=['scipy']
      )
