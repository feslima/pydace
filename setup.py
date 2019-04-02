from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='pydace',
      version='0.0.2',
      description='Desing and Analysis of Computational Experiments as python toolbox.',
      author='Felipe Souza Lima',
      author_email='felipe.lima@eq.ufcg.edu.br',
      url='https://github.com/feslima/pydace',
      license='Apache License 2.0',
      packages=find_packages(exclude=['tests']),
      long_description=long_description,
      keywords="surrogate, metamodel",
      setup_requires=['numpy'],
      install_requires=['scipy'],
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
          "Topic :: Scientific/Engineering",
      ]
      )
