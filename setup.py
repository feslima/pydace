from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

PACKAGE_NAME = 'pydace'
VERSION = '0.1.2'

setup(name=PACKAGE_NAME,
      version=VERSION,
      description='Design and Analysis of Computational Experiments as python toolbox.',
      author='Felipe Souza Lima',
      author_email='felipe.lima@eq.ufcg.edu.br',
      url='https://github.com/feslima/pydace',
      license='Apache License 2.0',
      packages=[PACKAGE_NAME],
      package_dir={'': 'src'},
      long_description=long_description,
      long_description_content_type="text/markdown",
      keywords="surrogate, metamodel",
      install_requires=['numpy>=1.16', 'scipy>=1.2.0', 'pyDOE2'],
      python_requires='>=3.5',
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
          "Topic :: Scientific/Engineering",
      ]
      )
