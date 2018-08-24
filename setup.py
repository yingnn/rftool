import os
import rftool
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


requirements = ['numpy>=1.13.3',
                'pandas>=0.21.0',
                'scipy>=1.0.0',
                'scikit-learn>=0.19.1',
                'chardet']

setup(name="rftool",
      packages=find_packages(),
      version=rftool.__version__,
      description='a random forest tool wrapper',
      author='yingnn',
      author_email='yingnn@live.cn',
      long_description=read('README.md'),
      keywords='random forest',
      url='',
      licence='',
      install_requires=requirements,
      include_package_data=True,
      scripts=['scripts/rf.py'])
