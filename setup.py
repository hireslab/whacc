from setuptools import setup
from setuptools import find_packages


setup(name='whacc',
      version='0.0.11',
      description='largely automatic and customizable pipeline for creating a CNN to predict whiskers contacting objects',
      packages=find_packages(),
      author_email='phillip.maire@gmail.com',
      zip_safe=False,
      install_requires=[
       # "pyicu",
       "natsort==7.1.1"])
