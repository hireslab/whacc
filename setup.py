from setuptools import setup
from setuptools import find_packages
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='whacc',
      version='0.1.15',
      author="Phillip Maire",
      license='MIT',
      description='Automatic and customizable pipeline for creating a CNN + GBM to predict whiskers contacting objects',
      packages=find_packages(),
      author_email='phillip.maire@gmail.com',
      long_description=long_description,
      long_description_content_type="text/markdown",
      zip_safe=False,
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
      python_requires='>=3.6',
      install_requires=[
       # "pyicu",
       "natsort==7.1.1"])

# package_data={
    #     'whacc': ['datasets/*.h5',
    #               'model_checkpoints/*',
    #               'model_checkpoints/model_201008/*',
    #               'model_checkpoints/model_201008/variables/*',
    #               'model_checkpoints/model_201008/assets/*'],
    # },
# include_package_data=True,



