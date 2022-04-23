from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='whacc',
      version='0.1.31',
      author="Phillip Maire",
      license='MIT',
      description='Automatic and customizable pipeline for creating a CNN + light GBM model to predict whiskers contacting objects',
      packages=find_packages(),
      author_email='phillip.maire@gmail.com',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/hireslab/whacc",
      zip_safe=False,
      classifiers=["Programming Language :: Python :: 3",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: OS Independent"],
      python_requires='>=3.6',
      install_requires=["natsort==7.1.1"],
      package_data={'whacc': ['whacc_data/*',
                              'whacc_data/feature_data/*',
                              'whacc_data/final_model/final_resnet50V2_full_model/variables/*',
                              'whacc_data/final_model/final_resnet50V2_full_model/assets/*',
                              'whacc_data/final_model/final_resnet50V2_full_model/*',
                              'whacc_data/final_model/*']},
      include_package_data=True
      )

