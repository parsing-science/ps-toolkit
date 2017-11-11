from setuptools import setup, find_packages


setup(
    name='PS_Toolkit',
    version='2.0.1',
    packages=find_packages(),
    include_package_data=False,
    zip_safe=False,
    install_requires=[
        'joblib',
        'matplotlib',
        'numpy',
        'pandas>=0.19',
        'pymc3>=3.2',
        'scipy',
        'seaborn',
        'sklearn'
    ]
)
