from setuptools import setup, find_packages


setup(
    name='PS_Toolkit',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=False,
    zip_safe=False,
    install_requires=[
        'joblib',
        'matplotlib',
        'numpy',
        'pandas>=0.19',
        'pymc3',
        'scipy',
        'seaborn',
        'sklearn'
    ]
)
