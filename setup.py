from importlib.metadata import entry_points
from setuptools import setup, find_packages

setup(
    name = 'frank',
    version = '0.1.0',
    packages = find_packages(),
    py_modules=['rankings'],
    install_requires = ['pandas', 'click', 'editdistance', 'tabulate', 'rich'],
    entry_points = {
        'console_scripts' : [
            'frank = rankings:rankings',
        ]
    }
)