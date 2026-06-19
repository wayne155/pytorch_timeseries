from setuptools import find_packages, setup

__version__ = '0.2.30'
URL = 'https://github.com/wayne155/pytorch_timeseries'

install_requires = [
    'tqdm',
    'numpy',
    'pandas',
    'einops',
    'torch',
    'sktime>=0.29.0',
    'torchmetrics>=1.1.1',
    'fire>=0.5.0',
    'PyYAML',
    'scikit-learn',
    'prettytable',
]

full_requires = [
    "torch_scatter",
]

irregular_requires = [
    "torchdiffeq>=0.2.3",
    "torchcde>=0.2.5",
    "torch_geometric>=2.0.0",
]

benchmark_requires = []

test_requires = [
    'pytest',
]

dev_requires = test_requires + [
    'pre-commit',
]

setup(
    name='torch-timeseries',
    version=__version__,
    description='Timeseries Learning Library for PyTorch',
    author='Weiwei Ye',
    author_email='wwye155@gmail.com',
    url=URL,
    download_url=f'{URL}/archive/{__version__}.tar.gz',
    keywords=[
        'deep-learning',
        'pytorch',
        'timeseries-deep-learning',
        'neural-networks',
    ],
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require={
        'full': full_requires,
        'irregular': irregular_requires,
        'benchmark': benchmark_requires,
        'test': test_requires,
        'dev': dev_requires,
    },
    entry_points={
        'console_scripts': [
            'pytexp = torch_timeseries:exp',
        ]
    },
    package_dir={"torch_timeseries":"torch_timeseries"},
    packages=find_packages(exclude=['test']),
    include_package_data=True,
)
