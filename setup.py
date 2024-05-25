from setuptools import find_packages, setup

__version__ = '0.0.1'
URL = 'https://github.com/wayne155/pytorch_timeseries'

install_requires = [
    'tqdm',
]

full_requires =  [
    
]

benchmark_requires = [

]

test_requires = [
    'pytest',
]

dev_requires = test_requires + [
    'pre-commit',
]

setup(
    name='torch_timeseries',
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
        'benchmark': benchmark_requires,
        'test': test_requires,
        'dev': dev_requires,
    },
    package_dir={"torch_timeseries":"torch_timeseries"},
    packages=find_packages(exclude=['test']),
    include_package_data=True,
)
