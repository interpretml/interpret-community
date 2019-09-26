# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Setup file for interpret-community package."""
from setuptools import setup, find_packages
import os
import shutil

_major = '0.1'
_minor = '0.0'

shutil.copyfile('../LICENSE', 'LICENSE.txt')

if os.path.exists('../major.version'):
    with open('../major.version', 'rt') as bf:
        _major = str(bf.read()).strip()

if os.path.exists('../minor.version'):
    with open('../minor.version', 'rt') as bf:
        _minor = str(bf.read()).strip()

VERSION = '{}.{}'.format(_major, _minor)
SELFVERSION = VERSION
if os.path.exists('patch.version'):
    with open('patch.version', 'rt') as bf:
        _patch = str(bf.read()).strip()
        SELFVERSION = '{}.{}'.format(VERSION, _patch)


CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: MacOS',
    'Operating System :: POSIX :: Linux'
]

DEPENDENCIES = [
    'numpy',
    'pandas',
    'scipy',
    'scikit-learn',
    'interpret',
    'shap>=0.20.0, <=0.29.3'
]

EXTRAS = {
    'sample': [
        'hdbscan'
    ],
    'deep': [
        'tensorflow'
    ],
    'mimic': [
        'lightgbm'
    ]
}

with open('README.md', 'r', encoding='utf-8') as f:
    README = f.read()
with open('HISTORY.rst', 'r', encoding='utf-8') as f:
    HISTORY = f.read()

setup(
    name='interpret-community',

    version=SELFVERSION,

    description='Microsoft Interpret Extensions SDK for Python',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Microsoft Corp',
    author_email='ilmat@microsoft.com',
    license='MIT License',
    url='https://docs.microsoft.com/en-us/azure/machine-learning/service/',

    classifiers=CLASSIFIERS,

    packages=find_packages(exclude=["*.tests"]),

    install_requires=DEPENDENCIES,

    entry_points={
        "interpret_ext_blackbox": [
            "TabularExplainer = interpret_community:TabularExplainer",
            "KernelExplainer = interpret_community.shap:KernelExplainer",
            "MimicExplainer = interpret_community.mimic:MimicExplainer",
            "PFIExplainer = interpret_community.permutation:PFIExplainer"
        ],
        "interpret_ext_greybox": [
            "LinearExplainer = interpret_community.shap:LinearExplainer",
            "DeepExplainer = interpret_community.shap:DeepExplainer",
            "TreeExplainer = interpret_community.shap:TreeExplainer"
        ]
    },

    extras_require=EXTRAS
)
