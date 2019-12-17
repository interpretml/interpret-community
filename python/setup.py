# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Setup file for interpret-community package."""
from setuptools import setup, find_packages
import os
import shutil

_major = '0'
_minor = '4'
_patch = '0'

README_FILE = 'README.md'
LICENSE_FILE = 'LICENSE.txt'

# Note: used when generating the wheel but not on pip install of the package
if os.path.exists('../LICENSE'):
    shutil.copyfile('../LICENSE', LICENSE_FILE)

VERSION = '{}.{}.{}'.format(_major, _minor, _patch)


CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
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
    'packaging',
    'interpret-core[required]==0.1.19',
    'shap>=0.20.0, <=0.33.0'
]

EXTRAS = {
    'sample': [
        'hdbscan'
    ],
    'deep': [
        'tensorflow',
        'pyyaml',
        'keras'
    ],
    'mimic': [
        'lightgbm'
    ]
}

with open(README_FILE, 'r', encoding='utf-8') as f:
    README = f.read()

setup(
    name='interpret-community',

    version=VERSION,

    description='Microsoft Interpret Extensions SDK for Python',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Microsoft Corp',
    author_email='ilmat@microsoft.com',
    license='MIT License',
    url='https://github.com/interpretml/interpret-community',

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
        ],
        "interpret_ext_glassbox": [
            "LGBMExplainableModel = interpret_community.mimic.models:LGBMExplainableModel",
            "LinearExplainableModel = interpret_community.mimic.models:LinearExplainableModel",
            "SGDExplainableModel = interpret_community.mimic.models:SGDExplainableModel",
            "DecisionTreeExplainableModel = interpret_community.mimic.models:DecisionTreeExplainableModel"
        ]
    },
    include_package_data=True,
    data_files=[
        ('share/jupyter/nbextensions/interpret-ml-widget', [
            'interpret_community/widget/static/extension.js',
            'interpret_community/widget/static/extension.js.map',
            'interpret_community/widget/static/index.js',
            'interpret_community/widget/static/index.js.map'
        ]),
        ("etc/jupyter/nbconfig/notebook.d", [
            "jupyter-config/nbconfig/notebook.d/interpret-ml-widget.json"
        ]),
        ('share/jupyter/lab/extensions', [
            'interpret_community/widget/js/'
            'interpret_ml_widget/labextension/interpret-ml-widget-0.1.7.tgz'
        ])],
    zip_safe=False,
    extras_require=EXTRAS
)
