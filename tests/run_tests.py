# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Run the repository test suites with optional frameworks isolated."""

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TESTS = _ROOT / 'tests'


def _module_available(name):
    return importlib.util.find_spec(name) is not None


def _run_pytest(arguments, tensorflow, pytorch):
    environment = os.environ.copy()
    environment['INTERPRET_TEST_TENSORFLOW'] = str(int(tensorflow))
    environment['INTERPRET_TEST_PYTORCH'] = str(int(pytorch))
    environment['PYTHONPATH'] = os.pathsep.join(
        [str(_TESTS), environment.get('PYTHONPATH', '')]).rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-p', 'framework_isolation'] +
        arguments,
        cwd=_ROOT,
        env=environment)
    if result.returncode:
        raise SystemExit(result.returncode)


def _dashboard_arguments():
    if _module_available('raiwidgets'):
        return []
    print('raiwidgets is not installed; skipping dashboard tests.')
    return ['--ignore=tests/test_explanation_dashboard.py']


def _coverage_arguments(append=False):
    arguments = [
        '--cov=interpret_community',
        '--cov-report=xml',
        '--cov-report=html'
    ]
    if append:
        arguments.insert(1, '--cov-append')
    return arguments


def _run_compatibility_tests(coverage, pytest_arguments):
    arguments = [
        'numpy_compatibility_tests',
        'python_compatibility_tests',
        '-v',
        '--durations=10'
    ] + pytest_arguments
    if coverage:
        arguments += _coverage_arguments()
    _run_pytest(arguments, tensorflow=False, pytorch=False)


def _run_selected_tests(coverage, framework, pytest_arguments):
    arguments = pytest_arguments
    if coverage:
        arguments += _coverage_arguments()
    _run_pytest(
        arguments,
        tensorflow=framework == 'tensorflow',
        pytorch=framework == 'pytorch')


def _run_all_tests(coverage):
    arguments = [
        'tests',
        'numpy_compatibility_tests',
        'python_compatibility_tests',
    ] + _dashboard_arguments() + [
        '-m',
        'not notebooks',
        '-s',
        '-v'
    ]
    if coverage:
        arguments += _coverage_arguments()
    _run_pytest(arguments, tensorflow=False, pytorch=True)

    if not _module_available('tensorflow'):
        print('TensorFlow is not installed; skipping TensorFlow integrations.')
        return

    arguments = [
        'tests',
        '--ignore=tests/test_explanation_adapter.py',
    ] + _dashboard_arguments() + [
        '-m',
        'not notebooks',
        '-k',
        'dnn or keras or deepexplainer',
        '-s',
        '-v'
    ]
    if coverage:
        arguments += _coverage_arguments(append=True)
    _run_pytest(arguments, tensorflow=True, pytorch=False)


def main():
    parser = argparse.ArgumentParser(
        description='Run interpret-community tests.',
        allow_abbrev=False)
    parser.add_argument(
        '--compatibility-only',
        action='store_true',
        help='Run only the NumPy and public API compatibility suites.')
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Collect XML and HTML coverage reports.')
    parser.add_argument(
        '--framework',
        choices=('none', 'pytorch', 'tensorflow'),
        help='Framework to enable for a targeted pytest invocation.')
    arguments, pytest_arguments = parser.parse_known_args()
    if pytest_arguments[:1] == ['--']:
        pytest_arguments = pytest_arguments[1:]

    if arguments.compatibility_only:
        if arguments.framework:
            parser.error(
                '--framework cannot be used with --compatibility-only')
        _run_compatibility_tests(
            arguments.coverage,
            pytest_arguments)
    elif pytest_arguments:
        if not arguments.framework:
            parser.error(
                'targeted tests require --framework none, pytorch, or '
                'tensorflow')
        _run_selected_tests(
            arguments.coverage,
            arguments.framework,
            pytest_arguments)
    elif arguments.framework:
        parser.error('--framework requires pytest arguments')
    else:
        _run_all_tests(arguments.coverage)


if __name__ == '__main__':
    main()
