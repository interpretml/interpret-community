# Tests

A number of tests are provided to validate the functionality of the repository.
All tests are in the test folder.

If you intend to run repository tests:

```bash
pip install -e ./python
pip install -r requirements-dev.txt
pip install -r requirements-test.txt
pip install -r requirements-vis.txt
```

## Execute all tests from the root folder:
```bash
python tests/run_tests.py
```

To collect the same coverage reports as CI:
```bash
python tests/run_tests.py --coverage
```

Individual tests can still be run directly:
```bash
python tests/run_tests.py --framework none path/to/test_filename.py
```

Select the matching isolated process for framework-specific tests:
```bash
python tests/run_tests.py --framework pytorch path/to/test_filename.py -k test_name
python tests/run_tests.py --framework tensorflow path/to/test_filename.py -k test_name
```

Use `--coverage --append-coverage` when a targeted run should contribute to
coverage data produced by an earlier test process.
