.. _contributing:

Contributing
============

Interpret-Community is an open-source project and anyone is welcome to contribute to the project.

Acceptance criteria
-------------------

All pull requests need to abide by the following criteria to be accepted:

* passing pipelines on the GitHub pull request
* approval from at least one maintainer
* tests for added / changed functionality


Development process
-------------------

The project can be installed locally in editable mode using the following command:

   .. code-block:: bash

      pip install -e .


Testing
-------

To run tests locally, please install all of the dependencies in your python environment:

   .. code-block:: bash

      pip install -r requirements-dev.txt
      pip install -r requirements-vis.txt
      pip install -r requirements-test.txt

The unit tests can then be run via the command:

   .. code-block:: bash

      pytest ./tests -m "not notebooks" -s -v

Code coverage can be run locally via the command:

   .. code-block:: bash

      pytest ./tests -m "not notebooks" -s -v --cov='interpret_community' --cov-report=xml --cov-report=html

Notebook tests can also be run via the command:

   .. code-block:: bash

      python -m pytest tests/ -m "notebooks" -s -v


Linting
-------

This repository uses flake8 for linting and isort to automatically sort imports.

Before running flake8, first please ensure that requirements-test.txt has been installed.

Then, you can run flake8 at the root folder of the repository via:

    .. code-block:: bash

       flake8 --max-line-length=119 .

You can use isort to identify if any imports are out of order:

    .. code-block:: bash

       isort . -c

You can even automatically fix your code by removing the -c argument:

    .. code-block:: bash

       isort .

After automatically sorting the imports in your code, make sure to add and commit your changes to git.
