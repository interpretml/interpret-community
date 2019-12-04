import papermill as pm
import scrapbook as sb
import pytest


@pytest.mark.notebooks
def test_explain_binary_classification_local():

    notebookname = "explain-binary-classification-local"
    input_notebook = "../notebooks/" + notebookname + ".ipynb"
    output_notebook = "./" + notebookname + ".output.ipynb"

    pm.execute_notebook(input_notebook, output_notebook)

    nb = sb.read_notebook(input_notebook)
    nb.scraps  # print a dict of all scraps by name

    return


@pytest.mark.notebooks
def test_explain_regression_local():

    notebookname = "explain-regression-local"
    input_notebook = "../notebooks/" + notebookname + ".ipynb"
    output_notebook = "./" + notebookname + ".output.ipynb"

    pm.execute_notebook(input_notebook, output_notebook)

    nb = sb.read_notebook(input_notebook)
    nb.scraps  # print a dict of all scraps by name

    return


@pytest.mark.notebooks
def test_advanced_feature_transformations_explain_local():

    notebookname = "advanced-feature-transformations-explain-local"
    input_notebook = "../notebooks/" + notebookname + ".ipynb"
    output_notebook = "./" + notebookname + ".output.ipynb"

    pm.execute_notebook(input_notebook, output_notebook)

    nb = sb.read_notebook(input_notebook)
    nb.scraps  # print a dict of all scraps by name

    return


@pytest.mark.notebooks
def test_explain_multiclass_classification_local():

    notebookname = "explain-multiclass-classification-local"
    input_notebook = "../notebooks/" + notebookname + ".ipynb"
    output_notebook = "./" + notebookname + ".output.ipynb"

    pm.execute_notebook(input_notebook, output_notebook)

    nb = sb.read_notebook(input_notebook)
    nb.scraps  # print a dict of all scraps by name

    return


@pytest.mark.notebooks
def test_simple_feature_transformations_explain_local():

    notebookname = "simple-feature-transformations-explain-local"
    input_notebook = "../notebooks/" + notebookname + ".ipynb"
    output_notebook = "./" + notebookname + ".output.ipynb"

    pm.execute_notebook(input_notebook, output_notebook)

    nb = sb.read_notebook(input_notebook)
    nb.scraps  # print a dict of all scraps by name

    return
