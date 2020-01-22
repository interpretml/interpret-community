import nbformat as nbf
import papermill as pm
import scrapbook as sb
import pytest


def append_scrapbook_commands(input_nb_path, output_nb_path, scrap_specs):
    notebook = nbf.read(input_nb_path, as_version=nbf.NO_CONVERT)

    scrapbook_cells = []
    # Always need to import nteract-scrapbook
    scrapbook_cells.append(nbf.v4.new_code_cell(source="import scrapbook as sb"))

    # Create a cell to store each key and value in the scrapbook
    for k, v in scrap_specs.items():
        source = "sb.glue(\"{0}\", {1})".format(k, v)
        scrapbook_cells.append(nbf.v4.new_code_cell(source=source))

    # Append the cells to the notebook
    [notebook['cells'].append(c) for c in scrapbook_cells]

    # Write out the new notebook
    nbf.write(notebook, output_nb_path)


def input_notebook_path(notebookname):
    return "notebooks/" + notebookname + ".ipynb"


def processed_notebook_path(notebookname):
    return "./test/" + notebookname + ".processed.ipynb"


def output_notebook_path(notebookname):
    return "./test/" + notebookname + ".output.ipynb"


@pytest.mark.notebooks
def test_explain_binary_classification_local():
    notebookname = "explain-binary-classification-local"
    input_notebook = input_notebook_path(notebookname)
    output_notebook = output_notebook_path(notebookname)
    processed_notebook = processed_notebook_path(notebookname)
    test_values = {}
    append_scrapbook_commands(input_notebook, processed_notebook, test_values)
    pm.execute_notebook(processed_notebook, output_notebook)

    nb = sb.read_notebook(output_notebook)
    nb.scraps  # print a dict of all scraps by name

    return


@pytest.mark.notebooks
def test_explain_regression_local():
    notebookname = "explain-regression-local"
    input_notebook = input_notebook_path(notebookname)
    output_notebook = output_notebook_path(notebookname)
    processed_notebook = processed_notebook_path(notebookname)
    test_values = {"local_imp": "sorted_local_importance_names"}
    append_scrapbook_commands(input_notebook, processed_notebook, test_values)
    pm.execute_notebook(processed_notebook, output_notebook)
    nb = sb.read_notebook(output_notebook)
    print(nb.scraps)  # print a dict of all scraps by name
    print(nb.scrap_dataframe)
    assert "AGE" in nb.scraps.data_dict["local_imp"]
    assert True

    return


@pytest.mark.notebooks
def test_advanced_feature_transformations_explain_local():
    notebookname = "advanced-feature-transformations-explain-local"
    input_notebook = input_notebook_path(notebookname)
    output_notebook = output_notebook_path(notebookname)
    processed_notebook = processed_notebook_path(notebookname)
    test_values = {}
    append_scrapbook_commands(input_notebook, processed_notebook, test_values)
    pm.execute_notebook(processed_notebook, output_notebook)

    nb = sb.read_notebook(output_notebook)
    nb.scraps  # print a dict of all scraps by name

    return


@pytest.mark.notebooks
def test_explain_multiclass_classification_local():
    notebookname = "explain-multiclass-classification-local"
    input_notebook = input_notebook_path(notebookname)
    output_notebook = output_notebook_path(notebookname)
    processed_notebook = processed_notebook_path(notebookname)
    test_values = {}
    append_scrapbook_commands(input_notebook, processed_notebook, test_values)
    pm.execute_notebook(processed_notebook, output_notebook)

    nb = sb.read_notebook(output_notebook)
    nb.scraps  # print a dict of all scraps by name

    return


@pytest.mark.notebooks
def test_simple_feature_transformations_explain_local():
    notebookname = "simple-feature-transformations-explain-local"
    input_notebook = input_notebook_path(notebookname)
    output_notebook = output_notebook_path(notebookname)
    processed_notebook = processed_notebook_path(notebookname)
    test_values = {}
    append_scrapbook_commands(input_notebook, processed_notebook, test_values)
    pm.execute_notebook(processed_notebook, output_notebook)

    nb = sb.read_notebook(output_notebook)
    nb.scraps  # print a dict of all scraps by name

    return
