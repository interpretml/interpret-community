import nbformat as nbf
import papermill as pm
import pytest
import scrapbook as sb

SORTED_LOCAL_IMPORTANCE_NAMES = 'sorted_local_importance_names'


def append_scrapbook_commands(input_nb_path, output_nb_path, scrap_specs):
    notebook = nbf.read(input_nb_path, as_version=4)
    notebook = nbf.v4.upgrade(notebook, from_version=4, from_minor=2)

    scrapbook_cells = []
    # Always need to import nteract-scrapbook
    scrapbook_cells.append(nbf.v4.new_code_cell(source='import scrapbook as sb'))

    # Create a cell to store each key and value in the scrapbook
    for k, v in scrap_specs.items():
        source = "sb.glue(\"{0}\", {1})".format(k, v)
        scrapbook_cells.append(nbf.v4.new_code_cell(source=source))

    # Append the cells to the notebook
    [notebook['cells'].append(c) for c in scrapbook_cells]

    # Write out the new notebook
    nbf.write(notebook, output_nb_path)


def input_notebook_path(notebookname):
    return "notebooks/{0}.ipynb".format(notebookname)


def processed_notebook_path(notebookname):
    return "./tests/{0}.processed.ipynb".format(notebookname)


def output_notebook_path(notebookname):
    return "./tests/{0}.output.ipynb".format(notebookname)


@pytest.mark.notebooks()
def test_explain_binary_classification_local():
    notebookname = 'explain-binary-classification-local'
    input_notebook = input_notebook_path(notebookname)
    output_notebook = output_notebook_path(notebookname)
    processed_notebook = processed_notebook_path(notebookname)
    test_values = {SORTED_LOCAL_IMPORTANCE_NAMES: SORTED_LOCAL_IMPORTANCE_NAMES}
    append_scrapbook_commands(input_notebook, processed_notebook, test_values)
    pm.execute_notebook(processed_notebook, output_notebook)
    nb = sb.read_notebook(output_notebook)
    assert 'worst area' in nb.scraps.data_dict[SORTED_LOCAL_IMPORTANCE_NAMES]


@pytest.mark.notebooks()
def test_explain_regression_local():
    notebookname = 'explain-regression-local'
    input_notebook = input_notebook_path(notebookname)
    output_notebook = output_notebook_path(notebookname)
    processed_notebook = processed_notebook_path(notebookname)
    test_values = {SORTED_LOCAL_IMPORTANCE_NAMES: SORTED_LOCAL_IMPORTANCE_NAMES}
    append_scrapbook_commands(input_notebook, processed_notebook, test_values)
    pm.execute_notebook(processed_notebook, output_notebook)
    nb = sb.read_notebook(output_notebook)
    assert 'Latitude' in nb.scraps.data_dict[SORTED_LOCAL_IMPORTANCE_NAMES]


@pytest.mark.notebooks()
def test_explain_regression_mimic_explainer():
    notebookname = 'explain-regression-mimic-explainer'
    input_notebook = input_notebook_path(notebookname)
    output_notebook = output_notebook_path(notebookname)
    processed_notebook = processed_notebook_path(notebookname)
    test_values = {SORTED_LOCAL_IMPORTANCE_NAMES: SORTED_LOCAL_IMPORTANCE_NAMES}
    append_scrapbook_commands(input_notebook, processed_notebook, test_values)
    pm.execute_notebook(processed_notebook, output_notebook)
    nb = sb.read_notebook(output_notebook)
    assert 'Latitude' in nb.scraps.data_dict[SORTED_LOCAL_IMPORTANCE_NAMES]


@pytest.mark.notebooks()
def test_advanced_feature_transformations_explain_local():
    notebookname = 'advanced-feature-transformations-explain-local'
    input_notebook = input_notebook_path(notebookname)
    output_notebook = output_notebook_path(notebookname)
    processed_notebook = processed_notebook_path(notebookname)
    test_values = {SORTED_LOCAL_IMPORTANCE_NAMES: SORTED_LOCAL_IMPORTANCE_NAMES}
    append_scrapbook_commands(input_notebook, processed_notebook, test_values)
    pm.execute_notebook(processed_notebook, output_notebook)
    nb = sb.read_notebook(output_notebook)
    assert 'embarked' in nb.scraps.data_dict[SORTED_LOCAL_IMPORTANCE_NAMES][0]


@pytest.mark.notebooks()
def test_explain_multiclass_classification_local():
    notebookname = 'explain-multiclass-classification-local'
    input_notebook = input_notebook_path(notebookname)
    output_notebook = output_notebook_path(notebookname)
    processed_notebook = processed_notebook_path(notebookname)
    test_values = {SORTED_LOCAL_IMPORTANCE_NAMES: SORTED_LOCAL_IMPORTANCE_NAMES}
    append_scrapbook_commands(input_notebook, processed_notebook, test_values)
    pm.execute_notebook(processed_notebook, output_notebook)
    nb = sb.read_notebook(output_notebook)
    assert 'petal width (cm)' in nb.scraps.data_dict[SORTED_LOCAL_IMPORTANCE_NAMES]


@pytest.mark.notebooks()
def test_simple_feature_transformations_explain_local():
    notebookname = 'simple-feature-transformations-explain-local'
    input_notebook = input_notebook_path(notebookname)
    output_notebook = output_notebook_path(notebookname)
    processed_notebook = processed_notebook_path(notebookname)
    test_values = {SORTED_LOCAL_IMPORTANCE_NAMES: SORTED_LOCAL_IMPORTANCE_NAMES}
    append_scrapbook_commands(input_notebook, processed_notebook, test_values)
    pm.execute_notebook(processed_notebook, output_notebook)
    nb = sb.read_notebook(output_notebook)
    assert 'TotalWorkingYears' in nb.scraps.data_dict[SORTED_LOCAL_IMPORTANCE_NAMES][0]


@pytest.mark.notebooks()
def test_captum_integration_example():
    notebookname = 'captum-integration-example'
    input_notebook = input_notebook_path(notebookname)
    output_notebook = output_notebook_path(notebookname)
    processed_notebook = processed_notebook_path(notebookname)
    test_values = {SORTED_LOCAL_IMPORTANCE_NAMES: SORTED_LOCAL_IMPORTANCE_NAMES}
    append_scrapbook_commands(input_notebook, processed_notebook, test_values)
    pm.execute_notebook(processed_notebook, output_notebook)
    nb = sb.read_notebook(output_notebook)
    assert 'B' in nb.scraps.data_dict[SORTED_LOCAL_IMPORTANCE_NAMES]
