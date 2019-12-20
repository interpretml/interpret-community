import papermill as pm
import scrapbook as sb
import pytest


@pytest.mark.notebooks
def test_explain_binary_classification_local():

    notebookname = "explain-binary-classification-local"
    input_notebook = "notebooks/" + notebookname + ".ipynb"
    output_notebook = "./test/" + notebookname + ".output.ipynb"

    pm.execute_notebook(input_notebook, output_notebook)

    nb = sb.read_notebook(output_notebook)
    nb.scraps  # print a dict of all scraps by name

    return


@pytest.mark.notebooks
def test_explain_regression_local():

    notebookname = "explain-regression-local"
    input_notebook = "notebooks/" + notebookname + ".ipynb"
    output_notebook = "./test/" + notebookname + ".output.ipynb"

    pm.execute_notebook(input_notebook, output_notebook)
    nb = sb.read_notebook(output_notebook)
    print(nb.scraps)  # print a dict of all scraps by name
    print(nb.scrap_dataframe)
    assert True

    return


@pytest.mark.notebooks
def test_read_sb():
    notebookname = "explain-regression-local"
    input_notebook = "notebooks/" + notebookname + ".ipynb"
    output_notebook = "./test/" + notebookname + ".output.ipynb"
    bn = sb.read_notebook(output_notebook)
    print(bn.scraps)
    print(bn.scrap_dataframe)
    print("local imp is ", bn.scraps.data_dict["local_imp"])
    if "AGE" in bn.scraps.data_dict["local_imp"]:
        print("found age")
        assert True
    else:
        assert False

    return


@pytest.mark.notebooks
def test_advanced_feature_transformations_explain_local():

    notebookname = "advanced-feature-transformations-explain-local"
    input_notebook = "notebooks/" + notebookname + ".ipynb"
    output_notebook = "./test/" + notebookname + ".output.ipynb"

    pm.execute_notebook(input_notebook, output_notebook)

    nb = sb.read_notebook(output_notebook)
    nb.scraps  # print a dict of all scraps by name

    return


@pytest.mark.notebooks
def test_explain_multiclass_classification_local():

    notebookname = "explain-multiclass-classification-local"
    input_notebook = "notebooks/" + notebookname + ".ipynb"
    output_notebook = "./test/" + notebookname + ".output.ipynb"

    pm.execute_notebook(input_notebook, output_notebook)

    nb = sb.read_notebook(output_notebook)
    nb.scraps  # print a dict of all scraps by name

    return


@pytest.mark.notebooks
def test_simple_feature_transformations_explain_local():

    notebookname = "simple-feature-transformations-explain-local"
    input_notebook = "notebooks/" + notebookname + ".ipynb"
    output_notebook = "./test/" + notebookname + ".output.ipynb"

    pm.execute_notebook(input_notebook, output_notebook)

    nb = sb.read_notebook(output_notebook)
    nb.scraps  # print a dict of all scraps by name

    return
