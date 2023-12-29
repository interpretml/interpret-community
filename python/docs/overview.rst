.. _overview:

Overview
========

Interpret-Community extends the `Interpret <https://github.com/interpretml/interpret>`_ repository and incorporates further community developed and experimental interpretability techniques and functionalities that are designed to enable interpretability for real world scenarios.
Interpret-Community enables adding new experimental techniques (or functionalities) and performing comparative analysis to evaluate them.

Interpret-Community 

1. Actively incorporates innovative experimental interpretability techniques and allows for further expansion by researchers and data scientists
2. Applies optimizations to make it possible to run interpretability techniques on real-world datasets at scale
3. Provides improvements such as the capability to "reverse the feature engineering pipeline" to provide model insights in terms of the original raw features rather than engineered features
4. Provides interactive and exploratory visualizations to empower data scientists to gain meaningful insight into their data


Getting Started
===============

The package can be installed from `pypi <https://pypi.org/project/interpret-community/>`_ with:

   .. code-block:: bash

      pip install interpret-community


You can use Anaconda to simplify package and environment management.

To setup on your local machine:

.. raw:: html

    <style>
        #inner {
            margin-left:50px; 
            margin-right:-50px;
        }
    </style>
    <details><summary><strong><em>1. Set up Environment</em></strong></summary>

        a. Install Anaconda with Python >= 3.7
        <br/>
        <a href="https://conda.io/projects/conda/en/latest/user-guide/install/index.html">Miniconda</a> is a quick way to get started.
        <br/>
        <br/>
        b. Create conda environment named interp and install packages
        <br/>

        <blockquote>
            <div>
                <div class="highlight-bash notranslate">
                    <pre>conda create --name interp python=3.7 anaconda</pre>
                </div>
            </div>
        </blockquote>

    <br/>
    Optional, additional reading:
    <br/>
    <a href="https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf">conda cheat sheet</a>
    <a href="https://pypi.org/project/jupyter/">jupyter</a>
    <a href="https://github.com/Anaconda-Platform/nb_conda">nb_conda</a>
    <div id="inner">
    <details><summary><strong><em>On Linux and Windows: c. Activate conda environment</strong></em></summary>

        <blockquote>
            <div>
                <div class="highlight-bash notranslate">
                    <pre>activate interp</pre>
                </div>
            </div>
        </blockquote>

    </details>

    <details><summary><strong><em>On Mac:</em> c. Activate conda environment</em></strong></summary>

        <blockquote>
            <div>
                <div class="highlight-bash notranslate">
                    <pre>source activate interp</pre>
                </div>
            </div>
        </blockquote>

    </details>
    </div>
    </details>

    <details>

    <summary><strong><em>2. Clone the Interpret-Community repository</em></strong></summary>

        Clone and cd into the repository

        <br/>

        <blockquote>
            <div>
                <div class="highlight-bash notranslate">
                    <pre>git clone https://github.com/interpretml/interpret-community
                    <br/>cd interpret-community
                    </pre>
                </div>
            </div>
        </blockquote>

    </details>

    <details>
    <summary><strong><em>3. Install Python module, packages and necessary distributions</em></strong></summary>

        <blockquote>
            <div>
                <div class="highlight-bash notranslate">
                    <pre>pip install interpret-community</pre>
                </div>
            </div>
        </blockquote>

        <br/>
        If you intend to run repository tests:
        <br/>
        <blockquote>
            <div>
                <div class="highlight-bash notranslate">
                    <pre>pip install -r requirements.txt</pre>
                </div>
            </div>
        </blockquote>
    <div id="inner">
    <details><summary><strong><em>On Windows: </strong></em></summary>

    Pytorch installation if desired:

    <blockquote>
        <div>
            <div class="highlight-bash notranslate">
                <pre>conda install --yes --quiet pytorch torchvision cpuonly -c pytorch
                <br/>pip install captum</pre>
            </div>
        </div>
    </blockquote>

    lightgbm installation if desired:

    <blockquote>
        <div>
            <div class="highlight-bash notranslate">
                <pre>pip install --upgrade lightgbm</pre>
            </div>
        </div>
    </blockquote>

    </details>
    <details><summary><strong><em>On Linux: </strong></em></summary>

    Pytorch installation if desired:

    <br/>
    <blockquote>
        <div>
            <div class="highlight-bash notranslate">
                <pre>conda install --yes --quiet pytorch torchvision cpuonly -c pytorch
                <br/>pip install captum</pre>
            </div>
        </div>
    </blockquote>
    <br/>

    lightgbm installation if desired:

    <blockquote>
        <div>
            <div class="highlight-bash notranslate">
                <pre>pip install --upgrade lightgbm</pre>
            </div>
        </div>
    </blockquote>
    </details>

    <details><summary><strong><em>On MacOS: </strong></em></summary>
    <br/>
    Pytorch installation if desired:
    <blockquote>
        <div>
            <div class="highlight-bash notranslate">
                <pre>conda install --yes --quiet pytorch torchvision -c pytorch
                <br/>pip install captum</pre>
            </div>
        </div>
    </blockquote>
    <br/>
    lightgbm installation if desired (requires Homebrew):
    <blockquote>
        <div>
            <div class="highlight-bash notranslate">
                <pre>brew install libomp
                <br/>pip install --upgrade lightgbm</pre>
            </div>
        </div>
    </blockquote>

    If installing the package generally gives an error about the `certifi` package, run this first:
    <blockquote>
        <div>
            <div class="highlight-bash notranslate">
                <pre>pip install --upgrade certifi --ignore-installed
                <br/>pip install interpret-community</pre>
            </div>
        </div>
    </blockquote>

    </details>
    </div>
    </details>

    <details>
    <summary><strong><em>4. Set up and run Jupyter Notebook server </em></strong></summary>

    Install and run Jupyter Notebook
    if needed:
    <blockquote>
        <div>
            <div class="highlight-bash notranslate">
                <pre>
                </br>pip install jupyter</pre>
            </div>
        </div>
    </blockquote>
    then:
    <blockquote>
        <div>
            <div class="highlight-bash notranslate">
                <pre>
                </br>jupyter notebook</pre>
            </div>
        </div>
    </blockquote>
    </details>
