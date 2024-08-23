# plotpal
Post-processing tools for plotting results from Dedalus simulations.
The code in this repository is backed up on Zenodo, and is citeable at: [![DOI](https://zenodo.org/badge/265006663.svg)](https://zenodo.org/badge/latestdoi/265006663)

For more information on the kinds of plots plotpal can make, refer to the examples/ folder and see the README files for specific examples.

# Installation
To install plotpal on your local machine, first [ensure that you have openssl installed](https://chatgpt.com/share/a88a5b9f-83bd-42bc-a663-0806f9a48c7a) and then install Dedalus in a conda environment according to the instructions provided [in the dedalus docs](https://dedalus-project.readthedocs.io/en/latest/pages/installation.html). 

Check that your dedalus installation with ssl was successful by running the following:

```sh
conda activate your-dedalus-environment
python3
>>> import ssl
>>> import dedalus.public as d3
```

Then, follow the steps below:

1. Clone the repository:
    ```sh
    git clone https://github.com/evanhanders/plotpal.git
    cd plotpal
    ```

3. Install the package and dependencies:
    ```sh
    pip install --use-pep517 -e .
    ```

# Usage

1. Copy one of the python scripts from the example/ directory somewhere closer to where you're running dedalus simulations (or just modify one of your local files there).
2. Put in the fields you care about plotting.
3. Make some plots!

By default, plots will be located within the parent directory of your Dedalus simulation, in a new folder.
