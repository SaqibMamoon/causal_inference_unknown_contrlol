# Stuffs

The code base is managed by `conda` since that gives simple installation for `numpy` and `scipy`.
The local code is only python, so it is installed as a `pip` editable package, to make sure imports work as they should.
Some dev-dependencies are included, notably `flake8` and `black`, but they are not needed for running the code really...



1. Make sure you have `conda` installed (anaconda or miniconda as you like)
1. Install the conda environment by `conda  env create --file=environment.yml`
1. When developing/making updates, do so via `conda env update --file=environment.yml --prune`
1. Do linting using flake8, by `flake8` from command line (or via your IDE)
1. Do formatting using black, by `black .` from command line (or via your IDE)

To run any code, run `python ./experiments/<experiment_name>.py` **from the root**.
Some of the experiments take command line options, e.g. `python ./experiments/<experiment_name>.py --option1=20`.
Output is put in the `./output` folder.