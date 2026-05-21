# Linear Regression with Unknown Truncation Beyond Gaussian Features
Simulation code for ICML 2026 paper (https://arxiv.org/abs/2602.12534v1)

## Purpose
The script `main.py` sets up a truncated linear regression problem, where each feature vector $x$ is drawn from a mixture of Gaussians in $\mathbb{R}^d$ and the scalar response is given by $y = x^\top w^\star + \xi$, where $w^\star$ is an unknown parameter vector and $\xi$ is Gaussian noise in $\mathbb{R}$. Samples $(x, y)$ are only observed when $y$ falls into some unknown survival set $S^\star$, and the goal is to estimate $w^\star$ from the observed samples.

After setting up the problem, the script runs one or more algorithms that attempt to estimate $w^\star$ from samples generated via the above model. The primary algorithm that can be run is the one presented in the paper, which first approximately learns the survival set $S^\star$ from samples, and then runs Projected Stochastic Gradient Descent (PSGD) on a generalized version of the Maximum Likelihood (MLE) objective to recover $w^\star$. For comparison, the script allows for the following algorithms to be run as well:

1. Ordinary Least Squares (OLS) without accounting for truncation. This serves as our baseline.
2. PSGD on the MLE objective with misspecified survival set $S \neq S^\star$.
3. PSGD on the MLE objective with the true survival set $S^\star$. Note that this is an idealized algorithm that cannot actually be implemented if $S^\star$ is unknown.

After running the algorithms, the script generates a plot showing, for each algorithm, the distance of the PSGD iterates $w_t$ from the true parameter vector $w^\star$.

## Usage
The basic usage is to first create a YAML configuration file to set up the experiment (see the example `config.yaml` file), and then run `python main.py --config PATH_TO_YAML_FILE`. To run the experiment multiple times, instead execute `python main.py --config PATH_TO_YAML_FILE --R NUMBER_OF_REPS`. In that case, the generated plot will show the standard deviation around the mean error over reruns. If no configuration file is provided (i.e. if simply `python main.py` is run), a default experiment will be run, which is the same experiment presented in the paper.

By default, the script will run all four algorithms presented above, for the sake of comparison. To run only a subset of these algorithms, pass the `--methods COMMA_SEPARATED_LIST` option with a comma-separated list of the algorithms to be run, as follows:
1. Primary algorithm: `full`
2. OLS: `ols`
3. PSGD with misspecified survival set: `wrong_set`
4. PSGD with true survival set: `true_set`

A few more helper options are provided, and they can be viewed by running `python main.py -h` or `python main.py --help`.
