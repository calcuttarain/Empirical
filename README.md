# Empirical

This repository implements a minimalist framework for printing, logging and saving results of numerical simulations. Its purpose is to emphasise simplicity and reduced complexity of the code that would otherwise distract from mathematical and methodological problems.

Similar ideas have been implemented, but none tailored for this specific problem. For instance, [MLflow](https://mlflow.org/) can be overly complex. Another framework, [Sacred](https://github.com/IDSIA/sacred), although very similar, relies on external databases and complex configuration systems.

---

## Installation

`Empirical` can be installed directly from GitHub using `pip`. 

For standard usage in your projects, run:
```bash
pip install git+https://github.com/calcuttarain/Empirical.git
```

For modifications, clone the repository and install it in editable mode:
```bash
git clone https://github.com/calcuttarain/Empirical.git
cd Empirical
pip install -e .
```

---

## Code Usage and Examples

This framework intends to solve the following problem.

Given a function with (possibly) various input parameter configurations which runs some heavyweight computations in a loop, it:
- saves results in an organised manner,
- stores the input parameters,
- tracks metadata about the experiments,
- ensures the reproducibility of the code,
- enables easy loading and inspecting of the saved data,
while maintaining clean code.

Check the [examples](examples/) folder for specific usages, starting with [experiment_example.ipynb](examples/experiment_example.ipynb).

---

## Contributions and Suggestions

For contributions, open a Pull Request, and for suggestions or bug reports, open an Issue.