# Prompting Log - Maham Asif

## Environment Setup
* **Prompt:** "Initialize a uv project with numpy, pandas, scikit-learn, pytest, and ruff."
* **Model:** Claude Code 
* **Outcome:** The agent successfully initialized the `.venv` virtual environment, and generated the `pyproject.toml` file locking in the core and development dependencies.

## Loading Dataset
* **Prompt:** "Write a python script in the next cell that loads the 'mnist_784' dataset directly using fetch_openml and downsamples it to 5,000 samples to optimize SVM training speed and splits it 80/20, and scales the features using StandardScaler."
* **Model:** Claude Code 
* **Outcome:** successfully produced an 80/20 split with training shape `(4000, 784)` and testing shape `(1000, 784)`

## Notebook Documentation & Refinement
* **Prompt:** "Please add markdown cells with short descriptions of each cell just to give an idea of what it's doing."
* **Model:** Claude Code 
* **Outcome:** The agent successfully documented the entire Jupyter Notebook, adding clear markdown headers and context descriptions above each code block. This ensures the experiment is self-documenting and easy for reviewers to follow.

## Configuring Ruff and lint pass
* **Prompt:** "configure ruff for my project to ensure a clean lint pass. After updating the configuration, run ruff on the notebook to clean up code and fix any formatting issues"
* **Model:** Claude Code 
* **Outcome:** Ruff found and fixed 6 lint errors (likely unused imports like pd, Pipeline, cross_val_score, GridSearchCV) and reformatted the notebook for consistent style.

## Testing 
* **Prompt:** "write pytest unit tests for:
1) Downsampling to check that output has exactly 5000 samples and that no duplicate indices were selected
2) the train/test split to ensure split is 80/20 and theres no overlap between the train and test indices.
3) StandardScaler to ensure x_train has mean 0 and std 1 after scaling and that x_test is transformed using train statistics. Also the scaler raises if transform is called before fit"
* **Model:** Claude Code 
* **Outcome:** All 7 tests passed