# Prompting Log - Maham Asif

## Week 1

### Environment Setup

- **Prompt:** "Initialize a uv project with numpy, pandas, scikit-learn, pytest, and ruff."
- **Model:** Claude Code
- **Outcome:** The agent successfully initialized the `.venv` virtual environment, and generated the `pyproject.toml` file locking in the core and development dependencies.

### Loading Dataset

- **Prompt:** "Write a python script in the next cell that loads the 'mnist_784' dataset directly using fetch_openml and downsamples it to 5,000 samples to optimize SVM training speed and splits it 80/20, and scales the features using StandardScaler."
- **Model:** Claude Code
- **Outcome:** Successfully produced an 80/20 split with training shape `(4000, 784)` and testing shape `(1000, 784)`.

### Notebook Documentation & Refinement

- **Prompt:** "Please add markdown cells with short descriptions of each cell just to give an idea of what it's doing."
- **Model:** Claude Code
- **Outcome:** The agent successfully documented the entire Jupyter Notebook, adding clear markdown headers and context descriptions above each code block. This ensures the experiment is self-documenting and easy for reviewers to follow.

### Configuring Ruff and Lint Pass

- **Prompt:** "Configure ruff for my project to ensure a clean lint pass. After updating the configuration, run ruff on the notebook to clean up code and fix any formatting issues."
- **Model:** Claude Code
- **Outcome:** Ruff found and fixed 6 lint errors (likely unused imports like `pd`, `Pipeline`, `cross_val_score`, `GridSearchCV`) and reformatted the notebook for consistent style.

### Testing

- **Prompt:** "Write pytest unit tests for: 1) Downsampling to check that output has exactly 5000 samples and that no duplicate indices were selected. 2) The train/test split to ensure split is 80/20 and there's no overlap between the train and test indices. 3) StandardScaler to ensure X_train has mean 0 and std 1 after scaling and that X_test is transformed using train statistics. Also the scaler raises if transform is called before fit."
- **Model:** Claude Code
- **Outcome:** All 7 tests passed.

## Week 2


## Week 3

### Environment Setup

- **Prompt:** "Use `uv` to initialize a new Python project for a local RAG pipeline. Please do the following: 1. Run `uv init` to set up the project and virtual environment. 2. Use `uv add` to install these exact dependencies: openai ollama pymupdf chromadb pydantic pytest python-dotenv. 3. Verify the environment is synced and ready for me to run my script."
- **Model:** Claude Code
- **Outcome:** Environment initialized and dependencies synced.

### Debugging Dependency Conflicts

- **Prompt:** "I am experiencing dependency and import errors in this project. Please analyze the code, identify the missing packages, and resolve all conflicts in my venv."
- **Model:** Claude Code
- **Outcome:** All imports in `main.py` now resolve cleanly. Root cause: `pyproject.toml` listed `google>=3.0.0` — an unrelated legacy PyPI package (an old Google-search wrapper), not part of any Google AI SDK. It shadowed the `google` namespace package, breaking `from google import genai`. Additionally, the newer `google-genai` SDK (which provides that exact import syntax) was never actually a dependency — only the older `google-generativeai` was, which exposes `google.generativeai` instead. Fix: swapped `google>=3.0.0` for `google-genai>=1.0.0` in `pyproject.toml` and ran `uv sync`. Everything else (`ollama`, `chromadb`, `pydantic`, `pymupdf`) was already correctly resolved as transitive deps of `langchain-ollama`/`langchain-chroma`.

### Hallucination Testing

- **Prompt:** "Run the existing RAG pipeline (main.py + pipeline.py) against live queries to find real cases where the LLM hallucinates, not hypothetical ones."
- **Model:** Claude Code
- **Outcome:** Confirmed the pipeline's pydantic validation only catches structural breaks — it's blind to factually fabricated content that happens to be well-typed.
