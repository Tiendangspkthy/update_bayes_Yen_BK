SHELL = /bin/bash
PYTHON_VERSION = 3.11
MAIN_FOLDER = pedago_ml
TEST_FOLDER = tests

# Environment
venv:
	uv venv --python ${PYTHON_VERSION} && \
	uv pip install -e ".[dev]" && \
	uv run pre-commit install

# Style
style:
	uv run black ./${MAIN_FOLDER}/ ./${TEST_FOLDER}/
	uv run flake8 ./${MAIN_FOLDER}/ ./${TEST_FOLDER}/
	uv run isort -rc ./${MAIN_FOLDER}/

test:
	uv run flake8 ./${MAIN_FOLDER}/
	uv run mypy ./${MAIN_FOLDER}/
	CUDA_VISIBLE_DEVICES="" uv run pytest -s --durations=0 --disable-warnings ${TEST_FOLDER}/
	uv run pylint ./${MAIN_FOLDER}/
