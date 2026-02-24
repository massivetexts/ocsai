This file provides guidance to coding agents when working with code in this repository.

## Overview

OCSAI (Open Creativity Scoring with AI) is a Python library for training and using automated originality scoring models based on Large Language Models. It's designed for evaluating divergent thinking and creativity in responses, particularly for educational and psychological research.

## Common Development Commands

### Installation and Setup
- **Development Setup**: `uv sync` - Creates virtual environment in `./.venv` and installs all dependencies
- **Standard Installation**: `pip install git+https://www.github.com/massivetexts/ocsai.git`

### Testing
- **Run Tests**: `pytest`
- **Run Specific Test**: `pytest tests/path/to/test_file.py::test_function_name`
- **Run Tests with Docker**: `pytest tests/cache/test_ocsai_redis_cache.py` (requires Docker for Redis tests)

### Code Quality
- **Linting**: `flake8` - Max line length is set to 120 characters (configured in `.flake8`)

### Building and Distribution
- **Build Package**: Uses setuptools with `pyproject.toml` configuration
- **Package Management**: Project uses UV for dependency management (`uv.lock`)

## Architecture

### Core Components

The project follows a modular, inheritance-based architecture with these key components:

1. **Scorers** (`ocsai/inference/`): Handle the actual scoring logic
   - `Base_Scorer`: Abstract base class defining the scorer interface
   - `Chat_Scorer`: For chat-based models (GPT-3.5-turbo, GPT-4)
   - `Classic_Scorer`: For legacy completion models
   - All scorers support async operations with configurable concurrency

2. **Prompters** (`ocsai/prompter/`): Generate prompts for different OCSAI versions
   - `Base_Prompter`: Abstract base defining the prompter interface
   - `Ocsai1_Prompter`: Original scoring approach
   - `Ocsai1p5_Prompter`: Enhanced version with improved prompting
   - `Ocsai2_Prompter`: Latest version with advanced features
   - Prompters handle prompt crafting and response parsing

3. **LLM Interfaces** (`ocsai/llm_interface/`): Abstraction layer for different LLM providers
   - `LLM_Base_Interface`: Abstract base class for all interfaces
   - `OpenAI_Chat_Interface`: For OpenAI chat models
   - `Anthropic_Interface`: For Claude models
   - Interfaces standardize responses across different providers

4. **Caching** (`ocsai/cache/`): Multiple caching backends for efficient inference
   - `Ocsai_Cache`: Base cache interface
   - `Ocsai_Redis_Cache`: Redis-based caching (requires Redis server)
   - `Ocsai_Parquet_Cache`: File-based caching using Parquet format
   - `Ocsai_PostReg_Cache`: PostReg caching implementation

5. **Data Processing** (`ocsai/data/`): Tools for loading and preprocessing datasets
   - `loader.py`: Dataset loading utilities
   - `preprocess.py`: Data preprocessing functions
   - Handles various creativity study datasets

### Key Design Patterns

- **Async Support**: Scorers use `asyncio` and `nest_asyncio` for concurrent batch processing
- **Pluggable Architecture**: Components (prompters, interfaces, caches) are interchangeable via dependency injection
- **Model Dictionary Pattern**: Models specified as `{"model_name": "model_id"}` dictionaries
- **Fine-tuned Models**: System primarily uses OpenAI fine-tuned models (format: `ft:base-model:org::id`)
- **Standardized Response Types**: All LLM responses converted to `StandardAIResponse` format

### Testing Strategy

- Unit tests for each major component in `tests/`
- Tests use pytest fixtures and mock fine-tuned model IDs
- Redis cache tests require Docker (uses `pytest-docker-tools`)
- Test files mirror source structure (e.g., `tests/inference/test_chat_scorer.py`)

### Important Notebooks

The `notebooks/` directory contains research and development workflows:
- `ocsai2-dataprep/`: Data preparation and training workflows
  - `cleanDatasets.ipynb`: Dataset cleaning and preprocessing
  - `trainOcsai2.ipynb`: Model training pipeline
  - `crossLingualClustering.ipynb`: Cross-lingual analysis
- `evaluation/`: Model evaluation notebooks
  - `LogProbsOcsai1.ipynb`: Log probability analysis
  - `Ocsai1_5_evaluation.ipynb`: OCSAI 1.5 performance evaluation
- `synthetic-augmentation/`: Data augmentation techniques

### Environment Variables

The system expects these environment variables:
- `OPENAI_API_KEY`: For OpenAI models
- `ANTHROPIC_API_KEY`: For Claude models
- Redis connection details (if using Redis cache):
  - `REDIS_HOST`
  - `REDIS_PORT`
  - `REDIS_PASSWORD`

### Data Organization

Training data in `data/` follows this structure:
- `raw/`: Original datasets from creativity studies
- `datasets/`: Processed CSV files ready for training
- `training/`: JSONL files formatted for fine-tuning
  - `gpt/`: OpenAI fine-tuning format
  - `ocsai2/`: OCSAI 2 training data
- `results/`: Model evaluation outputs
- `translation/`: Cross-lingual translation data
- `codebook/`: Scoring codebooks and guidelines

### Development Workflow

1. **Adding New Features**: Create feature in appropriate module, add tests, update relevant prompter
2. **Training Models**: Use notebooks in `ocsai2-dataprep/` to prepare data and train
3. **Evaluation**: Run evaluation notebooks to assess model performance
4. **Caching**: Configure cache backend for production use (Redis recommended)