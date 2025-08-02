# Named Entity Recognition (NER)

- This repository is used for extracting entities from text.

## Table of Content

- [Named Entity Recognition (NER)](#named-entity-recognition-ner)
  - [Table of Content](#table-of-content)
  - [Technologies Used](#technologies-used)
    - [Programming Language](#programming-language)
    - [Core Libraries](#core-libraries)
    - [LLM Integration](#llm-integration)
    - [Data Processing](#data-processing)
    - [Database](#database)
    - [Containerization](#containerization)
    - [Development Tools](#development-tools)
  - [Install UV Package Manager](#install-uv-package-manager)
    - [Install Specific Python Version](#install-specific-python-version)
  - [Install Dependencies](#install-dependencies)
  - [Set Up Environment Variables](#set-up-environment-variables)
  - [Extract Entities](#extract-entities)
    - [Get Help And Usage](#get-help-and-usage)
    - [Format Extracted Data](#format-extracted-data)
    - [Adjust The Config](#adjust-the-config)

## Technologies Used

This project leverages the following technologies:

### Programming Language

- **Python 3.12+** - Modern Python version for enhanced typing and performance

### Core Libraries

- **Instructor** - LLM guidance framework for structured outputs
- **Polars** - High-performance DataFrame library for data manipulation
- **SQLAlchemy** - SQL toolkit and ORM for database operations
- **Pydantic** - Data validation and settings management
- **Jinja2** - Templating engine for prompt engineering

### LLM Integration

- **OpenAI API Client** - For interfacing with OpenAI compatible models
- **OpenRouter** - API aggregation for multiple LLM providers
- **RunPod** - GPU infrastructure for running self-hosted models

### Data Processing

- **HTTPX** - Asynchronous HTTP client for API calls
- **Tenacity** - Retry library for robust API interactions
- **OmegaConf** - YAML configuration management
- **Rich** - Terminal formatting and visualization
- **TQDM** - Progress bar for batch processing operations

### Database

- **SQLite** - Lightweight database for storing entity extraction results

### Containerization

- **Docker** - For containerized deployment

### Development Tools

- **UV** - Fast Python package manager written in Rust
- **Black** - Code formatting
- **Ruff** - Fast Python linter

## Install UV Package Manager

- `UV` is an extremely fast Python package and project manager, written in Rust.

```sh
# On Mac/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install Specific Python Version

```sh
uv python install 3.12
```

## Install Dependencies

```sh
uv sync
```

## Set Up Environment Variables

```sh
# 1.) Create .env
cp .env.example .env

# 2.) Edit .env with your environment variables
```

## Extract Entities

### Get Help And Usage

```sh
# Help
uv run -m ner_extraction.get_predictions --help

# usage
uv run -m ner_extraction.get_predictions
```

### Format Extracted Data

- After extracting entities, the data is stored in a SQLite database.
- The data is extracted as a CSV file and formatted into JSONL using:

```sh
uv run format_data.py
```

### Adjust The Config

- The config file can be found in the NER extraction configuration file: [ner_extraction/config/config.yaml](ner_extraction/config/config.yaml)

```yaml
app_config:
  data:
    data_path: data/examples.jsonl
    download_data_path: data/ner_data.jsonl

  database:
    db_path: sqlite:///ner_extraction.db

  ---
  # Other configs

```
