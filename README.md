# Named Entity Recognition (NER)

- This repository is used for extracting entities from text.

## Table of Content

- [Named Entity Recognition (NER)](#named-entity-recognition-ner)
  - [Table of Content](#table-of-content)
  - [Install UV Package Manager](#install-uv-package-manager)
    - [Install Specific Python Version](#install-specific-python-version)
  - [Install Dependencies](#install-dependencies)
  - [Set Up Environment Variables](#set-up-environment-variables)
  - [Extract Entities](#extract-entities)
    - [Get Help And Usage](#get-help-and-usage)
    - [Adjust The Config](#adjust-the-config)

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
uv run -m ner_extraction.get_predictions \
        --batch-size N

# e.g.
uv run -m ner_extraction.get_predictions \
        --batch-size 20
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
