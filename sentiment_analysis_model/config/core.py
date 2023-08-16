# Path setup, and access the config.yml file, datasets folder & trained models
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel
from strictyaml import YAML, load

import sentiment_analysis_model

# Project Directories
PACKAGE_ROOT = Path(sentiment_analysis_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
#print(CONFIG_FILE_PATH)

DATASET_DIR = PACKAGE_ROOT / "datasets/data"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

class AppConfig(BaseModel):
    """
    Application-level config.
    """
    
    package_name: str
    dataset_file_path: str
    train_path: str
    validation_path: str
    test_path: str
    model_name: str
    model_save_file: str
    not_required_features: str
    tokenization_save_file: str
    clean_up_words: str
    load_existing_tokenizer: bool
    save_tokenizer: bool

class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """
        
    batch_size: int
    maxlen: int
    test_size: float
    validation_size: float

    random_state: int
    epochs: int
    optimizer: str
    loss: str
    accuracy_metric: str
    verbose: int
    earlystop: int
    monitor: str
    save_best_only: bool
#    label_mappings: Dict[int, str]


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
        
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config = AppConfig(**parsed_config.data),
        model_config = ModelConfig(**parsed_config.data),
    )

    return _config

# get from environment variables

config_filename = os.environ.get('CONFIG_FILEPATH', "config/config.yml")
    
print("Reading config-file : ", config_filename)    

config = create_and_validate_config()
