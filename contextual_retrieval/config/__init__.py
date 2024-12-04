import os.path as osp
import shutil
import yaml
from .schema import validate_config_item

here = osp.dirname(osp.abspath(__file__))


def update_dict(target_dict, new_dict, validate_item=None):
    """Recursively update dictionary with validation"""
    for key, value in new_dict.items():
        if validate_item:
            validate_item(key, value)
        if key not in target_dict:
            continue
        if isinstance(target_dict[key], dict) and isinstance(value, dict):
            update_dict(target_dict[key], value, validate_item=validate_item)
        else:
            target_dict[key] = value


def get_default_config():
    """Load default configuration from YAML"""
    config_file = osp.join(here, "default_config.yaml")
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Save default config to user's home directory
    user_config_file = osp.join(osp.expanduser("~"), ".contextual_retrievalrc")
    if not osp.exists(user_config_file):
        try:
            shutil.copy(config_file, user_config_file)
        except Exception:
            pass
    return config


def get_config(config_file_or_yaml=None, config_from_args=None):
    """Get configuration with overrides

    Args:
        config_file_or_yaml: Path to config file or YAML string
        config_from_args: Dictionary of config values from command line args

    Returns:
        dict: Final configuration
    """
    # 1. Load default config
    config = get_default_config()

    # 2. Update from file or YAML string
    if config_file_or_yaml is not None:
        try:
            config_from_yaml = yaml.safe_load(config_file_or_yaml)
            if not isinstance(config_from_yaml, dict):
                with open(config_from_yaml) as f:
                    config_from_yaml = yaml.safe_load(f)
            update_dict(config, config_from_yaml, validate_item=validate_config_item)
        except Exception as e:
            raise ValueError(f"Error loading config from {config_file_or_yaml}: {str(e)}")

    # 3. Update from command line args
    if config_from_args is not None:
        update_dict(config, config_from_args, validate_item=validate_config_item)

    return config


def create_config_class(config_dict):
    """Create a class from config dictionary for easy access"""

    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    setattr(self, key, create_config_class(value))
                else:
                    setattr(self, key, value)

    return Config(config_dict)