# schema.py
# Description: Configuration schema for the model server

def validate_config_item(key, value):
    """Configuration validation rules"""
    if key == "port" and not isinstance(value, int):
        raise ValueError(f"Port must be an integer, got: {value}")

    if key == "model_id" and not isinstance(value, str):
        raise ValueError(f"Model ID must be a string, got: {value}")

    if key == "dimension" and not isinstance(value, int):
        raise ValueError(f"Embedding dimension must be an integer, got: {value}")

    if key == "max_tokens" and not isinstance(value, int):
        raise ValueError(f"Max tokens must be an integer, got: {value}")

    if key == "temperature" and not (isinstance(value, (int, float)) and 0 <= value <= 1):
        raise ValueError(f"Temperature must be a number between 0 and 1, got: {value}")