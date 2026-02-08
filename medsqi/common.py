import logging
import yaml
from typing import Dict, Any

def setup_logger(log_output_file_path:str):
    logger = logging.getLogger("medsqi")
    logger.propagate = False 

    if not logger.handlers:
        file_handler = logging.FileHandler(log_output_file_path, mode='w')
        
        formatter = logging.Formatter('%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

def load_config(config_path:str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config