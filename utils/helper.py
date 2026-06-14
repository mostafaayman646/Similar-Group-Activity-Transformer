import time
import os
import yaml
import logging
import torch

def load_config(config_path: str):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def setup_logger(log_file):
    # Create a logger that writes to both the console and a file
    logger = logging.getLogger("Football_Sim_Play_Encoder")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def save_checkpoint(state, is_best, checkpoint_dir, filename="latest_checkpoint.pth"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, save_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)
