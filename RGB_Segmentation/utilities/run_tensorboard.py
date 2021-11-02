import os
from pathlib import Path

if __name__ == '__main__':
    base_path = Path(__file__).parents[1]

    log_path = os.path.join(base_path, r"logs\pytorch_logs")

    os.system(f'tensorboard --logdir {log_path}')