import os
from pathlib import Path
import re


def get_current_model_checkpoint(root_dir=os.path.join(Path(__file__).parents[1], "logs\\pytorch_logs"), epoch_num=None):
    print(f"Root directory = {root_dir}")
    # Get Version # from logs directory
    version_list = os.listdir(root_dir)
    version_list = [int(re.findall(r'\d{1,3}', vs)[-1]) for vs in version_list]
    version = max(version_list)
    # Check Epoch from Version subdirectory
    version_path = os.path.join(root_dir, f"version{version}\\checkpoints")

    if epoch_num is None:
        max_epoch = 0
        for item in os.listdir(version_path):
            output = re.findall(r'\d{1,}', item)
            v_num = int(output[0])
            e_num = int(output[1])

            if e_num > max_epoch:
                max_epoch = e_num
    else:
        max_epoch = epoch_num
    print(f"Model => v{version}_model_epoch{max_epoch}.ckpt")
    model_path = os.path.join(version_path, f"v{version}_model_epoch{max_epoch}.ckpt")
    return model_path


if __name__ == '__main__':

    print(os.path.join(Path(__file__).parents[1], "logs\\pytorch_logs"))