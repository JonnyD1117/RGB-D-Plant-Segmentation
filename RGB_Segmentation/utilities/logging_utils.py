import os
import re


def setup_version_logs(base_path, log_path):

    version_list = os.listdir(log_path)

    output = [int(re.findall(r'\d+', ver)[0]) for ver in version_list]

    if output == []:
        version_num = 0
    else:
        max_ver = max(output)
        version_num = max_ver + 1

    version_dir = os.path.join(base_path, f'logs\\pytorch_logs\\version{version_num}')
    checkpoint_dir = os.path.join(version_dir, 'checkpoints')

    # Create Log Dirs If they don't exist
    if os.path.exists(version_dir):
        pass
    else:
        os.mkdir(version_dir)

    if os.path.exists(checkpoint_dir):
        pass
    else:
        os.mkdir(checkpoint_dir)

    return version_num, version_dir, checkpoint_dir
