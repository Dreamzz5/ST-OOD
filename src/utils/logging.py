import os
import sys
import logging


def check_epoch_100(log_file):
    with open(log_file, "r") as f:
        for line in f:
            if "Epoch: 100," in line or "Early stop" in line:
                return True
    return False


def get_logger(log_dir, name, log_filename, level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # log_file = os.path.join(log_dir, log_filename)
    # if check_epoch_100(log_file):
    #     print("Training Finish")
    #     sys.exit(1)
    # else:
    #     print("Epoch has not been reached yet.")

    file_formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(file_formatter)

    console_formatter = logging.Formatter("%(asctime)s - %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    print("Log directory:", log_dir)

    return logger
