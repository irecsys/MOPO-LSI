import argparse

from utils.quick_start import run_optimization


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='user.yaml')

    args, _ = parser.parse_known_args()

    config_file_list = args.config.strip().split(' ') if args.config else None
    run_optimization(config_file_list=config_file_list)