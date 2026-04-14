import json
import argparse

from trainers.cil_trainer import train

def main():
    args = build_parser().parse_args()
    params = load_json(args.config)

    # JSON provides base config; explicit CLI args override JSON
    cli = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}
    merged = {**params, **cli}

    train(merged)

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--start-at-task', type=int, default=None, dest='start_at_task')
    parser.add_argument('--stop-at-task',  type=int, default=None, dest='stop_at_task')
    parser.add_argument('--start-epoch',   type=int, default=None, dest='start_epoch')
    parser.add_argument('--stop-epoch',    type=int, default=None, dest='stop_epoch')
    return parser

def load_json(settings_path):
    with open(settings_path) as f:
        return json.load(f)

if __name__ == '__main__':
    main()
