"""Command-line preparation, model development, evaluation and inference."""

import argparse
import json
from pathlib import Path

from .config import read_config
from .data import prepare_data
from .engine import evaluate, predict, train_cv
from .schema import feature_schema


def build_parser():
    parser = argparse.ArgumentParser(prog="ihca", description="Multimodal IHCA discrimination")
    sub = parser.add_subparsers(dest="command", required=True)
    schema = sub.add_parser("schema", help="Print the ordered data schema")
    schema.add_argument("--output")
    prepare = sub.add_parser("prepare", help="Convert an ECG and clinical manifest to a dataset")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--cohort", choices=["development", "external", "prediction"], required=True)
    prepare.add_argument("--xml-map", help="JSON mapping of vendor XML paths and calibration fields")
    train = sub.add_parser("train", help="Fit grouped outer folds with inner validation")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--config", default="configs/paper.json")
    train.add_argument("--mode", choices=["full", "clinical", "structured", "waveform"])
    train.add_argument("--device", default="auto")
    for command in ("evaluate", "predict"):
        entry = sub.add_parser(command, help="Apply the frozen fold-model ensemble")
        entry.add_argument("--data", required=True)
        entry.add_argument("--model-dir", required=True)
        entry.add_argument("--output", required=True)
        entry.add_argument("--batch-size", type=int, default=64)
        entry.add_argument("--threads", type=int, default=4)
        entry.add_argument("--device", default="auto")
        if command == "evaluate":
            entry.add_argument("--bootstrap", type=int, default=2000)
            entry.add_argument("--seed", type=int, default=1234)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "schema":
            result = feature_schema()
            if args.output:
                path = Path(args.output)
                if path.exists():
                    raise FileExistsError(f"Schema output already exists: {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        elif args.command == "prepare":
            result = prepare_data(args.manifest, args.output, args.cohort, args.xml_map)
        elif args.command == "train":
            model, training = read_config(args.config)
            if args.mode:
                model.mode = args.mode
            result = train_cv(args.data, args.output, model, training, args.device)
        else:
            if args.batch_size < 1 or args.threads < 1:
                raise ValueError("batch-size and threads must be positive")
            if args.command == "evaluate":
                if args.bootstrap < 0:
                    raise ValueError("bootstrap must be nonnegative")
                result = evaluate(args.data, args.model_dir, args.output, args.bootstrap, args.seed,
                                  args.batch_size, args.device, args.threads)
            else:
                result = predict(args.data, args.model_dir, args.output, args.batch_size, args.device, args.threads)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, FloatingPointError) as error:
        parser.exit(2, f"ihca: {error}\n")


if __name__ == "__main__":
    main()
