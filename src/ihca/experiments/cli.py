"""Command line for cohort preparation, experiment reports and the research interface."""

import argparse
import json
from pathlib import Path

import torch

from ihca.data import load_data
from . import analyses
from .attribution import explain
from .cohort import assemble
from .comparators import compare, frozen_comparators
from .matching import match
from .risksets import ratios
from .study import Study, aligned_metadata, new_result_directory

ANALYSES = {
    "compare": compare,
    "ablation": analyses.ablation,
    "shap": explain,
    "subgroups": analyses.subgroups,
    "ratios": ratios,
    "prevalence": analyses.prevalence,
    "describe": analyses.describe,
}


def parser():
    root = argparse.ArgumentParser(
        prog="ihca-experiments", description="IHCA cohort analyses and model comparison"
    )
    commands = root.add_subparsers(dest="command", required=True)
    assembly = commands.add_parser("assemble", help="Build 1:1 risk sets using hospital transfer histories")
    for field in ("encounters", "departments", "ecgs", "output"):
        assembly.add_argument("--" + field, required=True)
    assembly.add_argument("--xml-map", help="Vendor mapping used for waveform quality checks")
    pairing = commands.add_parser(
        "match", help="Construct 1:1 risk-set matches from eligible hospital exports"
    )
    pairing.add_argument("--cases", required=True)
    pairing.add_argument("--candidates", required=True)
    pairing.add_argument("--output", required=True)
    pairing.add_argument("--seed", type=int, default=42)
    run = commands.add_parser("run", help="Compute the study analyses and figures")
    run.add_argument("--study", default="configs/experiments.json")
    run.add_argument("--output", required=True)
    run.add_argument("--only", nargs="+", choices=list(ANALYSES))
    score = commands.add_parser(
        "score-comparators", help="Score a labeled external cohort with saved rules and trees"
    )
    for field in ("models", "data", "metadata", "output"):
        score.add_argument("--" + field, required=True)
    serve = commands.add_parser("serve", help="Open the local research interface on a loopback address")
    serve.add_argument("--study", default="configs/experiments.json")
    serve.add_argument("--port", type=int, default=8050)
    return root


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.command == "assemble":
            print(json.dumps(assemble(args.encounters, args.departments, args.ecgs,
                                      args.output, args.xml_map), indent=2))
            return
        if args.command == "match":
            print(f"Matched records: {match(args.cases, args.candidates, args.output, args.seed)}")
            return
        if args.command == "score-comparators":
            data = load_data(args.data)
            if str(data["cohort"]) != "external":
                raise ValueError("A labeled external dataset is required")
            meta = aligned_metadata(data, args.metadata)
            tables = frozen_comparators(args.models, data, meta)
            with new_result_directory(args.output) as out:
                for name, table in tables.items():
                    table.to_csv(out / (name + ".csv"))
            return
        study = Study(args.study)
        torch.set_num_threads(int(study.cfg["threads"]))
        if args.command == "serve":
            from .web import create_app

            create_app(study).run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
            return
        with new_result_directory(args.output) as out:
            for name in dict.fromkeys(args.only or ANALYSES):
                dest = out / name
                dest.mkdir()
                print(f"Running {name}", flush=True)
                ANALYSES[name](study, dest)
        print(f"Saved analyses to {Path(args.output).resolve()}")
    except (OSError, KeyError, ValueError, TypeError) as exc:
        cli.exit(2, f"ihca-experiments: {exc}\n")


if __name__ == "__main__":
    main()
