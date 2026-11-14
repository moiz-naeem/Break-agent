from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from railway.sim.model import Scenario
from railway.util import canonical_json

from .generator import generate
from .presets import PRESETS
from .seeds import candidate_seeds


def _parser():
    parser = argparse.ArgumentParser(prog="gen")
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("--config", required=True)
    sample.add_argument("--seed", required=True, type=int)
    sample.add_argument("--out", required=True)
    describe = commands.add_parser("describe")
    describe.add_argument("scenario")
    candidates = commands.add_parser("candidates")
    candidates.add_argument("--set", required=True, choices=("dev", "train", "heldout"))
    candidates.add_argument("--count", type=int, default=10)
    accept = commands.add_parser("accept", help="scan candidate seeds through the T06 seed filter")
    accept.add_argument("--config", required=True)
    accept.add_argument("--set", required=True, choices=("dev", "train", "heldout"))
    accept.add_argument("--count", required=True, type=int)
    # T07: its options are parsed by railway.harbor.export (`gen export -h`)
    commands.add_parser("export", help="export accepted scenarios as Harbor tasks", add_help=False)
    return parser


def main(argv=None):
    parser = _parser()
    args, extra = parser.parse_known_args(argv)
    if args.command == "export":
        from railway.harbor.export import main as export_main

        return export_main(extra)
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if args.command == "sample":
        if args.config not in PRESETS:
            valid = ", ".join(sorted(PRESETS))
            raise SystemExit(f"unknown preset {args.config!r}; valid presets: {valid}")
        scenario = generate(PRESETS[args.config], args.seed)
        Path(args.out).write_text(canonical_json(scenario.model_dump(mode="json")) + "\n", encoding="utf-8")
        return 0
    if args.command == "describe":
        scenario = Scenario.model_validate(json.loads(Path(args.scenario).read_text(encoding="utf-8")))
        kinds = Counter(event.kind for event in scenario.events)
        counts = Counter(train.type for train in scenario.trains)
        print(
            f"{scenario.preset}: {len(scenario.network.stations)} stations; "
            f"{len(scenario.trains)} trains ({', '.join(f'{key}={counts[key]}' for key in sorted(counts))}); "
            f"events ({', '.join(f'{key}={kinds[key]}' for key in sorted(kinds)) or 'none'})"
        )
        return 0
    if args.command == "accept":
        if args.config not in PRESETS or args.count <= 0:
            raise SystemExit(f"need a preset in {sorted(PRESETS)} and a positive --count")
        from railway.ref.seed_filter import accepted_seeds

        entries = accepted_seeds(args.config, args.set, args.count)
        print(f"accepted {len(entries)} -> seeds/{args.config}-{args.set}.json")
        return 0
    if args.command == "candidates":
        if args.count < 0:
            raise SystemExit("--count must be nonnegative")
        print("\n".join(str(seed) for seed in candidate_seeds(args.set, args.count)))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
