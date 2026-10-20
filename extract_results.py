"""Extract benchmark results from results_*.json files into CSV.

Usage:
    python extract_results.py /path/to/exp1 /path/to/exp2 ...
    python extract_results.py /path/to/parent_dir --auto

Each experiment directory should contain results_*.json files (one per language).
The --auto flag treats each subdirectory of the given path as an experiment.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


LANG_ORDER = ["eng", "nld", "zho"]

TASKS_BY_LANG = {
    "eng": [
        "zeroshot_eng",
        "blimp_babylm_filtered",
        "hellaswag_en_mubench",
        "multiblimp_eng",
        "winogrande_en_mubench",
        "xstorycloze_en_mubench",
    ],
    "nld": [
        "zeroshot_nld",
        "blimp_nl",
        "hellaswag_nl_mubench",
        "multiblimp_nld",
        "winogrande_nl_mubench",
        "xcomps_nl",
        "xstorycloze_nl_mubench",
    ],
    "zho": [
        "zeroshot_zho",
        "hellaswag_zh_mubench",
        "winogrande_zh_mubench",
        "xcomps_zh",
        "xstorycloze_zh_mubench",
        "zhoblimp",
    ],
}


def detect_language(data):
    for lang in LANG_ORDER:
        key = f"zeroshot_{lang}"
        if key in data.get("group_subtasks", {}):
            return lang
    return None


def extract_scores(data, lang):
    results = data.get("results", {})
    tasks = TASKS_BY_LANG[lang]
    scores = {}
    raw = {}
    for task in tasks:
        val = results.get(task, {}).get("acc,none")
        if val is not None:
            raw[task] = val
            scores[task] = round(val, 3)
    avg_tasks = [t for t in tasks if not t.startswith("zeroshot_")]
    avg_vals = [raw[t] for t in avg_tasks if t in raw]
    if avg_vals:
        scores["avg"] = round(sum(avg_vals) / len(avg_vals), 3)
    return scores


def load_experiment(exp_dir):
    exp_dir = Path(exp_dir)
    json_files = sorted(exp_dir.glob("results_*.json"))
    if not json_files:
        print(f"[warn] No results_*.json in {exp_dir}", file=sys.stderr)
        return {}

    lang_scores = {}
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        lang = detect_language(data)
        if lang is None:
            print(f"[warn] Cannot detect language in {jf}", file=sys.stderr)
            continue
        lang_scores[lang] = extract_scores(data, lang)
    return lang_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", help="Experiment directories (or parent dir with --auto)")
    parser.add_argument("--auto", action="store_true",
                        help="Treat each subdirectory of the given path as an experiment")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path (default: stdout)")
    args = parser.parse_args()

    if args.auto:
        parent = Path(args.dirs[0])
        exp_dirs = sorted([d for d in parent.iterdir() if d.is_dir()])
    else:
        exp_dirs = [Path(d) for d in args.dirs]

    experiments = {}
    for d in exp_dirs:
        experiments[d.name] = load_experiment(d)

    exp_names = list(experiments.keys())

    out = open(args.output, "w", newline="") if args.output else sys.stdout
    writer = csv.writer(out)

    header0 = ["zero-shot"] + [""] * len(exp_names)
    writer.writerow(header0)
    header1 = ["task"] + exp_names
    writer.writerow(header1)

    for lang in LANG_ORDER:
        tasks = TASKS_BY_LANG[lang]
        for task in tasks:
            row = [task]
            for exp in exp_names:
                val = experiments[exp].get(lang, {}).get(task, "")
                row.append(val)
            writer.writerow(row)
        row = ["avg"]
        for exp in exp_names:
            val = experiments[exp].get(lang, {}).get("avg", "")
            row.append(val)
        writer.writerow(row)

    if args.output:
        out.close()
        print(f"Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
