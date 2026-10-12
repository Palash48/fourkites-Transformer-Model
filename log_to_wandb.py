"""Upload the training run's logged metrics and config to Weights & Biases.

Training ran offline on a laptop, so the metrics recorded during training
(outputs/train_log.csv every 100 steps, outputs/eval_log.csv every 1,000 steps) are
replayed into one W&B run at their original step numbers.

    python log_to_wandb.py --project fr-en-transformer-from-scratch
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import wandb

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--project", default="fr-en-transformer-from-scratch")
ap.add_argument("--entity", default=None)
args = ap.parse_args()

results = json.load(open(ROOT / "outputs" / "results.json"))
train_log = pd.read_csv(ROOT / "outputs" / "train_log.csv")
eval_log = pd.read_csv(ROOT / "outputs" / "eval_log.csv")

config = dict(results["config"], n_params=results["n_params"], training_pairs=results["training_pairs"],
              device=results["device"], decoding=results["best_decoding"])
run = wandb.init(project=args.project, entity=args.entity, name="transformer-256d-4x4-cpu-30k",
                 config=config, notes="Metrics recorded during training and uploaded after the run (training was offline).")

rows = {}
for r in train_log.to_dict("records"):
    rows.setdefault(int(r["step"]), {}).update({f"train/{k}": v for k, v in r.items() if k not in ("step", "epoch")})
for r in eval_log.to_dict("records"):
    rows.setdefault(int(r["step"]), {}).update({f"eval/{k}": v for k, v in r.items() if k != "step"})
for step in sorted(rows):
    wandb.log(rows[step], step=step)

run.summary.update({"dev/OVERALL_final_beam": results["dev_overall"],
                    **{f"dev_final/{r['slice']}_{m}": r[m] for r in results["dev_final"] for m in ("bleu", "chrf")}})
print(run.url)
run.finish()
