"""Experiment tracking: always a local CSV (runs/<name>/metrics.csv), plus W&B or MLflow.

backend: "wandb" | "mlflow" | "csv". If the chosen backend is not installed or not logged in,
training continues with CSV only and says so (it never blocks a run).

W&B credentials, in order: WANDB_API_KEY, the key file at $WANDB_KEY_FILE
(default /workspace/.secrets/wandb_key, on the network volume), or a prior `wandb login`.
"""
import csv
import os
from pathlib import Path


class Tracker:
    def __init__(self, backend: str, run_dir: Path, run_name: str, config: dict, project: str = "sonar-glasses",
                 resume_id: str = None):
        self.run_dir = Path(run_dir)
        self.csv_path = self.run_dir / "metrics.csv"
        self.backend = "csv"
        self.run_id = None
        self._wandb = self._mlflow = None
        if backend == "wandb":
            key_file = Path(os.environ.get("WANDB_KEY_FILE", "/workspace/.secrets/wandb_key"))
            if not os.environ.get("WANDB_API_KEY") and key_file.exists():
                os.environ["WANDB_API_KEY"] = key_file.read_text().strip()  # never printed or logged
            try:
                import wandb
                if not (os.environ.get("WANDB_API_KEY") or wandb.api.api_key):
                    raise RuntimeError("not logged in (run `wandb login` or set WANDB_API_KEY)")
                run = wandb.init(project=project, name=run_name, config=config, dir=str(self.run_dir),
                                 id=resume_id, resume="allow")
                self._wandb, self.backend, self.run_id = wandb, "wandb", run.id
            except Exception as e:  # noqa: BLE001
                print(f"[tracking] W&B unavailable ({e}); logging to CSV only", flush=True)
        elif backend == "mlflow":
            try:
                import mlflow
                mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"file:{self.run_dir.parent / 'mlruns'}"))
                mlflow.set_experiment(project)
                run = mlflow.start_run(run_id=resume_id, run_name=None if resume_id else run_name)
                if not resume_id:
                    mlflow.log_params(_flatten(config))
                self._mlflow, self.backend, self.run_id = mlflow, "mlflow", run.info.run_id
            except Exception as e:  # noqa: BLE001
                print(f"[tracking] MLflow unavailable ({e}); logging to CSV only", flush=True)
        print(f"[tracking] backend: {self.backend}" + (f" (run id {self.run_id})" if self.run_id else ""), flush=True)

    def log(self, metrics: dict, step: int):
        metrics = {k: v for k, v in metrics.items() if v is not None}
        new = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as fh:
            w = csv.writer(fh)
            for k, v in metrics.items():
                if new:
                    w.writerow(["step", "key", "value"])
                    new = False
                w.writerow([step, k, v])
        if self._wandb:
            self._wandb.log(metrics, step=step)
        if self._mlflow:
            self._mlflow.log_metrics({k.replace("/", "_"): float(v) for k, v in metrics.items()}, step=step)

    def finish(self):
        if self._wandb:
            self._wandb.finish()
        if self._mlflow:
            self._mlflow.end_run()


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out
