"""Extract one preselected training window; never load model outputs or holdout."""
from pathlib import Path
import hashlib
import json
import pandas as pd

project = Path(__file__).resolve().parents[1]
repository = project.parents[1]
source = repository / "artifacts/nist/pilot-amendment-1-2-approved/20261016T105219Z-8ef9d07c/validate-data/canonical.parquet"
df = pd.read_parquet(source, columns=["interval_end_utc", "ac_power_kw"])
start = pd.Timestamp("2015-06-01T16:00Z")
end = pd.Timestamp("2015-06-01T18:00Z")
sample = df.loc[df.interval_end_utc.between(start, end)].copy()
sample["fixed_est"] = sample.interval_end_utc.dt.tz_convert("Etc/GMT+5")
sample["minutes_from_issue"] = (sample.interval_end_utc - pd.Timestamp("2015-06-01T17:00Z")).dt.total_seconds() / 60
assert len(sample) == 25 and sample.ac_power_kw.notna().all()
assert sample.interval_end_utc.diff().dropna().eq(pd.Timedelta(minutes=5)).all()
destination = project / "sources/nist_train_example.csv"
sample.to_csv(destination, index=False)
target = sample.loc[sample.minutes_from_issue.eq(15), "ac_power_kw"].item()
evidence = {
    "source": str(source),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "csv_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    "selection": "Fixed calendar window 2015-06-01 11:00–13:00 fixed EST, chosen before viewing its trace; no performance-based selection.",
    "split": "train", "observed_bins": len(sample),
    "interval_minutes": 5, "target_kw": target,
    "issue_utc": "2015-06-01T17:00:00Z",
    "target_interval_fixed_est": "2015-06-01 (12:10,12:15] UTC-05:00",
    "model_predictions_used": False,
    "holdout_loaded": False,
}
(project / "sources/train_example_provenance.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"rows":len(sample),"split":"train","target_kw":target,"continuous":True}))
