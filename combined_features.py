from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parent

TXT_SOURCES = [
	ROOT / "deepsurv_features_used.txt",
	ROOT / "rsf_interactions_iptw_bounded-11-22-1SE" / "features_used.txt",
	ROOT / "xgb_cox_interactions_iptw_bounded-1-25-1SE" / "features_used.txt",
]

EN_FEATURES = ROOT / "EN_features.csv"
OUTPUT = ROOT / "combined_features.txt"


def read_txt_features(path: Path) -> list[str]:
	if not path.exists():
		return []
	return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_en_features(path: Path) -> list[str]:
	if not path.exists():
		return []
	features: list[str] = []
	with path.open(newline="") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			value = (row.get("feature") or "").strip()
			if value:
				features.append(value)
	return features


def combine_unique(*lists: list[str]) -> list[str]:
	seen = OrderedDict()
	for items in lists:
		for item in items:
			if item not in seen:
				seen[item] = None
	return list(seen.keys())


def main() -> None:
	features: list[str] = []
	for source in TXT_SOURCES:
		features.extend(read_txt_features(source))
	features.extend(read_en_features(EN_FEATURES))

	unique_features = combine_unique(features)
	OUTPUT.write_text("\n".join(unique_features) + "\n")
	print(f"Wrote {len(unique_features)} unique features to {OUTPUT}")


if __name__ == "__main__":
	main()
