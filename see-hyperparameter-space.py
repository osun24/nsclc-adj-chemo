import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd


DEFAULT_STORAGE = "sqlite:///rsf_optuna_mar8_multiobj.db"
DEFAULT_STUDY_NAME = "rsf_3-8-mo-ci-rmst10"
DEFAULT_CANDIDATE_DIR = "rsf_interactions_iptw_bounded-3-8-mo-ci-rmst10-pareto"
DEFAULT_OUT_DIR = "rsf_hyperparameter_space_analysis-3-8-mo-ci-rmst10-pareto"


def load_study_trials(storage: str, study_name: str) -> pd.DataFrame:
	"""Load completed Optuna trials and flatten params/user_attrs into a DataFrame."""
	study = optuna.load_study(study_name=study_name, storage=storage)

	rows: List[Dict] = []
	for t in study.trials:
		if t.state != optuna.trial.TrialState.COMPLETE:
			continue

		row = {
			"trial_number": int(t.number),
		}

		# Handle both single-objective and multi-objective trials
		if t.values is not None and len(t.values) > 0:
			# Multi-objective trial
			row["objective_ci"] = float(t.values[0])
			row["objective_rmst_diff"] = float(t.values[1]) if len(t.values) > 1 else None
		elif t.value is not None:
			# Single-objective trial (fallback)
			row["objective_value"] = float(t.value)
		else:
			continue

		# Parameters from suggest_* calls
		for k, v in t.params.items():
			row[k] = v

		# Useful metadata stored by RSF-Optuna.py
		for attr_key in ["n_features", "k_main", "k_int", "dup_inter", "val_ci_median_10runs", "val_ci_se_10runs", "median_rmst_diff_10runs", "iqr_rmst_diff_10runs", "rmst_runs_n"]:
			if attr_key in t.user_attrs:
				row[attr_key] = t.user_attrs[attr_key]

		rows.append(row)

	if not rows:
		raise RuntimeError("No completed trials found in the study.")

	return pd.DataFrame(rows)


def parse_trial_number(file_name: str) -> Optional[int]:
	m = re.search(r"candidate_trial_(\d+)_validation_runs\.csv$", os.path.basename(file_name))
	return int(m.group(1)) if m else None


def extract_rmst_from_trial_attrs(trial_df: pd.DataFrame) -> pd.DataFrame:
	"""Extract RMST data from trial attributes (multi-objective optimization results)."""
	if "median_rmst_diff_10runs" not in trial_df.columns:
		return pd.DataFrame()
	
	rmst_cols = [
		"trial_number",
		"median_rmst_diff_10runs",
		"iqr_rmst_diff_10runs",
		"rmst_runs_n"
	]
	available_cols = [c for c in rmst_cols if c in trial_df.columns]
	
	if "trial_number" not in available_cols or "median_rmst_diff_10runs" not in available_cols:
		return pd.DataFrame()
	
	rmst_df = trial_df[available_cols].copy()
	rmst_df = rmst_df.rename(columns={
		"median_rmst_diff_10runs": "median_rmst_diff",
		"iqr_rmst_diff_10runs": "iqr_rmst_diff",
		"rmst_runs_n": "n_seed_runs"
	})
	rmst_df = rmst_df.dropna(subset=["median_rmst_diff"])
	return rmst_df


def load_median_rmst_from_candidate_runs(candidate_dir: str) -> pd.DataFrame:
	"""Build per-trial median RMST diff from saved candidate validation run CSVs (fallback)."""
	pattern = os.path.join(candidate_dir, "candidate_trial_*_validation_runs.csv")
	files = sorted(glob.glob(pattern))

	rows: List[Dict] = []
	for fp in files:
		tnum = parse_trial_number(fp)
		if tnum is None:
			continue

		try:
			df = pd.read_csv(fp)
		except Exception:
			continue

		if "rmst_diff" not in df.columns:
			continue

		vals = pd.to_numeric(df["rmst_diff"], errors="coerce").dropna().to_numpy(dtype=float)
		if vals.size == 0:
			continue

		q1, q3 = np.percentile(vals, [25, 75])
		rows.append(
			{
				"trial_number": int(tnum),
				"median_rmst_diff": float(np.median(vals)),
				"iqr_rmst_diff": float(q3 - q1),
				"n_seed_runs": int(vals.size),
			}
		)

	out = pd.DataFrame(rows)
	if out.empty:
		# Fallback to summary file if run-level files are absent.
		fallback = os.path.join(candidate_dir, "candidate_rmst_summary_final.csv")
		if os.path.exists(fallback):
			out = pd.read_csv(fallback)
	return out


def compute_correlations(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
	# Exclude non-informative columns that are derived from the same data as target
	rmst_related = ["median_rmst_diff", "iqr_rmst_diff", "n_seed_runs", "rmst_runs_n",
	                "objective_rmst_diff", "median_rmst_diff_10runs", "iqr_rmst_diff_10runs"]
	ci_related = ["objective_ci", "objective_value", "val_ci_median_10runs", "val_ci_se_10runs"]
	
	# If target is RMST-related, exclude all RMST-related columns; if CI-related, exclude CI columns
	if any(rmst in target_col.lower() for rmst in ["rmst", "alignment"]):
		exclude_list = rmst_related
	elif any(ci in target_col.lower() for ci in ["ci", "cindex", "concordance"]):
		exclude_list = ci_related
	else:
		exclude_list = []
	
	numeric_cols = [
		c
		for c in df.columns
		if c not in ["trial_number", target_col, "max_features_mode"] + exclude_list
		and pd.api.types.is_numeric_dtype(df[c])
	]

	rows = []
	for col in numeric_cols:
		tmp = df[[col, target_col]].dropna()
		if tmp[col].nunique() < 2:
			continue

		pearson = tmp[col].corr(tmp[target_col], method="pearson")
		spearman = tmp[col].corr(tmp[target_col], method="spearman")
		rows.append(
			{
				"feature": col,
				"pearson_r": float(pearson),
				"spearman_rho": float(spearman),
				"n": int(len(tmp)),
				"abs_spearman": float(abs(spearman)),
			}
		)

	corr_df = pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)
	return corr_df


def make_2d_plots(df: pd.DataFrame, corr_df: pd.DataFrame, out_dir: str, target_col: str) -> None:
	all_features = corr_df["feature"].tolist()
	if not all_features:
		return
	
	# Create multiple figures if needed, showing 12 plots per figure
	plots_per_fig = 12
	ncols = 3
	
	for fig_idx, start_idx in enumerate(range(0, len(all_features), plots_per_fig)):
		features_subset = all_features[start_idx:start_idx + plots_per_fig]
		n = len(features_subset)
		nrows = int(np.ceil(n / ncols))
		fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

		for i, feat in enumerate(features_subset):
			r, c = divmod(i, ncols)
			ax = axes[r][c]
			plot_df = df[[feat, target_col]].dropna()
			ax.scatter(plot_df[feat], plot_df[target_col], alpha=0.7)

			if plot_df[feat].nunique() > 1 and len(plot_df) >= 3:
				coeff = np.polyfit(plot_df[feat], plot_df[target_col], 1)
				xs = np.linspace(plot_df[feat].min(), plot_df[feat].max(), 100)
				ys = coeff[0] * xs + coeff[1]
				ax.plot(xs, ys, linestyle="--")

			rho = plot_df[feat].corr(plot_df[target_col], method="spearman")
			ax.set_title(f"{feat} vs {target_col}\nSpearman rho={rho:.3f}")
			ax.set_xlabel(feat)
			ax.set_ylabel(target_col)
			ax.grid(alpha=0.3)

		# Hide unused subplot axes
		for j in range(n, nrows * ncols):
			r, c = divmod(j, ncols)
			axes[r][c].axis("off")

		plt.tight_layout()
		if len(all_features) <= plots_per_fig:
			filename = "2d_scatter_all_features_vs_median_rmst.png"
		else:
			filename = f"2d_scatter_features_vs_median_rmst_part{fig_idx + 1}.png"
		plt.savefig(os.path.join(out_dir, filename), dpi=300)
		plt.close(fig)
		print(f"  Saved {filename} with {n} plots")

	# Categorical effect for max_features_mode (if present)
	if "max_features_mode" in df.columns:
		mode_df = df[["max_features_mode", target_col]].dropna()
		if not mode_df.empty:
			modes = sorted(mode_df["max_features_mode"].astype(str).unique())
			box_data = [mode_df.loc[mode_df["max_features_mode"].astype(str) == m, target_col].values for m in modes]

			plt.figure(figsize=(8, 5))
			plt.boxplot(box_data, labels=modes)
			plt.title("max_features_mode vs median RMST diff")
			plt.xlabel("max_features_mode")
			plt.ylabel(target_col)
			plt.grid(axis="y", alpha=0.3)
			plt.tight_layout()
			plt.savefig(os.path.join(out_dir, "2d_boxplot_max_features_mode_vs_median_rmst.png"), dpi=300)
			plt.close()

	# Spearman heatmap across top numerical variables + target
	heat_features = corr_df["feature"].head(10).tolist()
	if heat_features:
		cols = heat_features + [target_col]
		cmat = df[cols].corr(method="spearman")

		fig, ax = plt.subplots(figsize=(max(8, len(cols)), max(6, len(cols))))
		im = ax.imshow(cmat.values, aspect="auto")
		ax.set_xticks(range(len(cols)))
		ax.set_yticks(range(len(cols)))
		ax.set_xticklabels(cols, rotation=45, ha="right")
		ax.set_yticklabels(cols)
		ax.set_title("Spearman correlation heatmap")
		fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
		plt.tight_layout()
		plt.savefig(os.path.join(out_dir, "2d_spearman_heatmap.png"), dpi=300)
		plt.close(fig)


def make_3d_plot(df: pd.DataFrame, corr_df: pd.DataFrame, out_dir: str, target_col: str) -> Tuple[str, str]:
	sorted_feats = corr_df["feature"].tolist()
	if len(sorted_feats) < 2:
		raise RuntimeError("Need at least two numeric hyperparameters for 3D plot.")

	x_feat, y_feat = sorted_feats[0], sorted_feats[1]
	plot_df = df[[x_feat, y_feat, target_col]].dropna()

	fig = plt.figure(figsize=(9, 7))
	ax = fig.add_subplot(111, projection="3d")
	sc = ax.scatter(
		plot_df[x_feat],
		plot_df[y_feat],
		plot_df[target_col],
		c=plot_df[target_col],
		cmap="viridis",
		s=40,
		alpha=0.85,
	)
	ax.set_xlabel(x_feat)
	ax.set_ylabel(y_feat)
	ax.set_zlabel(target_col)
	ax.set_title(f"3D: {x_feat} & {y_feat} vs {target_col}")
	fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1, label=target_col)
	plt.tight_layout()
	out_path = os.path.join(out_dir, "3d_top2_hyperparameters_vs_median_rmst.png")
	plt.savefig(out_path, dpi=300)
	plt.show()

	return x_feat, y_feat


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Analyze correlation between RSF Optuna hyperparameters and median RMST from existing runs."
	)
	parser.add_argument("--storage", default=DEFAULT_STORAGE, help="Optuna storage URL, e.g. sqlite:///rsf_optuna_jan25.db")
	parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME, help="Optuna study name")
	parser.add_argument("--candidate-dir", default=DEFAULT_CANDIDATE_DIR, help="Directory containing candidate_trial_*_validation_runs.csv (optional fallback)")
	parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for plots and CSV summaries")
	parser.add_argument("--force-csv", action="store_true", help="Force loading RMST from CSV files instead of trial attributes")
	args = parser.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)

	trial_df = load_study_trials(storage=args.storage, study_name=args.study_name)
	
	# Try to get RMST from trial attributes first (multi-objective optimization)
	if not args.force_csv:
		rmst_df = extract_rmst_from_trial_attrs(trial_df)
		if not rmst_df.empty:
			print(f"Loaded RMST data from trial attributes: {len(rmst_df)} trials")
		else:
			print("No RMST data found in trial attributes, falling back to CSV files...")
	else:
		rmst_df = pd.DataFrame()
	
	# Fallback to CSV files if needed
	if rmst_df.empty:
		rmst_df = load_median_rmst_from_candidate_runs(candidate_dir=args.candidate_dir)
		if rmst_df.empty:
			raise RuntimeError(
				"No RMST data found in trial attributes or CSV files. "
				"Check study name/storage and candidate directory."
			)
		print(f"Loaded RMST data from CSV files: {len(rmst_df)} trials")

	merged = trial_df.merge(rmst_df, on="trial_number", how="inner")
	if merged.empty:
		raise RuntimeError(
			"Join between Optuna trials and RMST summaries produced 0 rows. "
			"Check that trials have RMST data available."
		)

	target_col = "median_rmst_diff"
	corr_df = compute_correlations(merged, target_col=target_col)
	if corr_df.empty:
		raise RuntimeError("No numeric hyperparameters available for correlation analysis.")

	merged.to_csv(os.path.join(args.out_dir, "trial_hyperparams_with_median_rmst.csv"), index=False)
	corr_df.to_csv(os.path.join(args.out_dir, "correlation_with_median_rmst.csv"), index=False)

	make_2d_plots(merged, corr_df, out_dir=args.out_dir, target_col=target_col)
	x_feat, y_feat = make_3d_plot(merged, corr_df, out_dir=args.out_dir, target_col=target_col)

	print("\nAnalysis complete.")
	print(f"Rows analyzed: {len(merged)}")
	print(f"Output directory: {os.path.abspath(args.out_dir)}")
	print("Top correlations (by |Spearman rho|):")
	print(corr_df[["feature", "pearson_r", "spearman_rho", "n"]].head(10).to_string(index=False))
	print(f"3D plot axes: x={x_feat}, y={y_feat}, z={target_col}")


if __name__ == "__main__":
	main()
