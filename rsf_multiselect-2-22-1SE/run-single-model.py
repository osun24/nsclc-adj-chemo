import os
import random
import inspect
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import restricted_mean_survival_time
from threadpoolctl import threadpool_limits

import optuna


warnings.filterwarnings(
	"ignore",
	message="Ties in event time detected; using efron's method to handle ties.",
)

np.random.seed(42)
random.seed(42)


# ---------- Config ----------
TRIAL_NUMBER = 50
SEED_LIST = list(range(1, 31))  # original random seeds from RSF-Optuna.py

TRAIN_CSV = "affyfRMATrain.csv"
VALID_CSV = "affyfRMAValidation.csv"
TEST_CSV = "affyfRMATest.csv"

GENES_CSV = "LOOCV_Genes2.csv"
OPTUNA_STORAGE = "sqlite:///rsf_optuna_jan25.db"
STUDY_NAME = "rsf_jan_25_1se"

OUT_DIR = "rsf_multiselect-2-22-1SE/trial_50_test_eval"
os.makedirs(OUT_DIR, exist_ok=True)

CLINICAL_VARS = [
	"Adjuvant Chemo",
	"Age",
	"IS_MALE",
	"Stage_IA",
	"Stage_IB",
	"Stage_II",
	"Stage_III",
	"Histology_Adenocarcinoma",
	"Histology_Large Cell Carcinoma",
	"Histology_Squamous Cell Carcinoma",
	"Race_African American",
	"Race_Asian",
	"Race_Caucasian",
	"Race_Native Hawaiian or Other Pacific Islander",
	"Race_Unknown",
	"Smoked?_No",
	"Smoked?_Unknown",
	"Smoked?_Yes",
]


def load_genes_list(genes_csv: str):
	g = pd.read_csv(genes_csv)
	g["Prop"] = pd.to_numeric(g["Prop"], errors="coerce").fillna(0)
	return g.loc[g["Prop"] == 1, "Gene"].astype(str).tolist()


def preprocess_split(df, clinical_vars, gene_names):
	if "Adjuvant Chemo" in df.columns:
		df["Adjuvant Chemo"] = df["Adjuvant Chemo"].map({"OBS": 0, "ACT": 1})
	for col in ["Adjuvant Chemo", "IS_MALE"]:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
	keep_cols = [c for c in clinical_vars if c in df.columns] + [g for g in gene_names if g in df.columns]
	cols = ["OS_STATUS", "OS_MONTHS"] + keep_cols
	return df[cols].copy()


def rank_genes_univariate(train_df, gene_cols):
	y = Surv.from_arrays(
		event=train_df["OS_STATUS"].astype(bool).values,
		time=train_df["OS_MONTHS"].values.astype(float),
	)
	ranks = []
	for g in gene_cols:
		Xg = train_df[[g]].to_numpy(dtype=np.float64)
		try:
			model = CoxPHSurvivalAnalysis(alpha=1e-12)
			model.fit(Xg, y)
			pred = model.predict(Xg)
			ci = concordance_index_censored(y["event"], y["time"], pred)[0]
			ranks.append((g, float(ci)))
		except Exception:
			ranks.append((g, 0.5))
	ranks.sort(key=lambda z: z[1], reverse=True)
	return [g for g, _ in ranks]


def build_features_with_interactions(df, main_genes, inter_genes, act_col="Adjuvant Chemo", dup_inter=1, clin_cols=None):
	if clin_cols is None:
		clin_cols = CLINICAL_VARS

	base_cols = list(clin_cols) + list(main_genes)
	X_base = df[base_cols].to_numpy(dtype=np.float64)
	A = df[act_col].to_numpy(dtype=np.float64).reshape(-1, 1)

	names = list(base_cols)
	blocks = [X_base]

	if len(inter_genes) > 0:
		X_int = df[list(inter_genes)].to_numpy(dtype=np.float64) * A
		names_int = [f"{g}*ACT" for g in inter_genes]
		blocks.append(X_int)
		names += names_int

		if int(dup_inter) > 1:
			for d in range(1, int(dup_inter)):
				blocks.append(X_int.copy())
				names += [f"{g}*ACT#dup{d}" for g in inter_genes]

	X = np.concatenate(blocks, axis=1) if len(blocks) > 1 else X_base
	return X, names


def compute_iptw(df, covariate_cols, act_col="Adjuvant Chemo", ps_clip=(0.05, 0.95), w_clip=(0.1, 10.0), ref_prev=None, model=None):
	A = df[act_col].astype(int).values
	X = df[covariate_cols].astype(float).values

	if model is None:
		model = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
		model.fit(X, A)

	ps = model.predict_proba(X)[:, 1]
	ps = np.clip(ps, ps_clip[0], ps_clip[1])

	if ref_prev is None:
		ref_prev = A.mean()

	w = np.where(A == 1, ref_prev / ps, (1 - ref_prev) / (1 - ps))
	w = np.clip(w, w_clip[0], w_clip[1])
	return w.astype(np.float64), model, float(ref_prev)


def cindex(pred, time, event):
	return float(concordance_index_censored(event.astype(bool), time.astype(float), pred)[0])


def make_rsf(**params):
	sig = inspect.signature(RandomSurvivalForest.__init__)
	allowed = set(sig.parameters.keys())
	allowed.discard("self")
	filtered = {k: v for k, v in params.items() if k in allowed}
	return RandomSurvivalForest(**filtered)


def compare_treatment_recommendation_km_rsf(model, df, genes_main, genes_inter, dup_inter, clin_cols, time_col="OS_MONTHS", event_col="OS_STATUS", out_png=None, weightings = None, p = None, q = None):
	df = df.copy()
	df["Adjuvant Chemo"] = df["Adjuvant Chemo"].astype(int)

	df_treated = df.copy()
	df_treated["Adjuvant Chemo"] = 1

	df_untreated = df.copy()
	df_untreated["Adjuvant Chemo"] = 0

	X_treated, _ = build_features_with_interactions(df_treated, genes_main, genes_inter, dup_inter=dup_inter, clin_cols=clin_cols)
	X_untreated, _ = build_features_with_interactions(df_untreated, genes_main, genes_inter, dup_inter=dup_inter, clin_cols=clin_cols)

	with threadpool_limits(limits=1):
		risk_treated = model.predict(X_treated.astype(np.float64))
		risk_untreated = model.predict(X_untreated.astype(np.float64))

	model_rec = np.where(risk_treated < risk_untreated, 1, 0)
	actual = df["Adjuvant Chemo"].to_numpy(int)
	alignment = actual == model_rec

	mask_aligned = alignment
	mask_not_aligned = ~alignment

	kmf_aligned = KaplanMeierFitter()
	kmf_not_aligned = KaplanMeierFitter()

	plt.figure(figsize=(10, 6))

	kmf_aligned.fit(
		durations=df.loc[mask_aligned, time_col],
		event_observed=df.loc[mask_aligned, event_col],
		label="Following RSF recommendation",
	)
	ax = kmf_aligned.plot(ci_show=True)

	kmf_not_aligned.fit(
		durations=df.loc[mask_not_aligned, time_col],
		event_observed=df.loc[mask_not_aligned, event_col],
		label="Not following RSF recommendation",
	)
	kmf_not_aligned.plot(ax=ax, ci_show=True)

	results = logrank_test(
		df.loc[mask_aligned, time_col],
		df.loc[mask_not_aligned, time_col],
		event_observed_A=df.loc[mask_aligned, event_col],
		event_observed_B=df.loc[mask_not_aligned, event_col],
        weightings=weightings,
        p=p,
        q=q,
	)

	tau = 60
	rmst_aligned = float(restricted_mean_survival_time(kmf_aligned, t=tau))
	rmst_not_aligned = float(restricted_mean_survival_time(kmf_not_aligned, t=tau))
	rmst_diff = rmst_aligned - rmst_not_aligned

	plt.title("Kaplan-Meier Curves by Recommendation Following (Trial 50)")
	plt.xlabel("Time (months)")
	plt.ylabel("Survival Probability")
	add_at_risk_counts(kmf_aligned, kmf_not_aligned)
	plt.text(0.1, 0.1, f"Log-rank p-value: {results.p_value:.4f}", transform=plt.gca().transAxes)
	ax.set_xlim(left=0)
	ax.set_ylim(bottom=0)
	plt.tight_layout()

	if out_png:
		plt.savefig(out_png, dpi=600)
	plt.close()

	return {
		"tau": tau,
		"rmst_aligned": rmst_aligned,
		"rmst_not_aligned": rmst_not_aligned,
		"rmst_diff": rmst_diff,
		"n_aligned": int(mask_aligned.sum()),
		"n_not_aligned": int(mask_not_aligned.sum()),
		"logrank_pvalue": float(results.p_value),
	}


def build_rsf_params_from_trial(trial, n_jobs=1):
	hp = trial.params
	mf_mode = hp.get("max_features_mode", "sqrt")
	if mf_mode == "all":
		max_features = None
	elif mf_mode == "frac":
		max_features = float(hp.get("max_features_frac", 1.0))
	else:
		max_features = mf_mode

	return dict(
		n_estimators=int(hp.get("n_estimators", 1000)),
		max_depth=hp.get("max_depth", None),
		min_samples_split=int(hp.get("min_samples_split", 6)),
		min_samples_leaf=int(hp.get("min_samples_leaf", 3)),
		min_weight_fraction_leaf=0.0,
		max_features=max_features,
		max_leaf_nodes=None,
		bootstrap=True,
		oob_score=False,
		n_jobs=n_jobs,
		random_state=None,
		verbose=0,
		warm_start=False,
	)


def main():
	print(f"Loading Optuna study: {STUDY_NAME} ({OPTUNA_STORAGE})")
	study = optuna.load_study(study_name=STUDY_NAME, storage=OPTUNA_STORAGE)

	trial = None
	for t in study.trials:
		if t.number == TRIAL_NUMBER and t.state == optuna.trial.TrialState.COMPLETE:
			trial = t
			break
	if trial is None:
		raise RuntimeError(f"Trial #{TRIAL_NUMBER} not found or not complete.")

	print(f"Using trial #{trial.number} with params: {trial.params}")

	train_raw = pd.read_csv(TRAIN_CSV)
	valid_raw = pd.read_csv(VALID_CSV)
	test_raw = pd.read_csv(TEST_CSV)

	genes = load_genes_list(GENES_CSV)

	train_df = preprocess_split(train_raw, CLINICAL_VARS, genes)
	valid_df = preprocess_split(valid_raw, CLINICAL_VARS, genes)
	test_df = preprocess_split(test_raw, CLINICAL_VARS, genes)

	feat_candidates = [
		c for c in (CLINICAL_VARS + genes)
		if c in train_df.columns and c in valid_df.columns and c in test_df.columns
	]
	clin_feats = [c for c in CLINICAL_VARS if c in feat_candidates]
	gene_feats = [g for g in genes if g in feat_candidates]
	clin_feats_pretx = [c for c in clin_feats if c != "Adjuvant Chemo"]

	train_df = train_df.sort_values(by=["OS_MONTHS", "OS_STATUS"], ascending=[False, False]).reset_index(drop=True)
	valid_df = valid_df.sort_values(by=["OS_MONTHS", "OS_STATUS"], ascending=[False, False]).reset_index(drop=True)
	test_df = test_df.sort_values(by=["OS_MONTHS", "OS_STATUS"], ascending=[False, False]).reset_index(drop=True)

	gene_rank = rank_genes_univariate(train_df, gene_feats)

	trainval_df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
	trainval_df = trainval_df.sort_values(by=["OS_MONTHS", "OS_STATUS"], ascending=[False, False]).reset_index(drop=True)

	w_trv, ps_model_fin, pi_fin = compute_iptw(trainval_df, covariate_cols=clin_feats_pretx)
	_w_te, _, _ = compute_iptw(test_df, covariate_cols=clin_feats_pretx, ref_prev=pi_fin, model=ps_model_fin)

	k_main = int(trial.user_attrs["k_main"])
	k_int = int(trial.user_attrs["k_int"])
	dup_inter = int(trial.user_attrs.get("dup_inter", 1))
	genes_main = gene_rank[:k_main]
	genes_inter = genes_main[:k_int]

	X_trv, feat_names = build_features_with_interactions(trainval_df, genes_main, genes_inter, dup_inter=dup_inter, clin_cols=clin_feats)
	X_te, _ = build_features_with_interactions(test_df, genes_main, genes_inter, dup_inter=dup_inter, clin_cols=clin_feats)

	y_trv = Surv.from_arrays(
		event=trainval_df["OS_STATUS"].astype(bool).values,
		time=trainval_df["OS_MONTHS"].astype(float).values,
	)

	t_te = test_df["OS_MONTHS"].values.astype(float)
	e_te = test_df["OS_STATUS"].values.astype(int)

	rsf_params = build_rsf_params_from_trial(trial, n_jobs=1)

	all_runs = []

	print("\nRunning deterministic seed sweep (n_jobs=1, threadpool_limits=1)...")
	for seed in SEED_LIST:
		rsf_params["random_state"] = seed
		with threadpool_limits(limits=1):
			model = make_rsf(**rsf_params)
			model.fit(X_trv.astype(np.float64), y_trv, sample_weight=w_trv)
			pred_te = model.predict(X_te.astype(np.float64))

		ci_te = cindex(pred_te, t_te, e_te)
		km_res = compare_treatment_recommendation_km_rsf(
			model,
			test_df,
			genes_main=genes_main,
			genes_inter=genes_inter,
			dup_inter=dup_inter,
			clin_cols=clin_feats,
			out_png=None,
		)
		all_runs.append(
			{
				"seed": seed,
				"ci_test": ci_te,
				**km_res,
			}
		)

	runs_df = pd.DataFrame(all_runs)
	runs_df.to_csv(os.path.join(OUT_DIR, "trial_50_test_runs_all_seeds.csv"), index=False)

	median_pvalue = float(runs_df["logrank_pvalue"].median())
	runs_df["p_abs_dev"] = (runs_df["logrank_pvalue"] - median_pvalue).abs()
	selected_row = runs_df.sort_values(
		by=["p_abs_dev", "logrank_pvalue", "seed"],
		ascending=[True, True, True],
	).iloc[0]
	median_seed = int(selected_row["seed"])
	print(
		"Selected seed closest to median log-rank p-value: "
		f"{median_seed} (median={median_pvalue:.6f}, selected={selected_row['logrank_pvalue']:.6f})"
	)

	rsf_params["random_state"] = median_seed
	with threadpool_limits(limits=1):
		model_final = make_rsf(**rsf_params)
		model_final.fit(X_trv.astype(np.float64), y_trv, sample_weight=w_trv)

	km_png = os.path.join(OUT_DIR, "trial_50_test_km_alignment.png")
	km_png_weighted = os.path.join(OUT_DIR, "trial_50_test_km_alignment_weighted.png")
	with threadpool_limits(limits=1):
		km_final_unweighted = compare_treatment_recommendation_km_rsf(
			model_final,
			test_df,
			genes_main=genes_main,
			genes_inter=genes_inter,
			dup_inter=dup_inter,
			clin_cols=clin_feats,
			out_png=km_png,
		)
        
		p = 1
		q = 0
		weightings = "fleming-harrington"
  
		km_final_weighted = compare_treatment_recommendation_km_rsf(
			model_final,
			test_df,
			genes_main=genes_main,
			genes_inter=genes_inter,
			dup_inter=dup_inter,
			clin_cols=clin_feats,
			out_png=km_png_weighted,
            weightings=weightings,
            p=p,
            q=q,
		)

	print("\n=== Trial 50 Test KM + RMST (median-p seed) ===")
	print(f"Seed: {median_seed}")
	print(f"Log-rank p-value (unweighted): {km_final_unweighted['logrank_pvalue']:.6f}")
	print(f"Log-rank p-value (Fleming-Harrington p={p}, q={q}): {km_final_weighted['logrank_pvalue']:.6f}")
 
	print(f"RMST following (tau={km_final_unweighted['tau']}): {km_final_unweighted['rmst_aligned']:.6f}")
	print(f"RMST not following (tau={km_final_unweighted['tau']}): {km_final_unweighted['rmst_not_aligned']:.6f}")
	print(f"RMST difference (following - not following): {km_final_unweighted['rmst_diff']:.6f}")
	print(f"N following: {km_final_unweighted['n_aligned']}, N not following: {km_final_unweighted['n_not_aligned']}")
	print(f"Saved KM plot: {km_png}")
	print(f"Saved weighted KM plot: {km_png_weighted}")
	print(f"Saved all-seed results: {os.path.join(OUT_DIR, 'trial_50_test_runs_all_seeds.csv')}")


if __name__ == "__main__":
	main()