"""
Traffic Demand Prediction - Full Pipeline (v2)
Goal: Maximize R^2 score. Target: 98.5%+
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import time

SEED     = 42
N_FOLDS  = 5
DATA_DIR = "./"
np.random.seed(SEED)

print("=" * 65)
print("  TRAFFIC DEMAND PREDICTION - FULL PIPELINE v2")
print("=" * 65)

# ------------------------------------------------------------------
# 1. LOAD
# ------------------------------------------------------------------
print("\n[1] Loading data ...")
train = pd.read_csv(DATA_DIR + "train.csv")
test  = pd.read_csv(DATA_DIR + "test.csv")
print(f"    Train: {train.shape}  |  Test: {test.shape}")

# ------------------------------------------------------------------
# 2. FILL MISSING VALUES
# ------------------------------------------------------------------
print("[2] Filling missing values ...")
temp_median = train["Temperature"].median()
for df in [train, test]:
    df["Temperature"] = df["Temperature"].fillna(temp_median)
    df["Weather"]     = df["Weather"].fillna("Unknown")
    df["RoadType"]    = df["RoadType"].fillna("Unknown")

# ------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ------------------------------------------------------------------
print("[3] Feature engineering ...")

# -- 3A: Geohash decoding ------------------------------------------
BASE32  = "0123456789bcdefghjkmnpqrstuvwxyz"
B32_MAP = {c: i for i, c in enumerate(BASE32)}

def decode_geohash(gh):
    lat_min, lat_max = -90.0,  90.0
    lon_min, lon_max = -180.0, 180.0
    is_lon = True
    for ch in gh:
        bits = B32_MAP[ch]
        for shift in range(4, -1, -1):
            bit = (bits >> shift) & 1
            if is_lon:
                mid = (lon_min + lon_max) / 2.0
                lon_min, lon_max = (mid, lon_max) if bit else (lon_min, mid)
            else:
                mid = (lat_min + lat_max) / 2.0
                lat_min, lat_max = (mid, lat_max) if bit else (lat_min, mid)
            is_lon = not is_lon
    return ((lat_min+lat_max)/2, (lon_min+lon_max)/2,
            (lat_max-lat_min)/2, (lon_max-lon_min)/2)

def add_geo_features(df):
    dec = df["geohash"].apply(decode_geohash)
    df["latitude"]  = dec.apply(lambda x: x[0])
    df["longitude"] = dec.apply(lambda x: x[1])
    df["lat_error"] = dec.apply(lambda x: x[2])
    df["lon_error"] = dec.apply(lambda x: x[3])
    df["geo_3"] = df["geohash"].str[:3]
    df["geo_4"] = df["geohash"].str[:4]
    df["geo_5"] = df["geohash"].str[:5]
    return df

print("    Decoding geohashes ...")
train = add_geo_features(train)
test  = add_geo_features(test)

for col in ["geo_3", "geo_4", "geo_5"]:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]], ignore_index=True))
    train[col+"_enc"] = le.transform(train[col])
    test[col+"_enc"]  = le.transform(test[col])

# -- 3B: Temporal --------------------------------------------------
def parse_ts(df):
    parts = df["timestamp"].str.split(":", expand=True).astype(int)
    df["hour"]         = parts[0]
    df["minute"]       = parts[1]
    df["time_minutes"] = df["hour"] * 60 + df["minute"]
    df["hour_sin"]  = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]  = np.cos(2*np.pi*df["hour"]/24)
    df["time_sin"]  = np.sin(2*np.pi*df["time_minutes"]/1440)
    df["time_cos"]  = np.cos(2*np.pi*df["time_minutes"]/1440)
    df["is_rush_hour_morning"] = df["hour"].between(7,9).astype(int)
    df["is_rush_hour_evening"] = df["hour"].between(17,19).astype(int)
    df["is_rush_hour"] = (df["is_rush_hour_morning"] | df["is_rush_hour_evening"]).astype(int)
    df["is_night"]     = ((df["hour"]>=22)|(df["hour"]<=5)).astype(int)
    def tp(h):
        if h<=5:  return 0
        if h<=11: return 1
        if h<=16: return 2
        if h<=20: return 3
        return 4
    df["time_period"] = df["hour"].apply(tp)
    return df

train = parse_ts(train)
test  = parse_ts(test)

# -- 3C: Categorical encoding -------------------------------------
ROAD_MAP    = {"Unknown":0,"Residential":1,"Street":2,"Highway":3}
WEATHER_MAP = {"Unknown":0,"Sunny":1,"Rainy":2,"Foggy":3,"Snowy":4}
for df in [train, test]:
    df["LargeVehicles_bin"] = (df["LargeVehicles"]=="Allowed").astype(int)
    df["Landmarks_bin"]     = (df["Landmarks"]=="Yes").astype(int)
    df["RoadType_enc"] = df["RoadType"].map(ROAD_MAP).fillna(0).astype(int)
    df["Weather_enc"]  = df["Weather"].map(WEATHER_MAP).fillna(0).astype(int)

# -- 3D: Frequency features ---------------------------------------
for col, fcol in [("geohash","geohash_freq"),("geo_3","geo_3_freq"),
                   ("geo_4","geo_4_freq"),("geo_5","geo_5_freq")]:
    freq = pd.concat([train[col], test[col]], ignore_index=True).value_counts().to_dict()
    train[fcol] = train[col].map(freq).fillna(0)
    test[fcol]  = test[col].map(freq).fillna(0)

# -- 3E: Geohash target statistics (TRAIN ONLY - no leakage) ------
print("    Computing geohash target statistics ...")
global_mean = train["demand"].mean()
global_std  = train["demand"].std()

def safe_merge(df, stats, on):
    return df.merge(stats, on=on, how="left")

def fill_stats(df, mean_cols, zero_cols):
    for c in mean_cols:
        df[c] = df[c].fillna(global_mean)
    for c in zero_cols:
        df[c] = df[c].fillna(0)
    return df

# geohash-level
g_stat = train.groupby("geohash")["demand"].agg(
    geo_demand_mean="mean", geo_demand_std="std",
    geo_demand_median="median", geo_demand_max="max",
    geo_demand_min="min", geo_demand_count="count").reset_index()
g_stat["geo_demand_std"] = g_stat["geo_demand_std"].fillna(0)

# geohash + hour
gh_stat = train.groupby(["geohash","hour"])["demand"].agg(
    geo_hour_demand_mean="mean", geo_hour_demand_std="std").reset_index()
gh_stat["geo_hour_demand_std"] = gh_stat["geo_hour_demand_std"].fillna(0)

# geohash + time_minutes  <-- KEY NEW FEATURE
gt_stat = train.groupby(["geohash","time_minutes"])["demand"].agg(
    geo_time_demand_mean="mean", geo_time_demand_std="std").reset_index()
gt_stat["geo_time_demand_std"] = gt_stat["geo_time_demand_std"].fillna(0)

# geohash + day
gd_stat = train.groupby(["geohash","day"])["demand"].agg(
    geo_day_demand_mean="mean").reset_index()

# geohash + RoadType
gr_stat = train.groupby(["geohash","RoadType"])["demand"].agg(
    geo_road_demand_mean="mean").reset_index()

# geo_3 / geo_4 / geo_5 demand means
g3_stat = train.groupby("geo_3")["demand"].agg(geo_3_demand_mean="mean", geo_3_demand_std="std").reset_index()
g4_stat = train.groupby("geo_4")["demand"].agg(geo_4_demand_mean="mean", geo_4_demand_std="std").reset_index()
g5_stat = train.groupby("geo_5")["demand"].agg(geo_5_demand_mean="mean", geo_5_demand_std="std").reset_index()
for s in [g3_stat, g4_stat, g5_stat]:
    for c in s.columns:
        if "std" in c:
            s[c] = s[c].fillna(0)

def merge_all_stats(df):
    df = safe_merge(df, g_stat,  "geohash")
    df = safe_merge(df, gh_stat, ["geohash","hour"])
    df = safe_merge(df, gt_stat, ["geohash","time_minutes"])
    df = safe_merge(df, gd_stat, ["geohash","day"])
    df = safe_merge(df, gr_stat, ["geohash","RoadType"])
    df = safe_merge(df, g3_stat, "geo_3")
    df = safe_merge(df, g4_stat, "geo_4")
    df = safe_merge(df, g5_stat, "geo_5")
    mean_cols = ["geo_demand_mean","geo_demand_median","geo_demand_max","geo_demand_min",
                 "geo_hour_demand_mean","geo_time_demand_mean","geo_day_demand_mean",
                 "geo_road_demand_mean","geo_3_demand_mean","geo_4_demand_mean","geo_5_demand_mean"]
    zero_cols = ["geo_demand_std","geo_demand_count","geo_hour_demand_std",
                 "geo_time_demand_std","geo_3_demand_std","geo_4_demand_std","geo_5_demand_std"]
    df = fill_stats(df, mean_cols, zero_cols)
    return df

train = merge_all_stats(train)
test  = merge_all_stats(test)

# -- 3F: Spatial clustering ----------------------------------------
print("    KMeans spatial clustering ...")
coords = pd.concat([train[["latitude","longitude"]],
                    test[["latitude","longitude"]]], ignore_index=True).values
km = KMeans(n_clusters=30, random_state=SEED, n_init=10)
km.fit(coords)
train["geo_cluster"] = km.predict(train[["latitude","longitude"]].values)
test["geo_cluster"]  = km.predict(test[["latitude","longitude"]].values)

# cluster + hour demand stat
gc_stat = train.groupby(["geo_cluster","hour"])["demand"].agg(
    geo_cluster_hour_mean="mean").reset_index()
train = train.merge(gc_stat, on=["geo_cluster","hour"], how="left")
test  = test.merge(gc_stat,  on=["geo_cluster","hour"], how="left")
train["geo_cluster_hour_mean"] = train["geo_cluster_hour_mean"].fillna(global_mean)
test["geo_cluster_hour_mean"]  = test["geo_cluster_hour_mean"].fillna(global_mean)

# -- 3G: Geohash label encoding ------------------------------------
le_gh = LabelEncoder()
le_gh.fit(pd.concat([train["geohash"], test["geohash"]], ignore_index=True))
train["geohash_enc"] = le_gh.transform(train["geohash"])
test["geohash_enc"]  = le_gh.transform(test["geohash"])

# -- 3H: Interaction features --------------------------------------
for df in [train, test]:
    df["temp_hour"]       = df["Temperature"] * df["hour"]
    df["temp_squared"]    = df["Temperature"] ** 2
    df["lane_road"]       = df["NumberofLanes"] * df["RoadType_enc"]
    df["lane_vehicles"]   = df["NumberofLanes"] * df["LargeVehicles_bin"]
    df["geo_freq_temp"]   = df["geohash_freq"] * df["Temperature"]
    df["lat_lon_product"] = df["latitude"] * df["longitude"]
    df["weather_temp"]    = df["Weather_enc"] * df["Temperature"]
    df["rush_road"]       = df["is_rush_hour"] * df["RoadType_enc"]
    df["rush_lane"]       = df["is_rush_hour"] * df["NumberofLanes"]
    df["rush_weather"]    = df["is_rush_hour"] * df["Weather_enc"]
    df["night_weather"]   = df["is_night"] * df["Weather_enc"]
    # new ratio features
    df["time_geo_ratio"]  = df["geo_time_demand_mean"] / (df["geo_demand_mean"] + 1e-6)
    df["hour_geo_ratio"]  = df["geo_hour_demand_mean"] / (df["geo_demand_mean"] + 1e-6)

# ------------------------------------------------------------------
# 4. FEATURE LIST
# ------------------------------------------------------------------
FEATURE_COLS = [
    "geohash_enc", "geo_3_enc", "geo_4_enc", "geo_5_enc",
    "latitude", "longitude", "lat_error", "lon_error",
    "day", "NumberofLanes",
    "hour", "minute", "time_minutes",
    "hour_sin", "hour_cos", "time_sin", "time_cos",
    "is_rush_hour_morning", "is_rush_hour_evening", "is_rush_hour", "is_night",
    "time_period",
    "LargeVehicles_bin", "Landmarks_bin", "RoadType_enc", "Weather_enc", "Temperature",
    "geohash_freq", "geo_3_freq", "geo_4_freq", "geo_5_freq",
    # target stats
    "geo_demand_mean", "geo_demand_std", "geo_demand_median",
    "geo_demand_max", "geo_demand_min", "geo_demand_count",
    "geo_hour_demand_mean", "geo_hour_demand_std",
    "geo_time_demand_mean", "geo_time_demand_std",
    "geo_day_demand_mean",
    "geo_road_demand_mean",
    "geo_3_demand_mean", "geo_3_demand_std",
    "geo_4_demand_mean", "geo_4_demand_std",
    "geo_5_demand_mean", "geo_5_demand_std",
    "geo_cluster_hour_mean",
    # interactions
    "temp_hour", "temp_squared", "lane_road", "lane_vehicles",
    "geo_freq_temp", "lat_lon_product", "weather_temp",
    "rush_road", "rush_lane", "rush_weather", "night_weather",
    "time_geo_ratio", "hour_geo_ratio",
    "geo_cluster",
]

X      = train[FEATURE_COLS].values.astype(np.float32)
y      = train["demand"].values.astype(np.float32)
X_test = test[FEATURE_COLS].values.astype(np.float32)

print(f"\n    Total features: {len(FEATURE_COLS)}")

# ------------------------------------------------------------------
# 5. 5-FOLD CV — 4 MODELS
# ------------------------------------------------------------------
print("\n[4] 5-Fold CV (LGB + XGB + CAT + HGB) ...\n")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb  = np.zeros(len(y))
oof_xgb  = np.zeros(len(y))
oof_cat  = np.zeros(len(y))
oof_hgb  = np.zeros(len(y))

test_lgb  = np.zeros(len(X_test))
test_xgb  = np.zeros(len(X_test))
test_cat  = np.zeros(len(X_test))
test_hgb  = np.zeros(len(X_test))

lgb_scores, xgb_scores, cat_scores, hgb_scores = [], [], [], []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    print(f"  -- Fold {fold}/{N_FOLDS} --")
    t0 = time.time()

    # LightGBM
    lgb_m = lgb.LGBMRegressor(
        n_estimators=5000, learning_rate=0.01, max_depth=10,
        num_leaves=255, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.05, reg_lambda=0.1,
        random_state=SEED, n_jobs=-1, verbosity=-1)
    lgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(period=-1)])
    oof_lgb[val_idx]  = lgb_m.predict(X_val)
    test_lgb         += lgb_m.predict(X_test) / N_FOLDS
    s = r2_score(y_val, oof_lgb[val_idx]); lgb_scores.append(s)
    print(f"    LGB  R^2={s:.5f}  (iter={lgb_m.best_iteration_})")

    # XGBoost
    xgb_m = xgb.XGBRegressor(
        n_estimators=5000, learning_rate=0.01, max_depth=8,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.05, reg_lambda=0.1,
        early_stopping_rounds=100,
        random_state=SEED, n_jobs=-1, verbosity=0, tree_method="hist")
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx]  = xgb_m.predict(X_val)
    test_xgb         += xgb_m.predict(X_test) / N_FOLDS
    s = r2_score(y_val, oof_xgb[val_idx]); xgb_scores.append(s)
    print(f"    XGB  R^2={s:.5f}  (iter={xgb_m.best_iteration})")

    # CatBoost
    cat_m = CatBoostRegressor(
        iterations=5000, learning_rate=0.01, depth=8,
        l2_leaf_reg=3, subsample=0.8,
        early_stopping_rounds=100, random_seed=SEED, verbose=False)
    cat_m.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    oof_cat[val_idx]  = cat_m.predict(X_val)
    test_cat         += cat_m.predict(X_test) / N_FOLDS
    s = r2_score(y_val, oof_cat[val_idx]); cat_scores.append(s)
    print(f"    CAT  R^2={s:.5f}  (iter={cat_m.best_iteration_})")

    # HistGradientBoosting (sklearn - complements tree ensembles)
    hgb_m = HistGradientBoostingRegressor(
        max_iter=2000, learning_rate=0.02, max_depth=10,
        max_leaf_nodes=127, min_samples_leaf=20,
        l2_regularization=0.1, random_state=SEED,
        early_stopping=True, n_iter_no_change=50,
        validation_fraction=0.1)
    hgb_m.fit(X_tr, y_tr)
    oof_hgb[val_idx]  = hgb_m.predict(X_val)
    test_hgb         += hgb_m.predict(X_test) / N_FOLDS
    s = r2_score(y_val, oof_hgb[val_idx]); hgb_scores.append(s)
    elapsed = time.time() - t0
    print(f"    HGB  R^2={s:.5f}  [{elapsed:.0f}s]\n")

print("-" * 50)
print(f"  LightGBM  CV R^2: {np.mean(lgb_scores):.5f}")
print(f"  XGBoost   CV R^2: {np.mean(xgb_scores):.5f}")
print(f"  CatBoost  CV R^2: {np.mean(cat_scores):.5f}")
print(f"  HGB       CV R^2: {np.mean(hgb_scores):.5f}")
print("-" * 50)

# ------------------------------------------------------------------
# 6. ENSEMBLE - NELDER-MEAD + RIDGE STACKER
# ------------------------------------------------------------------
print("\n[5] Optimizing ensemble weights ...")

oof_mat  = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_hgb])
test_mat = np.column_stack([test_lgb, test_xgb, test_cat, test_hgb])

def neg_r2(weights):
    w = np.clip(weights, 0, None)
    s = w.sum()
    if s < 1e-9: return 1.0
    return -r2_score(y, oof_mat @ (w/s))

starts = [
    [0.4,0.2,0.3,0.1],[0.3,0.3,0.3,0.1],[0.25,0.25,0.4,0.1],
    [0.5,0.1,0.3,0.1],[0.2,0.2,0.5,0.1],[0.3,0.2,0.4,0.1],
    [0.4,0.1,0.4,0.1],[0.33,0.33,0.17,0.17],
]
best = None
for sp in starts:
    res = minimize(neg_r2, sp, method="Nelder-Mead",
                   options={"maxiter":10000,"xatol":1e-8,"fatol":1e-8})
    if best is None or res.fun < best.fun:
        best = res

raw = np.clip(best.x, 0, None)
opt_w = raw / raw.sum()

# Also try Ridge meta-stacker
ridge = Ridge(alpha=1.0, fit_intercept=True)
ridge.fit(oof_mat, y)
oof_ridge = ridge.predict(oof_mat)
r2_ridge  = r2_score(y, oof_ridge)

# Pick the better approach
oof_nm  = oof_mat @ opt_w
r2_nm   = r2_score(y, oof_nm)

if r2_ridge > r2_nm:
    print("  Using Ridge meta-stacker (better than Nelder-Mead)")
    final_test_preds = ridge.predict(test_mat)
    r2_ensemble = r2_ridge
    use_ridge = True
else:
    print("  Using Nelder-Mead weights (better than Ridge)")
    final_test_preds = test_mat @ opt_w
    r2_ensemble = r2_nm
    use_ridge = False

r2_lgb_oof = r2_score(y, oof_lgb)
r2_xgb_oof = r2_score(y, oof_xgb)
r2_cat_oof = r2_score(y, oof_cat)
r2_hgb_oof = r2_score(y, oof_hgb)

print(f"\n  Weights (NM)     -> LGB={opt_w[0]:.4f} XGB={opt_w[1]:.4f} CAT={opt_w[2]:.4f} HGB={opt_w[3]:.4f}")
print(f"  LightGBM OOF R^2 : {r2_lgb_oof:.5f}")
print(f"  XGBoost  OOF R^2 : {r2_xgb_oof:.5f}")
print(f"  CatBoost OOF R^2 : {r2_cat_oof:.5f}")
print(f"  HGB      OOF R^2 : {r2_hgb_oof:.5f}")
print(f"  Ridge    OOF R^2 : {r2_ridge:.5f}")
print(f"  ENSEMBLE OOF R^2 : {r2_ensemble:.5f}  <- score x100 = {r2_ensemble*100:.2f}")

# ------------------------------------------------------------------
# 7. SUBMISSION
# ------------------------------------------------------------------
print("\n[6] Generating submission.csv ...")
final_preds = np.clip(final_test_preds, 0.0, 1.0)
sub = pd.DataFrame({"Index": test["Index"].values, "demand": final_preds})
sub.to_csv(DATA_DIR + "submission.csv", index=False)
print(f"  Shape : {sub.shape}")
print(f"  Min   : {final_preds.min():.6f}")
print(f"  Max   : {final_preds.max():.6f}")
print(f"  Mean  : {final_preds.mean():.6f}")
print(f"  Median: {np.median(final_preds):.6f}")
print(f"\n  First 5 rows:")
print(sub.head(5).to_string(index=False))

# ------------------------------------------------------------------
# 8. FINAL SUMMARY
# ------------------------------------------------------------------
print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
print(f"  Features          : {len(FEATURE_COLS)}")
print(f"  LightGBM  OOF R^2 : {r2_lgb_oof:.5f}  (score={r2_lgb_oof*100:.2f})")
print(f"  XGBoost   OOF R^2 : {r2_xgb_oof:.5f}  (score={r2_xgb_oof*100:.2f})")
print(f"  CatBoost  OOF R^2 : {r2_cat_oof:.5f}  (score={r2_cat_oof*100:.2f})")
print(f"  HGB       OOF R^2 : {r2_hgb_oof:.5f}  (score={r2_hgb_oof*100:.2f})")
print(f"  ENSEMBLE  OOF R^2 : {r2_ensemble:.5f}  (score={r2_ensemble*100:.2f})")
print("=" * 65)
print("  submission.csv saved.")
print("=" * 65)
