import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Tuple


# =========================================================
# 1. 설정
# =========================================================

@dataclass
class DataConfig:
    num_devices: int = 300
    num_days: int = 365
    forecast_horizon: int = 7
    random_state: int = 42

    danger_temp_threshold: float = 95.0
    danger_health_threshold: float = 30.0

    specs: Tuple[str, ...] = ("AC_7kW", "DC_50kW", "DC_100kW")
    locations: Tuple[str, ...] = ("Indoor", "Outdoor")
    scenarios: Tuple[str, ...] = (
        "normal",
        "gradual_overheat",
        "sudden_fault",
        "recover_and_relapse",
        "health_drop",
        "borderline",
    )

    save_dir: str = "./EV_project/data"


# =========================================================
# 2. 메타 생성
# =========================================================

def generate_device_meta(cfg: DataConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_state)

    rows = []
    for i in range(cfg.num_devices):
        spec = rng.choice(cfg.specs, p=[0.38, 0.37, 0.25])
        loc = rng.choice(cfg.locations, p=[0.50, 0.50])

        scenario = rng.choice(
            cfg.scenarios,
            p=[0.40, 0.14, 0.10, 0.16, 0.08, 0.12]
        )

        rows.append({
            "ID": f"EV_{i:04d}",
            "Spec": spec,
            "Loc": loc,
            "Scenario": scenario,
            "temp_bias": round(float(rng.normal(0, 3.0)), 4),
            "health_bias": round(float(rng.normal(0, 4.0)), 4),
            "usage_bias": round(float(rng.normal(0, 1.1)), 4),
            "voltage_bias": round(float(rng.normal(0, 3.0)), 4),
            "current_bias": round(float(rng.normal(0, 2.0)), 4),
            "sensor_quality": round(float(rng.uniform(0.85, 1.15)), 4),
        })

    return pd.DataFrame(rows)


# =========================================================
# 3. 보조 함수
# =========================================================

def _spec_to_capacity(spec: str) -> int:
    mapping = {
        "AC_7kW": 7,
        "DC_50kW": 50,
        "DC_100kW": 100,
    }
    if spec not in mapping:
        raise ValueError(f"Unknown spec: {spec}")
    return mapping[spec]


def _base_usage(spec: str) -> float:
    return {
        "AC_7kW": 4.5,
        "DC_50kW": 8.5,
        "DC_100kW": 10.5,
    }[spec]


def _base_temp(spec: str, loc: str) -> float:
    spec_base = {
        "AC_7kW": 37.0,
        "DC_50kW": 57.0,
        "DC_100kW": 67.0,
    }[spec]
    loc_add = 4.0 if loc == "Outdoor" else 0.0
    return spec_base + loc_add


def _base_voltage(spec: str) -> float:
    return {
        "AC_7kW": 220.0,
        "DC_50kW": 310.0,
        "DC_100kW": 380.0,
    }[spec]


def _yearly_signal(day: int) -> float:
    return 5.0 * np.sin(2 * np.pi * day / 365.0)


def _monthly_signal(day: int) -> float:
    return 2.2 * np.sin(2 * np.pi * day / 30.0)


def _weekly_usage_signal(day: int) -> float:
    return 0.8 * np.sin(2 * np.pi * day / 7.0)


def _event_window(day: int, start: int, length: int) -> bool:
    return start <= day < start + length


def _maybe_missing(value: float, rng: np.random.Generator, p: float) -> float:
    return np.nan if rng.random() < p else value


# =========================================================
# 4. 장비별 시계열 생성
# =========================================================

def simulate_device_timeseries(
    device_row: pd.Series,
    cfg: DataConfig,
    rng: np.random.Generator
) -> pd.DataFrame:
    device_id = device_row["ID"]
    spec = device_row["Spec"]
    loc = device_row["Loc"]
    scenario = device_row["Scenario"]

    capacity = _spec_to_capacity(spec)
    usage_base = _base_usage(spec) + float(device_row["usage_bias"])
    temp_base = _base_temp(spec, loc) + float(device_row["temp_bias"])
    voltage_base = _base_voltage(spec) + float(device_row["voltage_bias"])
    current_bias = float(device_row["current_bias"])
    sensor_quality = float(device_row["sensor_quality"])

    health = float(np.clip(100 + float(device_row["health_bias"]), 82, 100))
    total_kwh = 0.0

    degrade_base = float(rng.uniform(0.015, 0.08))
    temp_drift = float(rng.normal(0.0, 0.006))
    voltage_drift = float(rng.normal(0.0, 0.01))

    temp_prev = temp_base
    health_prev = health

    gradual_start = int(rng.integers(90, 270))
    sudden_start = int(rng.integers(120, 330))
    relapse_start1 = int(rng.integers(70, 180))
    relapse_start2 = int(rng.integers(220, 330))
    health_drop_start = int(rng.integers(110, 310))
    borderline_start = int(rng.integers(140, 310))

    rows = []

    for day in range(1, cfg.num_days + 1):
        seasonal_temp = _yearly_signal(day)
        monthly_temp = _monthly_signal(day)
        weekly_usage = _weekly_usage_signal(day)

        usage_hrs = usage_base + weekly_usage + rng.normal(0, 1.2)

        if rng.random() < 0.05:
            usage_hrs += float(rng.uniform(1.5, 4.5))
        if rng.random() < 0.04:
            usage_hrs -= float(rng.uniform(1.0, 3.0))

        usage_hrs = float(np.clip(usage_hrs, 0.5, 18.0))

        load_factor = float(rng.uniform(0.40, 0.92))
        daily_kwh = usage_hrs * capacity * load_factor
        total_kwh += daily_kwh

        temp_shock = 0.0
        health_shock = 0.0
        recovery = 0.0

        # =====================================================
        # 시나리오별 패턴
        # =====================================================

        if scenario == "normal":
            if rng.random() < 0.05:
                temp_shock += float(rng.uniform(4, 10))

            if rng.random() < 0.04:
                recovery -= float(rng.uniform(2, 5))

            if rng.random() < 0.02:
                temp_shock += float(rng.uniform(10, 16))

            if rng.random() < 0.02:
                health_shock += float(rng.uniform(-0.3, 0.5))

            if rng.random() < 0.03:
                recovery -= float(rng.uniform(2, 6))

        elif scenario == "gradual_overheat":
            if day >= gradual_start:
                progress = (day - gradual_start) / max(cfg.num_days - gradual_start, 1)
                temp_shock += min(16.0, 1.5 + 18.0 * progress)
                health_shock += min(0.9, 0.05 + 0.9 * progress)

                if rng.random() < 0.05:
                    recovery -= float(rng.uniform(1.5, 4.0))

        elif scenario == "sudden_fault":
            if _event_window(day, sudden_start, 18):
                temp_shock += float(rng.uniform(12, 26))
                health_shock += float(rng.uniform(1.0, 2.6))

                if rng.random() < 0.18:
                    temp_shock += float(rng.uniform(3, 8))

            elif day > sudden_start + 18:
                temp_shock += float(rng.uniform(3, 9))
                health_shock += float(rng.uniform(0.15, 0.8))

        elif scenario == "recover_and_relapse":
            if _event_window(day, relapse_start1, 18):
                temp_shock += float(rng.uniform(6, 12))
                health_shock += float(rng.uniform(0.4, 1.0))

            elif _event_window(day, relapse_start1 + 18, 40):
                recovery -= float(rng.uniform(2, 6))
                health += float(rng.uniform(0.05, 0.3))

                if rng.random() < 0.08:
                    temp_shock -= float(rng.uniform(1, 3))

            elif _event_window(day, relapse_start1 + 58, 25):
                temp_shock += float(rng.uniform(-1, 3))
                health_shock += float(rng.uniform(0.0, 0.3))

            elif _event_window(day, relapse_start2, 35):
                temp_shock += float(rng.uniform(9, 18))
                health_shock += float(rng.uniform(0.8, 1.8))

                if rng.random() < 0.12:
                    temp_shock += float(rng.uniform(3, 8))

            elif day > relapse_start2 + 35:
                temp_shock += float(rng.uniform(2, 7))
                health_shock += float(rng.uniform(0.2, 0.8))

        elif scenario == "health_drop":
            if day >= health_drop_start:
                progress = (day - health_drop_start) / max(cfg.num_days - health_drop_start, 1)
                health_shock += min(1.5, 0.18 + 1.6 * progress)
                temp_shock += float(rng.uniform(-2, 2.5))

        elif scenario == "borderline":
            if day >= borderline_start:
                progress = (day - borderline_start) / max(cfg.num_days - borderline_start, 1)

                temp_shock += min(5.6, 0.4 + 5.0 * progress)
                health_shock += min(0.45, 0.02 + 0.48 * progress)

                if rng.random() < 0.10:
                    recovery -= float(rng.uniform(0.8, 3.0))

                if rng.random() < 0.07:
                    temp_shock += float(rng.uniform(1.5, 5.0))

                if rng.random() < 0.05:
                    temp_shock -= float(rng.uniform(0.8, 2.5))

                if rng.random() < 0.07:
                    health_shock += float(rng.uniform(0.08, 0.28))

        # =====================================================
        # 온도 생성
        # =====================================================

        peak_temp = (
            temp_base
            + 0.56 * usage_hrs
            + 0.00015 * total_kwh
            + seasonal_temp
            + monthly_temp
            + temp_drift * day
            + temp_shock
            + recovery
            + rng.normal(0, 3.0 * sensor_quality)
        )

        if rng.random() < 0.03:
            peak_temp += float(rng.uniform(6, 15))
        if rng.random() < 0.02:
            peak_temp -= float(rng.uniform(3, 7))

        if scenario in ("health_drop", "recover_and_relapse") and rng.random() < 0.07:
            peak_temp -= float(rng.uniform(2, 7))
        elif scenario == "borderline" and rng.random() < 0.05:
            peak_temp -= float(rng.uniform(1.5, 5.0))

        peak_temp = float(max(15.0, peak_temp))

        # =====================================================
        # 건강도 생성
        # =====================================================

        health_drop = (
            degrade_base
            + 0.0015 * usage_hrs
            + 0.000012 * daily_kwh
            + health_shock
            + rng.normal(0, 0.09 * sensor_quality)
        )

        if peak_temp > 85:
            health_drop += float(rng.uniform(0.10, 0.40))
        if peak_temp > 95:
            health_drop += float(rng.uniform(0.25, 0.90))

        if rng.random() < 0.025:
            health_drop += float(rng.uniform(-0.4, 0.8))

        health -= health_drop
        health += float(rng.normal(0, 0.55 * sensor_quality))
        health = float(np.clip(health, 5, 100))

        # =====================================================
        # 전압 / 전류
        # =====================================================

        voltage = voltage_base + voltage_drift * day + rng.normal(0, 4.0 * sensor_quality)

        if scenario == "sudden_fault" and _event_window(day, sudden_start, 18):
            voltage += float(rng.uniform(-14, 12))

        if rng.random() < 0.02:
            voltage += float(rng.uniform(-15, 12))

        current = daily_kwh / max(usage_hrs, 0.5)
        current += current_bias + rng.normal(0, max(current * 0.06, 0.8) * sensor_quality)

        if scenario == "sudden_fault" and _event_window(day, sudden_start, 18) and rng.random() < 0.18:
            current += float(rng.uniform(5, 14))
        elif rng.random() < 0.025:
            current += float(rng.uniform(4, 12))

        # =====================================================
        # 변화량
        # =====================================================

        temp_change = peak_temp - temp_prev
        health_change = health - health_prev

        temp_prev = peak_temp
        health_prev = health

        # =====================================================
        # 결측
        # =====================================================

        peak_temp_obs = _maybe_missing(round(float(peak_temp), 2), rng, 0.012)
        health_obs = _maybe_missing(round(float(health), 2), rng, 0.012)
        voltage_obs = _maybe_missing(round(float(voltage), 2), rng, 0.010)
        current_obs = _maybe_missing(round(float(current), 2), rng, 0.010)

        rows.append({
            "ID": device_id,
            "Spec": spec,
            "Loc": loc,
            "Scenario": scenario,
            "Day": day,
            "Usage_Hrs": round(float(usage_hrs), 2),
            "Daily_KWh": round(float(daily_kwh), 2),
            "Total_KWh": round(float(total_kwh), 2),
            "Voltage": voltage_obs,
            "Current": current_obs,
            "Peak_T": peak_temp_obs,
            "Health": health_obs,
            "Temp_Change": round(float(temp_change), 2),
            "Health_Change": round(float(health_change), 2),
            "Spec_Val": capacity,
            "Loc_Val": 1 if loc == "Outdoor" else 0,
        })

    return pd.DataFrame(rows)


def generate_raw_timeseries(cfg: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.random_state)
    meta_df = generate_device_meta(cfg)

    frames = []
    for _, row in meta_df.iterrows():
        frames.append(simulate_device_timeseries(row, cfg, rng))

    raw_df = pd.concat(frames, ignore_index=True)
    raw_df.columns = raw_df.columns.str.strip()
    raw_df = raw_df.sort_values(["ID", "Day"]).reset_index(drop=True)

    return meta_df, raw_df


# =========================================================
# 5. 현재 위험 / 미래 위험 / 상태 라벨
# =========================================================

def add_current_danger_flag(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.strip()

    peak_t_for_label = out["Peak_T"].copy()
    health_for_label = out["Health"].copy()

    peak_t_for_label = peak_t_for_label.groupby(out["ID"]).transform(lambda s: s.ffill().bfill())
    health_for_label = health_for_label.groupby(out["ID"]).transform(lambda s: s.ffill().bfill())

    out["Danger_Now"] = (
        (peak_t_for_label >= cfg.danger_temp_threshold) |
        (health_for_label <= cfg.danger_health_threshold)
    ).astype(int)

    return out


def add_future_risk_label(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.strip()

    if "ID" not in out.columns or "Day" not in out.columns or "Danger_Now" not in out.columns:
        raise KeyError("ID / Day / Danger_Now 컬럼을 확인하세요.")

    out = out.sort_values(["ID", "Day"]).reset_index(drop=True)
    out["Label_RiskInHorizon"] = 0

    parts = []

    for _, g in out.groupby("ID", sort=False):
        g = g.sort_values("Day").reset_index(drop=True).copy()

        peak_t = g["Peak_T"].ffill().bfill()
        health = g["Health"].ffill().bfill()
        temp_change = g["Temp_Change"].ffill().bfill().fillna(0)
        danger_arr = g["Danger_Now"].to_numpy()

        labels = np.zeros(len(g), dtype=int)

        for i in range(len(g)):
            start = i + 1
            end = min(i + 1 + cfg.forecast_horizon, len(g))

            future_danger = danger_arr[start:end].sum() > 0

            if future_danger:
                labels[i] = 1
                continue

            future_peak_t = peak_t.iloc[start:end]
            future_health = health.iloc[start:end]
            future_temp_change = temp_change.iloc[start:end]

            # soft risk 조건은 너무 넓지 않게:
            # "온도 상승 + 건강도 저하"가 같이 있을 때만 위험 후보로 본다
            soft_risk = (
                ((future_peak_t >= 80) & (future_peak_t < cfg.danger_temp_threshold)).any()
                and (future_health <= 55).any()
            )

            # 더 민감하게 보고 싶으면 아래 줄 켜면 됨
            # soft_risk = soft_risk and (future_temp_change >= 5).any()

            labels[i] = 1 if soft_risk else 0

        g["Label_RiskInHorizon"] = labels
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["ID", "Day"]).reset_index(drop=True)
    return out


def add_status_3class_label(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.strip()

    peak_t = out.groupby("ID")["Peak_T"].transform(lambda s: s.ffill().bfill())
    health = out.groupby("ID")["Health"].transform(lambda s: s.ffill().bfill())
    current = out.groupby("ID")["Current"].transform(lambda s: s.ffill().bfill())
    temp_change = out.groupby("ID")["Temp_Change"].transform(lambda s: s.ffill().bfill()).fillna(0)
    health_change = out.groupby("ID")["Health_Change"].transform(lambda s: s.ffill().bfill()).fillna(0)

    out["Label_Status3"] = 0

    danger_cond = (
        (out["Danger_Now"] == 1) |
        (peak_t >= 85) |
        (health <= 30)
    )
    out.loc[danger_cond, "Label_Status3"] = 2

    warning_cond = (
        (out["Label_Status3"] == 0) & (
            ((peak_t >= 65) & (peak_t < 85)) |
            ((health > 30) & (health <= 50)) |
            (current >= 140) |
            (temp_change >= 5) |
            (health_change <= -1.0) |
            ((out["Label_RiskInHorizon"] == 1) & (out["Danger_Now"] == 0))
        )
    )
    out.loc[warning_cond, "Label_Status3"] = 1

    return out


def add_time_to_risk_label(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.strip()

    if "ID" not in out.columns or "Day" not in out.columns or "Danger_Now" not in out.columns:
        raise KeyError("ID / Day / Danger_Now 컬럼 확인")

    out = out.sort_values(["ID", "Day"]).reset_index(drop=True)
    out["Time_To_Risk"] = np.nan

    parts = []

    for _, g in out.groupby("ID", sort=False):
        g = g.sort_values("Day").reset_index(drop=True).copy()

        danger_arr = g["Danger_Now"].to_numpy()
        ttr = np.full(len(g), np.nan)

    for i in range(len(g)):
        future_idx = np.where(danger_arr[i + 1:] == 1)[0]

        if len(future_idx) == 0:
            ttr[i] = cfg.forecast_horizon + 1
        else:
            days_to_risk = future_idx[0] + 1
            if days_to_risk <= cfg.forecast_horizon:
                ttr[i] = days_to_risk
            else:
                ttr[i] = cfg.forecast_horizon + 1

        g["Time_To_Risk"] = ttr
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["ID", "Day"]).reset_index(drop=True)

    return out


# =========================================================
# 6. 피처 엔지니어링
# =========================================================

def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "Day",
        "Usage_Hrs",
        "Daily_KWh",
        "Total_KWh",
        "Voltage",
        "Current",
        "Peak_T",
        "Health",
        "Temp_Change",
        "Health_Change",
        "Spec_Val",
        "Loc_Val",
    ]   

    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    out.columns = out.columns.str.strip()
    out = out.sort_values(["ID", "Day"]).reset_index(drop=True)

    group = out.groupby("ID", sort=False)

    base_cols = [
        "Peak_T", "Health", "Usage_Hrs", "Daily_KWh",
        "Voltage", "Current", "Temp_Change", "Health_Change"
    ]

    for col in ["Peak_T", "Health", "Voltage", "Current"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out.groupby("ID")[col].transform(
            lambda s: s.interpolate(limit_direction="both")
        )
        out[col] = out.groupby("ID")[col].transform(
            lambda s: s.ffill().bfill()
        )

    group = out.groupby("ID", group_keys=False)

    for col in base_cols:
        out[f"{col}_lag1"] = group[col].shift(1)
        out[f"{col}_lag3"] = group[col].shift(3)
        out[f"{col}_lag7"] = group[col].shift(7)

    out["Peak_T_delta1"] = out["Peak_T"] - out["Peak_T_lag1"]
    out["Peak_T_delta3"] = out["Peak_T"] - out["Peak_T_lag3"]
    out["Peak_T_delta7"] = out["Peak_T"] - out["Peak_T_lag7"]

    out["Health_delta1"] = out["Health"] - out["Health_lag1"]
    out["Health_delta3"] = out["Health"] - out["Health_lag3"]
    out["Health_delta7"] = out["Health"] - out["Health_lag7"]

    out["Usage_Hrs_delta1"] = out["Usage_Hrs"] - out["Usage_Hrs_lag1"]
    out["Daily_KWh_delta1"] = out["Daily_KWh"] - out["Daily_KWh_lag1"]
    out["Voltage_delta1"] = out["Voltage"] - out["Voltage_lag1"]
    out["Current_delta1"] = out["Current"] - out["Current_lag1"]

    for col in base_cols:
        out[f"{col}_ma3"] = group[col].transform(lambda s: s.rolling(3, min_periods=1).mean())
        out[f"{col}_ma7"] = group[col].transform(lambda s: s.rolling(7, min_periods=1).mean())
        out[f"{col}_ma14"] = group[col].transform(lambda s: s.rolling(14, min_periods=1).mean())

        out[f"{col}_std3"] = group[col].transform(lambda s: s.rolling(3, min_periods=1).std())
        out[f"{col}_std7"] = group[col].transform(lambda s: s.rolling(7, min_periods=1).std())
        out[f"{col}_std14"] = group[col].transform(lambda s: s.rolling(14, min_periods=1).std())

    out["Temp_per_KWh"] = out["Peak_T"] / (out["Daily_KWh"] + 1e-6)
    out["Current_per_Usage"] = out["Current"] / (out["Usage_Hrs"] + 1e-6)
    out["Health_per_TotalKWh"] = out["Health"] / (out["Total_KWh"] + 1.0)
    out["Temp_per_Usage"] = out["Peak_T"] / (out["Usage_Hrs"] + 1e-6)
    out["Voltage_per_Current"] = out["Voltage"] / (np.abs(out["Current"]) + 1e-6)

    grp_cols = ["Spec", "Loc", "Day"]
    out["Group_Peak_T_Mean"] = out.groupby(grp_cols)["Peak_T"].transform("mean")
    out["Group_Health_Mean"] = out.groupby(grp_cols)["Health"].transform("mean")
    out["Group_Current_Mean"] = out.groupby(grp_cols)["Current"].transform("mean")
    out["Group_Usage_Mean"] = out.groupby(grp_cols)["Usage_Hrs"].transform("mean")

    out["Peak_T_vs_Group"] = out["Peak_T"] - out["Group_Peak_T_Mean"]
    out["Health_vs_Group"] = out["Health"] - out["Group_Health_Mean"]
    out["Current_vs_Group"] = out["Current"] - out["Group_Current_Mean"]
    out["Usage_vs_Group"] = out["Usage_Hrs"] - out["Group_Usage_Mean"]

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    out[numeric_cols] = out.groupby("ID", sort=False)[numeric_cols].transform(lambda x: x.bfill().ffill())
    out[numeric_cols] = out[numeric_cols].fillna(0)

    return out


# =========================================================
# 7. 데이터셋 생성
# =========================================================

def build_training_dataset(raw_df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = df.columns.str.strip()
    df = df.sort_values(["ID", "Day"]).reset_index(drop=True)

    df = add_current_danger_flag(df, cfg)
    df = add_future_risk_label(df, cfg)
    df = add_status_3class_label(df, cfg)
    df = add_time_to_risk_label(df, cfg)
    df = add_time_series_features(df)

    max_day_per_id = df.groupby("ID")["Day"].transform("max")
    df = df[df["Day"] <= (max_day_per_id - cfg.forecast_horizon)].copy()

    df = df.sort_values(["ID", "Day"]).reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude_cols = {
        "ID", "Spec", "Loc", "Scenario",
        "Danger_Now", "Label_RiskInHorizon", "Label_Status3", "Time_To_Risk"
    }
    return [c for c in df.columns if c not in exclude_cols]


# =========================================================
# 8. 분할
# =========================================================

def time_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15
):
    out = df.copy()
    max_day = int(out["Day"].max())

    train_end = int(max_day * train_ratio)
    valid_end = int(max_day * (train_ratio + valid_ratio))

    train_df = out[out["Day"] <= train_end].copy()
    valid_df = out[(out["Day"] > train_end) & (out["Day"] <= valid_end)].copy()
    test_df = out[out["Day"] > valid_end].copy()

    return train_df, valid_df, test_df


# =========================================================
# 9. 저장
# =========================================================

def save_config(cfg: DataConfig) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=4, ensure_ascii=False)


def save_datasets(
    meta_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_dir: str
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    meta_df.to_csv(os.path.join(save_dir, "device_meta.csv"), index=False)
    raw_df.to_csv(os.path.join(save_dir, "raw.csv"), index=False)
    processed_df.to_csv(os.path.join(save_dir, "processed.csv"), index=False)
    train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    valid_df.to_csv(os.path.join(save_dir, "valid.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)


# =========================================================
# 10. 실행
# =========================================================

def main():
    cfg = DataConfig(
        num_devices=300,
        num_days=365,
        forecast_horizon=7,
        random_state=42,
        save_dir="./EV_project/data"
    )

    print("========== Generating Meta / Raw Data ==========")
    meta_df, raw_df = generate_raw_timeseries(cfg)

    print("========== Building Processed Dataset ==========")
    processed_df = build_training_dataset(raw_df, cfg)

    print("========== Splitting Dataset ==========")
    train_df, valid_df, test_df = time_based_split(processed_df)

    print("========== Saving Files ==========")
    save_datasets(
        meta_df=meta_df,
        raw_df=raw_df,
        processed_df=processed_df,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        save_dir=cfg.save_dir,
    )
    save_config(cfg)

    print("\n========== Device Meta Preview ==========")
    print(meta_df.head())

    print("\n========== Scenario Distribution ==========")
    print(meta_df["Scenario"].value_counts())

    print("\n========== Raw Data Shape ==========")
    print(raw_df.shape)

    print("\n========== Processed Data Shape ==========")
    print(processed_df.shape)

    print("\n========== Label_Status3 Distribution ==========")
    print(processed_df["Label_Status3"].value_counts(normalize=True).sort_index())

    print("\n========== Label_RiskInHorizon Distribution ==========")
    print(processed_df["Label_RiskInHorizon"].value_counts(normalize=True).sort_index())

    print("\n========== Time_To_Risk Distribution ==========")
    print(processed_df["Time_To_Risk"].describe())

    print("\n========== Current Danger Distribution ==========")
    print(processed_df["Danger_Now"].value_counts(normalize=True))

    print("\n========== Split Sizes ==========")
    print("train :", train_df.shape)
    print("valid :", valid_df.shape)
    print("test  :", test_df.shape)

    print("\n========== Saved File List ==========")
    print(os.listdir(cfg.save_dir))

    print(f"\n✅ Saved all datasets to '{cfg.save_dir}'")


if __name__ == "__main__":
    main()