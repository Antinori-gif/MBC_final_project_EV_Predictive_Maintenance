from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
import joblib
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from psycopg2.extras import RealDictCursor

load_dotenv()

from EV_data import add_time_series_features
from LightGBM_train_ttr import InputFeatures, diagnose

app = FastAPI(title="EV Charger Predictive Maintenance API")


# =========================================================
# 1. 경로 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATUS_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_lightgbm_model.pkl")
STATUS_FEATURE_PATH = os.path.join(BASE_DIR, "models", "lightgbm_feature_columns.json")
REFERENCE_PATH = os.path.join(BASE_DIR, "data", "train.csv")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 2. DB 설정
# =========================================================
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "myDB"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}


# =========================================================
# 3. 모델 / 설정 로드
# =========================================================
for path, name in [
    (STATUS_MODEL_PATH, "상태 모델"),
    (STATUS_FEATURE_PATH, "상태 feature"),
]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} 파일이 없습니다: {path}")

status_model = joblib.load(STATUS_MODEL_PATH)

with open(STATUS_FEATURE_PATH, "r", encoding="utf-8") as f:
    status_feature_cols = json.load(f)

reference_df = None
reference_stats = {}

if os.path.exists(REFERENCE_PATH):
    reference_df = pd.read_csv(REFERENCE_PATH)
    reference_df.columns = reference_df.columns.str.strip()

    candidate_cols = [
        "Peak_T", "Peak_T_ma7", "Peak_T_ma14",
        "Health", "Health_ma14",
        "Current", "Current_std14",
        "Voltage_std14",
        "Temp_Change", "Health_Change",
    ]

    for col in candidate_cols:
        if col in reference_df.columns:
            ref_series = pd.to_numeric(reference_df[col], errors="coerce").dropna()
            if len(ref_series) > 0:
                mean = float(ref_series.mean())
                std = float(ref_series.std())
                if std == 0 or np.isnan(std):
                    std = 1.0
                reference_stats[col] = {"mean": mean, "std": std}

print("✅ 상태 모델 로드 완료")
print(f"✅ 상태 feature 개수: {len(status_feature_cols)}")
print("✅ ttr.py 연동 완료")


# =========================================================
# 4. 공통 유틸
# =========================================================
def get_connection():
    return psycopg2.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        dbname=DB_CONFIG["dbname"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
    )


def add_static_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "ID" not in out.columns:
        out["ID"] = "TEMP_DEVICE"

    if "Spec_Val" not in out.columns:
        spec_map = {
            "AC_7kW": 7,
            "DC_50kW": 50,
            "DC_100kW": 100,
        }
        if "Spec" not in out.columns:
            raise HTTPException(status_code=400, detail="Spec 컬럼이 필요합니다.")
        out["Spec_Val"] = out["Spec"].map(spec_map).fillna(7)

    if "Loc_Val" not in out.columns:
        if "Loc" not in out.columns:
            raise HTTPException(status_code=400, detail="Loc 컬럼이 필요합니다.")
        out["Loc_Val"] = out["Loc"].apply(lambda x: 1 if x == "Outdoor" else 0)

    return out


def align_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    aligned = df.copy()

    for col in feature_cols:
        if col not in aligned.columns:
            aligned[col] = 0.0

    aligned = aligned[feature_cols].copy()
    aligned = aligned.replace([np.inf, -np.inf], np.nan).fillna(0)

    return aligned


def class_to_status(pred_class: int) -> str:
    mapping = {
        0: "정상",
        1: "점검",
        2: "위험",
    }
    return mapping.get(pred_class, "알 수 없음")


def class_to_ai_status(pred_class: int) -> str:
    mapping = {
        0: "NORMAL",
        1: "CHECK",
        2: "RISK",
    }
    return mapping.get(pred_class, "NORMAL")


def class_to_action(pred_class: int) -> str:
    mapping = {
        0: "모니터링",
        1: "점검 필요",
        2: "현장 출동 및 작동 중지",
    }
    return mapping.get(pred_class, "상태 확인 필요")


def class_to_message(pred_class: int) -> str:
    mapping = {
        0: "현재 이상 없음",
        1: "이상 징후 감지, 점검 필요",
        2: "위험 상태 진입, 즉시 조치 필요",
    }
    return mapping.get(pred_class, "상태 확인 필요")


def feature_to_reason(feature: str, pred_class: int) -> str:
    warning_mapping = {
        "Peak_T": "내부 온도 상승",
        "Peak_T_ma7": "내부 온도 상승 감지",
        "Peak_T_ma14": "내부 온도 상승 감지",
        "Health": "기기 수명 저하",
        "Health_ma14": "기기 수명 저하",
        "Current": "전류 상승",
        "Current_std14": "전류 변동 감지",
        "Voltage_std14": "전압 변동 감지",
        "Temp_Change": "온도 변화 감지",
        "Health_Change": "기기 수명 감소",
    }

    risk_mapping = {
        "Peak_T": "과열 상태",
        "Peak_T_ma7": "과열 상태",
        "Peak_T_ma14": "과열 상태",
        "Health": "배터리 성능 저하",
        "Health_ma14": "기기 수명 저하",
        "Current": "과전류 발생",
        "Current_std14": "과전류 발생",
        "Voltage_std14": "전압 이상",
        "Temp_Change": "급격한 온도 상승",
        "Health_Change": "기기 수명 저하",
    }

    if pred_class == 1:
        return warning_mapping.get(feature, "이상 징후 감지")
    if pred_class == 2:
        return risk_mapping.get(feature, "이상 상태 발생")
    return ""


def extract_top_reason(
    latest_row: pd.Series,
    feature_cols: list[str],
    model,
    pred_class: int,
) -> str:
    if pred_class == 0:
        return ""

    candidates = [
        "Peak_T", "Peak_T_ma7", "Peak_T_ma14",
        "Health", "Health_ma14",
        "Current", "Current_std14",
        "Voltage_std14",
        "Temp_Change", "Health_Change",
    ]

    importances = dict(zip(feature_cols, model.feature_importances_))
    scored: list[tuple[str, float]] = []

    for f in candidates:
        if f not in latest_row.index or f not in importances:
            continue
        if f not in reference_stats:
            continue

        try:
            val = float(latest_row[f])
        except (TypeError, ValueError):
            continue

        mean = reference_stats[f]["mean"]
        std = reference_stats[f]["std"]
        z = abs((val - mean) / std)
        score = z * float(importances[f])
        scored.append((f, score))

    if not scored:
        for f in candidates:
            if f in latest_row.index:
                return feature_to_reason(f, pred_class)
        return "이상 징후 감지" if pred_class == 1 else "이상 상태 발생"

    scored.sort(key=lambda x: x[1], reverse=True)
    top_feature = scored[0][0]
    return feature_to_reason(top_feature, pred_class).strip()


def validate_history_df(df: pd.DataFrame):
    required_cols = [
        "Day", "Usage_Hrs", "Daily_KWh", "Total_KWh",
        "Voltage", "Current", "Peak_T", "Health",
        "Temp_Change", "Health_Change", "Spec", "Loc"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"입력 데이터에 필요한 컬럼이 없습니다: {missing}"
        )


def build_response(
    pred_class: int,
    status_probs: np.ndarray,
    ttr_result: dict,
    current_charger_status: str,
    inspection_requested: bool,
    latest: pd.Series,
    main_reason: str,
) -> dict:
    status = class_to_status(pred_class)
    ai_status = class_to_ai_status(pred_class)
    action = class_to_action(pred_class)
    message = class_to_message(pred_class)
    alarm = pred_class in [1, 2]

    raw_fault_prob = ttr_result.get("fault_prob_7d")
    raw_fault_prob = float(raw_fault_prob) if raw_fault_prob is not None else 0.0

    # 모델 상태는 그대로 쓰고, 확률만 상태에 맞게 보정
    if ai_status == "NORMAL":
        fault_prob_7d = max(0.0, min(raw_fault_prob, 0.29))
    elif ai_status == "CHECK":
        fault_prob_7d = max(0.30, min(raw_fault_prob, 0.49))
    else:  # RISK
        fault_prob_7d = max(0.50, min(raw_fault_prob, 1.0))

    if current_charger_status == "POWER_OFF":
        device_status = "위험 감지로 인한 강제 중지"
    elif current_charger_status == "CHARGING":
        device_status = "작동 중"
    else:
        device_status = "대기 중"

    result = {
        "pred_class": pred_class,
        "status": status,
        "ai_status": ai_status,
        "action": None if pred_class == 0 else action,
        "alarm": alarm,
        "message": message,
        "main_reason": None if pred_class == 0 else main_reason,
        "device_status": device_status,
        "inspection_requested": inspection_requested,
        "fault_prob_7d": fault_prob_7d,
        "prob_normal": float(status_probs[0]),
        "prob_check": float(status_probs[1]),
        "prob_risk": float(status_probs[2]),
        "temperature": float(latest["Peak_T"]) if pd.notna(latest["Peak_T"]) else None,
        "voltage": float(latest["Voltage"]) if pd.notna(latest["Voltage"]) else None,
        "current": float(latest["Current"]) if pd.notna(latest["Current"]) else None,
    }

    return result


def run_prediction_from_history(
    history: list,
    current_charger_status: str,
    inspection_requested: bool,
) -> tuple[dict, pd.Series]:
    if not isinstance(history, list) or len(history) == 0:
        raise HTTPException(status_code=400, detail="history는 비어 있지 않은 리스트여야 합니다.")

    df = pd.DataFrame(history)
    df.columns = df.columns.str.strip()

    validate_history_df(df)

    df = add_static_columns(df)
    df = add_time_series_features(df)

    latest = df.iloc[-1].copy()

    X_status = pd.DataFrame([latest])
    X_status = align_features(X_status, status_feature_cols)

    status_probs = status_model.predict_proba(X_status)[0]
    pred_class = int(np.argmax(status_probs))
    
    ttr_features = InputFeatures(
        health=float(latest["Health"]),
        peak_temp=float(latest["Peak_T"]),
        usage_hours=float(latest["Usage_Hrs"]),
        voltage=float(latest["Voltage"]),
        current=float(latest["Current"]),
        is_already_fault=False,
    )
    
    ttr_result = diagnose(ttr_features).to_dict()
    
    # 최종 상태는 TTR 기준으로 확정
    ttr_state = ttr_result.get("state")
    if ttr_state == "정상":
        pred_class = 0
    elif ttr_state == "점검":
        pred_class = 1
    elif ttr_state == "위험":
        pred_class = 2
    
    # 최종 pred_class가 확정된 뒤에 main_reason 추출
    main_reason = extract_top_reason(
        latest_row=latest,
        feature_cols=status_feature_cols,
        model=status_model,
        pred_class=pred_class,
    )
    
    result = build_response(
        pred_class=pred_class,
        status_probs=status_probs,
        ttr_result=ttr_result,
        current_charger_status=current_charger_status,
        inspection_requested=inspection_requested,
        latest=latest,
        main_reason=main_reason,
    )

    return result, latest


# =========================================================
# 5. DB 조회 유틸
# =========================================================
def map_charger_type_to_spec(charger_type: str) -> str:
    if charger_type == "FAST":
        return "DC_100kW"
    return "AC_7kW"


def reason_to_flags(main_reason: str | None) -> tuple[bool, bool, bool]:
    if not main_reason:
        return False, False, False

    reason = str(main_reason)
    temperature_flag = ("온도" in reason) or ("과열" in reason)
    voltage_flag = "전압" in reason
    current_flag = "전류" in reason

    return temperature_flag, voltage_flag, current_flag


def map_ai_status_to_charger_status(
    ai_status: str,
    current_charger_status: str,
) -> str:
    # 강제 종료 / 고장 상태는 예측 결과로 덮어쓰지 않음
    if current_charger_status in ("POWER_OFF", "FAULT"):
        return current_charger_status

    if ai_status == "NORMAL":
        if current_charger_status == "CHARGING":
            return "CHARGING"
        return "STANDBY"

    if ai_status == "CHECK":
        return "CHECK"

    if ai_status == "RISK":
        return "RISK"

    return current_charger_status


def get_connection():
    return psycopg2.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        dbname=DB_CONFIG["dbname"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
    )


def fetch_prediction_input_from_db(charger_id: str, limit: int = 14) -> tuple[list[dict], str, bool]:
    conn = None
    cur = None

    try:
        if limit <= 0:
            raise HTTPException(status_code=400, detail="limit는 1 이상이어야 합니다.")

        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        state_query = """
            SELECT
                CASE
                    WHEN c.charger_status = 'CHARGING' THEN TRUE
                    ELSE FALSE
                END AS is_operating,
                CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM ev_issue_log il
                        WHERE il.ev_charger_id = c.ev_charger_id
                          AND il.process_status IN ('INSPECTION_SENT', 'CHECK_IN_PROGRESS')
                    )
                    THEN TRUE
                    ELSE FALSE
                END AS inspection_requested,
                c.charger_type,
                c.charger_status,
                ps.parking_floor
            FROM ev_charger c
            JOIN parking_spot ps
              ON ps.parking_spot_id = c.parking_spot_id
            WHERE c.ev_charger_id = %s
        """
        cur.execute(state_query, (charger_id,))
        state_row = cur.fetchone()

        if not state_row:
            raise HTTPException(status_code=404, detail=f"charger_id={charger_id} 충전기 정보가 없습니다.")

        history_query = """
            WITH latest_sensor AS (
                SELECT
                    s.ev_sensor_log_id,
                    s.ev_charger_id,
                    s.measured_time,
                    s.temperature,
                    s.voltage,
                    s.current
                FROM ev_sensor_log s
                WHERE s.ev_charger_id = %s
                ORDER BY s.measured_time DESC
                LIMIT %s
            ),
            ordered_sensor AS (
                SELECT *
                FROM latest_sensor
                ORDER BY measured_time ASC
            )
            SELECT *
            FROM (
                SELECT
                    ROW_NUMBER() OVER (ORDER BY s.measured_time ASC) AS "Day",

                    CASE
                        WHEN c.charger_status = 'CHARGING' AND cl.start_time IS NOT NULL THEN
                            COALESCE(
                                EXTRACT(EPOCH FROM (
                                    COALESCE(cl.end_time, s.measured_time) - cl.start_time
                                )) / 3600.0,
                                0
                            )
                        ELSE 0
                    END AS "Usage_Hrs",

                    COALESCE(cl.total_charge_kwh, 0) AS "Daily_KWh",

                    COALESCE(
                        SUM(COALESCE(cl.total_charge_kwh, 0)) OVER (
                            ORDER BY s.measured_time ASC
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ),
                        0
                    ) AS "Total_KWh",

                    COALESCE(s.voltage, 0) AS "Voltage",
                    COALESCE(s.current, 0) AS "Current",
                    COALESCE(s.temperature, 0) AS "Peak_T",

                    100.0 AS "Health",

                    COALESCE(
                        s.temperature
                        - LAG(s.temperature) OVER (ORDER BY s.measured_time ASC),
                        0
                    ) AS "Temp_Change",

                    0.0 AS "Health_Change",

                    CASE
                        WHEN c.charger_type = 'FAST' THEN 'DC_100kW'
                        ELSE 'AC_7kW'
                    END AS "Spec",

                    'Indoor' AS "Loc",
                    c.ev_charger_id AS "ID",
                    s.measured_time

                FROM ordered_sensor s
                JOIN ev_charger c
                  ON c.ev_charger_id = s.ev_charger_id

                LEFT JOIN LATERAL (
                    SELECT cl.*
                    FROM ev_charging_log cl
                    WHERE cl.ev_charger_id = s.ev_charger_id
                      AND cl.start_time <= s.measured_time
                    ORDER BY cl.start_time DESC
                    LIMIT 1
                ) cl ON TRUE
            ) t
            ORDER BY t.measured_time ASC
        """
        cur.execute(history_query, (charger_id, limit))
        rows = cur.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail=f"charger_id={charger_id} 이력 데이터가 없습니다.")

        history = []
        for row in rows:
            row_dict = dict(row)
            row_dict.pop("measured_time", None)
            history.append(row_dict)

        current_charger_status = state_row["charger_status"]
        inspection_requested = bool(state_row["inspection_requested"])

        return history, current_charger_status, inspection_requested

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 예측 입력 조회 실패: {str(e)}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def insert_prediction_result(conn, charger_id: str, result: dict):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ev_prediction_result (
                ev_charger_id,
                predicted_time,
                ai_status,
                fault_prob_7d,
                main_reason,
                prob_normal,
                prob_check,
                prob_risk,
                temperature_value,
                voltage_value,
                current_value
            )
            VALUES (
                %s, now(), %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """,
            (
                charger_id,
                result.get("ai_status"),
                result.get("fault_prob_7d"),
                result.get("main_reason"),
                result.get("prob_normal"),
                result.get("prob_check"),
                result.get("prob_risk"),
                result.get("temperature"),
                result.get("voltage"),
                result.get("current"),
            ),
        )


def upsert_issue_log(
    conn,
    charger_id: str,
    prediction_result_id: int,
    ai_status: str,
    main_reason: str | None,
):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if ai_status == "NORMAL":
            cur.execute(
                """
                UPDATE ev_issue_log
                SET process_status = 'STATUS_UPDATED'
                WHERE ev_issue_log_id = (
                    SELECT ev_issue_log_id
                    FROM ev_issue_log
                    WHERE ev_charger_id = %s
                      AND process_status NOT IN ('DONE', 'STATUS_UPDATED')
                    ORDER BY occurred_time DESC, ev_issue_log_id DESC
                    LIMIT 1
                )
                """,
                (charger_id,),
            )
            return

        if ai_status not in ("CHECK", "RISK"):
            return

    issue_status = ai_status
    detail_content = main_reason if main_reason else ("점검 필요" if ai_status == "CHECK" else "위험 상태")
    temperature_flag, voltage_flag, current_flag = reason_to_flags(main_reason)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                ev_issue_log_id,
                issue_status,
                process_status,
                power_off_done
            FROM ev_issue_log
            WHERE ev_charger_id = %s
                AND process_status NOT IN ('DONE', 'STATUS_UPDATED')
            ORDER BY occurred_time DESC, ev_issue_log_id DESC
            LIMIT 1
            """,
            (charger_id,),
        )
        latest_issue = cur.fetchone()

        if not latest_issue:
            cur.execute(
                """
                INSERT INTO ev_issue_log (
                    ev_charger_id,
                    ev_prediction_result_id,
                    issue_status,
                    process_status,
                    occurred_time,
                    detail_content,
                    temperature_flag,
                    voltage_flag,
                    current_flag,
                    power_off_done
                )
                VALUES (
                    %s, %s, %s, 'UNPROCESSED', now(), %s, %s, %s, %s, FALSE
                )
                """,
                (
                    charger_id,
                    prediction_result_id,
                    issue_status,
                    detail_content,
                    temperature_flag,
                    voltage_flag,
                    current_flag,
                ),
            )
            return

        latest_issue_id = latest_issue["ev_issue_log_id"]
        latest_issue_status = latest_issue["issue_status"]
        latest_process_status = latest_issue["process_status"]
        latest_power_off_done = latest_issue["power_off_done"]

        if latest_issue_status == issue_status:
            return

        cur.execute(
            """
            UPDATE ev_issue_log
            SET process_status = 'STATUS_UPDATED'
            WHERE ev_issue_log_id = %s
            """,
            (latest_issue_id,),
        )
        # 정상 상태 이력이 없고 새로 점검/위험이 들어온 경우, 또는 그 외 새 이슈 시작
        cur.execute(
            """
            INSERT INTO ev_issue_log (
                ev_charger_id,
                ev_prediction_result_id,
                issue_status,
                process_status,
                occurred_time,
                detail_content,
                temperature_flag,
                voltage_flag,
                current_flag,
                power_off_done
            )
            VALUES (
                %s, %s, %s, 'UNPROCESSED', now(), %s, %s, %s, %s, FALSE
            )
            """,
            (
                charger_id,
                prediction_result_id,
                issue_status,
                detail_content,
                temperature_flag,
                voltage_flag,
                current_flag,
            ),
        )


def update_ev_charger_status(conn, charger_id: str, new_status: str):
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ev_charger
            SET charger_status = %s
            WHERE ev_charger_id = %s
            """,
            (new_status, charger_id),
        )


# =========================================================
# 6. 기본 엔드포인트
# =========================================================
@app.get("/")
def home():
    return {"message": "OK"}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "status_model_loaded": True,
        "status_num_features": len(status_feature_cols),
        "ttr_linked": True,
    }


# =========================================================
# 7. 테스트용 예측 엔드포인트
# =========================================================
@app.post("/predict")
def predict(data: dict):
    if "history" not in data:
        raise HTTPException(status_code=400, detail="history 필드가 필요합니다.")

    history = data["history"]
    current_charger_status = data.get("charger_status", "STANDBY")
    inspection_requested = data.get("inspection_requested", False)

    result, _latest = run_prediction_from_history(
        history=history,
        current_charger_status=current_charger_status,
        inspection_requested=inspection_requested,
    )

    return result


# =========================================================
# 8. DB 연동용 예측 엔드포인트
# =========================================================
@app.get("/predict/db/{charger_id}")
def predict_from_db(charger_id: str, limit: int = 14):
    history, current_charger_status, inspection_requested = fetch_prediction_input_from_db(
        charger_id=charger_id,
        limit=limit,
    )

    result, latest = run_prediction_from_history(
        history=history,
        current_charger_status=current_charger_status,
        inspection_requested=inspection_requested,
    )

    new_charger_status = map_ai_status_to_charger_status(
        result["ai_status"],
        current_charger_status,
    )

    # -------------------------------
    # 강제 종료 상태 유지
    # -------------------------------
    shutdown_done = current_charger_status == "POWER_OFF"
    if shutdown_done:
        new_charger_status = "POWER_OFF"
        result["device_status"] = "전원꺼짐"

    conn = None
    try:
        conn = get_connection()

        with conn.cursor() as cur:
            insert_prediction_result(conn, charger_id, result)

            cur.execute("SELECT currval(pg_get_serial_sequence('ev_prediction_result', 'ev_prediction_result_id'))")
            prediction_result_id = cur.fetchone()[0]

            upsert_issue_log(
                conn=conn,
                charger_id=charger_id,
                prediction_result_id=prediction_result_id,
                ai_status=result["ai_status"],
                main_reason=result.get("main_reason"),
            )

            update_ev_charger_status(conn, charger_id, new_charger_status)

        conn.commit()

    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"예측 결과 저장 실패: {str(e)}")

    finally:
        if conn:
            conn.close()

    # -------------------------------
    # 프론트 버튼 유지용 응답값 추가
    # -------------------------------
    result["inspection_requested"] = inspection_requested
    result["shutdown_done"] = shutdown_done

    if shutdown_done:
        result["device_status"] = "전원꺼짐"

    return result


# =========================================================
# 9. 최신 예측 결과 조회
# =========================================================
@app.get("/prediction/latest/{charger_id}")
def get_latest_prediction(charger_id: str):
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute(
            """
            SELECT
                ev_prediction_result_id,
                ev_charger_id,
                predicted_time,
                ai_status,
                fault_prob_7d,
                main_reason,
                prob_normal,
                prob_check,
                prob_risk,
                temperature_value,
                voltage_value,
                current_value
            FROM ev_prediction_result
            WHERE ev_charger_id = %s
            ORDER BY predicted_time DESC, ev_prediction_result_id DESC
            LIMIT 1
            """,
            (charger_id,),
        )

        row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"charger_id={charger_id} 예측 결과가 없습니다.")

        return dict(row)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"최신 예측 결과 조회 실패: {str(e)}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


# =========================================================
# 10. 장애 이력 조회
# =========================================================
@app.get("/issue-log")
def get_all_issue_log(limit: int = 10):
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute(
            """
            SELECT
                il.ev_issue_log_id,
                il.ev_charger_id,
                il.ev_prediction_result_id,
                il.issue_status,
                il.process_status,
                il.occurred_time,
                il.detail_content,
                il.temperature_flag,
                il.voltage_flag,
                il.current_flag,
                il.power_off_done,
                pr.fault_prob_7d
            FROM ev_issue_log il
            LEFT JOIN ev_prediction_result pr
              ON pr.ev_prediction_result_id = il.ev_prediction_result_id
            ORDER BY il.occurred_time DESC, il.ev_issue_log_id DESC
            LIMIT %s
            """,
            (limit,),
        )

        rows = cur.fetchall()
        return [dict(row) for row in rows]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"전체 장애 이력 조회 실패: {str(e)}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            
# =========================================================
# 11. 센서 추이 조회
# =========================================================
@app.get("/sensor-history/{charger_id}")
def get_sensor_history(charger_id: str, limit: int = 30):
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute(
            """
            WITH latest_sensor AS (
                SELECT
                    ev_sensor_log_id,
                    ev_charger_id,
                    measured_time,
                    temperature,
                    voltage,
                    current
                FROM ev_sensor_log
                WHERE ev_charger_id = %s
                ORDER BY measured_time DESC
                LIMIT %s
            )
            SELECT
                s.ev_sensor_log_id,
                s.ev_charger_id,
                s.measured_time,
                s.temperature,
                COALESCE(
                    s.temperature - LAG(s.temperature) OVER (ORDER BY s.measured_time ASC),
                    0
                ) AS temperature_change,
                s.voltage,
                COALESCE(
                    s.voltage - LAG(s.voltage) OVER (ORDER BY s.measured_time ASC),
                    0
                ) AS voltage_change,
                s.current,
                COALESCE(
                    s.current - LAG(s.current) OVER (ORDER BY s.measured_time ASC),
                    0
                ) AS current_change
            FROM (
                SELECT *
                FROM latest_sensor
                ORDER BY measured_time ASC
            ) s
            ORDER BY s.measured_time ASC
            """,
            (charger_id, limit),
        )

        rows = cur.fetchall()
        return [dict(row) for row in rows]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"센서 이력 조회 실패: {str(e)}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


# =========================================================
# 12. 충전기 목록 + 최신 상태 요약
# =========================================================
@app.get("/chargers/summary")
def get_chargers_summary():
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute(
            """
            SELECT
                c.ev_charger_id,
                c.parking_spot_id,
                c.charger_type,
                c.charger_status,
                c.create_time,

                ps.parking_floor,
                ps.parking_row,
                ps.parking_column,

                p.ai_status,
                p.fault_prob_7d,
                p.main_reason,
                p.prob_normal,
                p.prob_check,
                p.prob_risk,
                p.predicted_time,

                il.issue_status,
                il.process_status,
                il.detail_content,
                il.power_off_done,
                il.occurred_time

            FROM ev_charger c
            JOIN parking_spot ps
              ON ps.parking_spot_id = c.parking_spot_id

            LEFT JOIN LATERAL (
                SELECT
                    p.ai_status,
                    p.fault_prob_7d,
                    p.main_reason,
                    p.prob_normal,
                    p.prob_check,
                    p.prob_risk,
                    p.predicted_time
                FROM ev_prediction_result p
                WHERE p.ev_charger_id = c.ev_charger_id
                ORDER BY p.predicted_time DESC, p.ev_prediction_result_id DESC
                LIMIT 1
            ) p ON TRUE

            LEFT JOIN LATERAL (
                SELECT
                    il.issue_status,
                    il.process_status,
                    il.detail_content,
                    il.power_off_done,
                    il.occurred_time
                FROM ev_issue_log il
                WHERE il.ev_charger_id = c.ev_charger_id
                ORDER BY il.occurred_time DESC, il.ev_issue_log_id DESC
                LIMIT 1
            ) il ON TRUE

            ORDER BY ps.parking_floor, ps.parking_row, ps.parking_column, c.ev_charger_id
            """
        )

        rows = cur.fetchall()
        return [dict(row) for row in rows]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"충전기 요약 조회 실패: {str(e)}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

# =========================================================
# 13. 점검 요청
# =========================================================
class InspectionRequestBody(BaseModel):
    chargerId: str
    targetDeptName: str
    aiStatus: str | None = None
    faultProb7d: float | None = None
    mainReason: str | None = None
    requestReason: str
    reasonTypes: dict | None = None


@app.post("/inspection-request")
def inspection_request(body: InspectionRequestBody):
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute(
            """
            SELECT ev_issue_log_id, ev_charger_id, process_status
            FROM ev_issue_log
            WHERE ev_charger_id = %s
            ORDER BY occurred_time DESC, ev_issue_log_id DESC
            LIMIT 1
            """,
            (body.chargerId,),
        )

        row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="점검 요청 대상 장애 이력이 없습니다.")

        ev_issue_log_id = row["ev_issue_log_id"]

        cur.execute(
            """
            UPDATE ev_issue_log
            SET process_status = %s
            WHERE ev_issue_log_id = %s
            """,
            ("INSPECTION_SENT", ev_issue_log_id),
        )

        if cur.rowcount != 1:
            raise HTTPException(status_code=404, detail="점검 요청 대상 장애 이력 업데이트에 실패했습니다.")

        conn.commit()

        return {
            "success": True,
            "ev_issue_log_id": ev_issue_log_id,
            "charger_id": row["ev_charger_id"],
            "process_status": "INSPECTION_SENT"
        }

    except HTTPException:
        if conn:
            conn.rollback()
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        print("inspection_request error:", repr(e))
        raise HTTPException(status_code=500, detail=f"점검 요청 처리 실패: {repr(e)}")

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


# ==========================================
# 14. 전원 차단
# ==========================================
class ForceShutdownBody(BaseModel):
    chargerId: str
    commandType: str = "POWER_OFF"
    confirmedTwice: bool = True


@app.post("/control/power-off")
def power_off(body: ForceShutdownBody):
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            UPDATE ev_charger
            SET charger_status = 'POWER_OFF'
            WHERE ev_charger_id = %s
            """,
            (body.chargerId,),
        )

        cur.execute(
            """
            UPDATE ev_issue_log
            SET power_off_done = TRUE
            WHERE ev_charger_id = %s
            """,
            (body.chargerId,),
        )

        conn.commit()
        return {"success": True}

    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"강제 종료 처리 실패: {str(e)}")

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()