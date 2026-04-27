import os
import time
import random
import math
from datetime import datetime

import psycopg2
import requests
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "myDB"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")
INSERT_INTERVAL_SECONDS = 10


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def fetch_chargers():
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute(
            """
            SELECT
                ev_charger_id,
                charger_status,
                charger_type
            FROM ev_charger
            ORDER BY ev_charger_id
            """
        )

        return [dict(row) for row in cur.fetchall()]

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def update_charger_status(ev_charger_id, charger_status: str):
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            UPDATE ev_charger
            SET charger_status = %s
            WHERE ev_charger_id = %s
            """,
            (charger_status, ev_charger_id),
        )

        conn.commit()

    except Exception:
        if conn:
            conn.rollback()
        raise

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def fetch_charger_status_from_db(ev_charger_id) -> str:
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT charger_status FROM ev_charger WHERE ev_charger_id = %s",
            (ev_charger_id,),
        )
        row = cur.fetchone()
        return row[0] if row else "STANDBY"

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def generate_power_off_sensor(prev: dict) -> dict:
    t = max(float(prev["temperature"]) - 2.5 + random.uniform(-0.2, 0.2), 22.0)
    v = max(float(prev["voltage"]) - 3.5 + random.uniform(-0.1, 0.1), 0.0)
    c = max(float(prev["current"]) - 1.5 + random.uniform(-0.05, 0.05), 0.0)
    return {
        "temperature": round(t, 2),
        "voltage": round(v, 2),
        "current": round(c, 2),
    }


def insert_sensor_log(ev_charger_id, sensor: dict):
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO ev_sensor_log (
                ev_charger_id,
                measured_time,
                temperature,
                voltage,
                current
            )
            VALUES (%s, now(), %s, %s, %s)
            """,
            (
                ev_charger_id,
                sensor["temperature"],
                sensor["voltage"],
                sensor["current"],
            ),
        )

        conn.commit()

    except Exception:
        if conn:
            conn.rollback()
        raise

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def trigger_prediction(ev_charger_id):
    response = requests.get(f"{FASTAPI_BASE_URL}/predict/db/{ev_charger_id}", timeout=10)
    response.raise_for_status()
    return response.json()


def get_scenario(ev_charger_id) -> str:
    ev_charger_id = str(ev_charger_id).strip()

    # A: 충전 정상 (8대)
    charging_ids = {
        "1F-D-08", "1F-D-09", "1F-D-10",
        "2F-D-08", "2F-D-09", "2F-D-10",
        "3F-D-08", "3F-D-09",
    }

    # D: 항상 대기중 (4대)
    standby_ids = {
        "3F-D-10",
        "4F-D-08", "4F-D-09", "4F-D-10",
    }

    # B: 점검 도달 (2대)
    check_ids = {
        "5F-D-08", "5F-D-09",
    }

    # C: 위험 도달 (1대)
    risk_ids = {
        "5F-D-10",
    }

    if ev_charger_id in charging_ids:
        return "A"
    if ev_charger_id in standby_ids:
        return "D"
    if ev_charger_id in check_ids:
        return "B"
    if ev_charger_id in risk_ids:
        return "C"

    return "A"


def build_phase_sequence(scenario: str):
    # A: 충전 정상 — 약 6분 후 충전중 진입, 이후 무한 유지
    if scenario == "A":
        return [
            {"name": "standby_normal", "cycles": 3, "status": "STANDBY"},
            {"name": "charging_ramp", "cycles": 5, "status": "CHARGING"},
            {"name": "charging_normal", "cycles": 999999, "status": "CHARGING"},
        ]

    # D: 항상 대기중 — 충전하지 않음
    if scenario == "D":
        return [
            {"name": "standby_normal", "cycles": 999999, "status": "STANDBY"},
        ]

    # B: 점검 도달 — 약 17분(22사이클) 후 check_hold 진입
    if scenario == "B":
        return [
            {"name": "standby_normal", "cycles": 3, "status": "STANDBY"},
            {"name": "charging_ramp", "cycles": 5, "status": "CHARGING"},
            {"name": "charging_normal", "cycles": 9, "status": "CHARGING"},
            {"name": "rising_to_check", "cycles": 5, "status": "CHARGING"},
            {"name": "check_hold", "cycles": 999999, "status": "CHARGING"},
        ]

    # C: 위험 도달 — 약 26분(33사이클) 후 risk_hold 진입
    if scenario == "C":
        return [
            {"name": "standby_normal", "cycles": 3, "status": "STANDBY"},
            {"name": "charging_ramp", "cycles": 5, "status": "CHARGING"},
            {"name": "charging_normal", "cycles": 9, "status": "CHARGING"},
            {"name": "rising_to_check", "cycles": 5, "status": "CHARGING"},
            {"name": "check_hold", "cycles": 6, "status": "CHARGING"},
            {"name": "rising_to_risk", "cycles": 5, "status": "CHARGING"},
            {"name": "risk_hold", "cycles": 999999, "status": "CHARGING"},
        ]

    return [
        {"name": "standby_normal", "cycles": 3, "status": "STANDBY"},
        {"name": "charging_ramp", "cycles": 5, "status": "CHARGING"},
        {"name": "charging_normal", "cycles": 999999, "status": "CHARGING"},
    ]


def build_initial_sensor_by_phase(phase_name: str, charger_type: str) -> dict:
    if phase_name == "standby_normal":
        return {"temperature": 34.0, "voltage": 223.0, "current": 0.5}

    if phase_name == "charging_ramp":
        return {"temperature": 34.2, "voltage": 223.0, "current": 0.9}

    if phase_name == "charging_normal":
        return {"temperature": 41.0, "voltage": 221.4, "current": 4.8}

    if phase_name == "rising_to_check":
        return {"temperature": 48.0, "voltage": 220.4, "current": 9.5}

    if phase_name == "check_hold":
        return {"temperature": 67.5, "voltage": 216.2, "current": 35.2}

    if phase_name == "rising_to_risk":
        return {"temperature": 73.0, "voltage": 214.8, "current": 36.5}

    if phase_name == "risk_hold":
        return {"temperature": 88.2, "voltage": 210.2, "current": 42.2}

    return {"temperature": 34.0, "voltage": 223.0, "current": 0.5}


_WAVE_WRAP = 100  # wave() 주기 계산이 유효한 최대 cycle 값


def next_phase_state(state: dict):
    sequence = state["sequence"]
    phase_index = state["phase_index"]
    phase_cycle = state["phase_cycle"]

    phase = sequence[phase_index]
    phase_cycle += 1

    if phase_cycle >= phase["cycles"]:
        if phase_index < len(sequence) - 1:
            phase_index += 1
            phase_cycle = 0
        else:
            # 마지막 페이즈: wave()가 발산하지 않도록 cycle을 wrap
            phase_cycle = phase_cycle % _WAVE_WRAP

    state["phase_index"] = phase_index
    state["phase_cycle"] = phase_cycle
    return state


def get_current_phase(state: dict):
    return state["sequence"][state["phase_index"]]


def generate_sensor_by_phase(prev: dict, phase_name: str, charger_type: str, phase_cycle: int) -> dict:
    t = float(prev["temperature"])
    v = float(prev["voltage"])
    c = float(prev["current"])

    def n(x):
        return random.uniform(-x, x)

    def wave(amplitude: float, period: float, cycle: int, shift: float = 0.0) -> float:
        if period <= 0:
            return 0.0
        return amplitude * math.sin((cycle / period) * 2 * math.pi + shift)

    if phase_name == "standby_normal":
        t = clamp(34.0 + wave(0.5, 8, phase_cycle) + n(0.18), 33.1, 35.1)
        v = clamp(223.0 + wave(0.35, 10, phase_cycle, 0.7) + n(0.08), 222.2, 223.8)
        c = clamp(0.5 + wave(0.18, 6, phase_cycle, 1.1) + n(0.05), 0.18, 0.92)

    elif phase_name == "charging_ramp":
        t = clamp(t + 0.32 + wave(0.12, 6, phase_cycle) + n(0.03), 34.2, 38.2)
        v = clamp(v - 0.05 + wave(0.08, 7, phase_cycle, 0.3) + n(0.02), 222.1, 223.1)
        c = clamp(c + 0.18 + wave(0.10, 5, phase_cycle, 0.8) + n(0.02), 0.9, 2.6)

    elif phase_name == "charging_normal":
        target_t = 41.0 + wave(1.1, 9, phase_cycle)
        target_v = 221.4 + wave(0.45, 11, phase_cycle, 0.6)
        target_c = 4.8 + wave(0.75, 8, phase_cycle, 1.0)

        t = clamp(t + (target_t - t) * 0.24 + n(0.08), 39.4, 42.8)
        v = clamp(v + (target_v - v) * 0.22 + n(0.03), 220.6, 222.1)
        c = clamp(c + (target_c - c) * 0.24 + n(0.05), 3.7, 5.9)

    elif phase_name == "rising_to_check":
        t = clamp(t + 1.75 + wave(0.65, 4, phase_cycle) + n(0.10), 48.0, 67.2)
        v = clamp(v - 0.30 + wave(0.22, 5, phase_cycle, 0.2) + n(0.03), 216.4, 220.5)
        c = clamp(c + 2.20 + wave(0.75, 4, phase_cycle, 0.4) + n(0.08), 9.5, 34.8)

    elif phase_name == "check_hold":
        target_t = 67.8 + wave(1.6, 7, phase_cycle)
        target_v = 216.1 + wave(0.55, 8, phase_cycle, 0.5)
        target_c = 35.2 + wave(1.1, 6, phase_cycle, 0.9)

        t = clamp(t + (target_t - t) * 0.28 + n(0.08), 65.8, 70.4)
        v = clamp(v + (target_v - v) * 0.22 + n(0.03), 215.0, 217.0)
        c = clamp(c + (target_c - c) * 0.24 + n(0.05), 33.6, 36.8)

    elif phase_name == "rising_to_risk":
        t = clamp(t + 1.95 + wave(0.85, 4, phase_cycle) + n(0.12), 73.0, 87.4)
        v = clamp(v - 0.32 + wave(0.24, 5, phase_cycle, 0.1) + n(0.03), 210.6, 214.8)
        c = clamp(c + 0.78 + wave(0.42, 4, phase_cycle, 0.7) + n(0.05), 36.5, 42.1)

    elif phase_name == "risk_hold":
        target_t = 88.4 + wave(1.8, 6, phase_cycle)
        target_v = 210.2 + wave(0.65, 7, phase_cycle, 0.6)
        target_c = 42.2 + wave(0.95, 5, phase_cycle, 0.9)

        t = clamp(t + (target_t - t) * 0.28 + n(0.08), 85.8, 91.2)
        v = clamp(v + (target_v - v) * 0.22 + n(0.03), 209.0, 211.4)
        c = clamp(c + (target_c - c) * 0.24 + n(0.04), 40.8, 43.6)

    elif phase_name == "power_on_restart":
        # 전원 복구 후 standby 수준으로 서서히 복귀
        t = clamp(t + (34.0 - t) * 0.30 + n(0.05), 22.0, 36.0)
        v = clamp(v + (223.0 - v) * 0.35 + n(0.05), 0.0, 224.0)
        c = clamp(c + (0.5 - c) * 0.25 + n(0.02), 0.0, 1.0)

    else:
        t = clamp(34.0 + wave(0.5, 8, phase_cycle) + n(0.18), 33.1, 35.1)
        v = clamp(223.0 + wave(0.35, 10, phase_cycle, 0.7) + n(0.08), 222.2, 223.8)
        c = clamp(0.5 + wave(0.18, 6, phase_cycle, 1.1) + n(0.05), 0.18, 0.92)

    return {
        "temperature": round(t, 2),
        "voltage": round(v, 2),
        "current": round(c, 2),
    }


def main():
    print("sensor_simulator started")

    chargers = fetch_chargers()
    if not chargers:
        print("ev_charger 테이블에 충전기 데이터가 없습니다.")
        return

    sim_state = {}

    for charger in chargers:
        ev_charger_id = charger["ev_charger_id"]
        charger_type = str(charger.get("charger_type") or "FAST").strip()

        scenario = get_scenario(ev_charger_id)
        sequence = build_phase_sequence(scenario)
        first_phase = sequence[0]

        sim_state[ev_charger_id] = {
            "scenario": scenario,
            "charger_type": charger_type,
            "sequence": sequence,
            "phase_index": 0,
            "phase_cycle": 0,
            "sensor": build_initial_sensor_by_phase(first_phase["name"], charger_type),
        }

        update_charger_status(ev_charger_id, first_phase["status"])

        print(
            f"INIT | ID={ev_charger_id} | TYPE={charger_type} | "
            f"SCENARIO={scenario} | START_PHASE={first_phase['name']}"
        )

    print(f"시뮬레이터 시작 - 대상 충전기 수: {len(chargers)} / 주기: {INSERT_INTERVAL_SECONDS}초")

    while True:
        now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{now_text}] 센서 생성 시작")

        for charger in chargers:
            ev_charger_id = charger["ev_charger_id"]

            try:
                state = sim_state[ev_charger_id]
                charger_type = state["charger_type"]

                # 프론트/외부에서 POWER_OFF가 설정됐는지 확인
                actual_db_status = fetch_charger_status_from_db(ev_charger_id)
                was_power_off = state.get("power_off", False)
                is_power_off = actual_db_status == "POWER_OFF"

                if is_power_off:
                    # 강제종료 상태: 냉각 센서 삽입만, AI 재예측 금지 (상태 고정)
                    state["power_off"] = True
                    next_sensor = generate_power_off_sensor(state["sensor"])
                    insert_sensor_log(ev_charger_id, next_sensor)
                    print(
                        f"{ev_charger_id} | POWER_OFF (냉각 중) | "
                        f"T={next_sensor['temperature']}℃ "
                        f"V={next_sensor['voltage']}V "
                        f"I={next_sensor['current']}A | AI 상태 고정"
                    )
                    state["sensor"] = next_sensor
                    sim_state[ev_charger_id] = state
                    continue

                # POWER_OFF → 정상 복귀 시 재시작 시퀀스 주입
                if was_power_off:
                    state["power_off"] = False
                    restart_seq = [{"name": "power_on_restart", "cycles": 10, "status": "STANDBY"}]
                    state["sequence"] = restart_seq + build_phase_sequence(state["scenario"])
                    state["phase_index"] = 0
                    state["phase_cycle"] = 0
                    print(f"{ev_charger_id} | POWER_ON - 재시작 (standby 복귀 중)")

                phase = get_current_phase(state)
                phase_name = phase["name"]
                charger_status = phase["status"]

                update_charger_status(ev_charger_id, charger_status)

                next_sensor = generate_sensor_by_phase(
                    prev=state["sensor"],
                    phase_name=phase_name,
                    charger_type=charger_type,
                    phase_cycle=state["phase_cycle"],
                )

                insert_sensor_log(ev_charger_id, next_sensor)
                result = trigger_prediction(ev_charger_id)

                ai_status = result.get("ai_status") or result.get("status")
                main_reason = result.get("main_reason")
                fault_prob_7d = result.get("fault_prob_7d")

                print(
                    f"{ev_charger_id} | "
                    f"TYPE={charger_type} | "
                    f"SCENARIO={state['scenario']} | "
                    f"PHASE={phase_name} | "
                    f"STATUS={charger_status} | "
                    f"T={next_sensor['temperature']}℃ "
                    f"V={next_sensor['voltage']}V "
                    f"I={next_sensor['current']}A | "
                    f"AI={ai_status} "
                    f"사유={main_reason} "
                    f"고장확률={fault_prob_7d}"
                )

                state["sensor"] = next_sensor
                sim_state[ev_charger_id] = next_phase_state(state)

            except Exception as e:
                print(f"{ev_charger_id} 처리 실패: {e}")

        time.sleep(INSERT_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()