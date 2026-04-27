# ttr.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import math


# =========================================================
# 설정값
# =========================================================

@dataclass
class TTRConfig:
    """
    예지보전 판단 기준 설정
    - 최대 예측 범위는 14일(336시간) 유지
    - 화면에 보여줄 고장확률은 7일 내 고장확률만 사용
    """
    # TTR 기준 (시간)
    risk_ttr_hours: float = 48.0         # 2일 이내 위험 도달 -> 위험
    inspect_ttr_hours: float = 168.0     # 7일 이내 위험 도달 -> 점검

    # 상태 확률 기준
    risk_prob_threshold: float = 0.70
    inspect_prob_threshold: float = 0.55

    # 센서 기준
    temp_warning_start: float = 60.0
    temp_risk_start: float = 85.0

    health_warning_below: float = 60.0
    health_risk_below: float = 35.0

    usage_warning_hours: float = 14.0
    usage_risk_hours: float = 17.0

    # 최대 TTR: 14일 = 336시간
    max_ttr_hours: float = 336.0

    # 화면 표기용 고정 horizon
    display_fault_horizon_hours: float = 168.0   # 7일


# =========================================================
# 입력/출력 데이터 구조
# =========================================================

@dataclass
class InputFeatures:
    health: float
    peak_temp: float
    usage_hours: float
    voltage: float
    current: float
    is_already_fault: bool = False


@dataclass
class ProbabilityResult:
    normal: float
    inspect: float
    risk: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "normal": round(self.normal, 4),
            "inspect": round(self.inspect, 4),
            "risk": round(self.risk, 4),
        }


@dataclass
class DiagnosisResult:
    state: str
    action: Optional[str]
    risk_score: float
    reason: str
    ttr_hours: Optional[float] = None
    fault_prob_7d: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict:
        result = {
            "state": self.state,
            "action": self.action,
            "risk_score": round(self.risk_score, 4),
            "reason": self.reason,
            "fault_prob_7d": round(self.fault_prob_7d, 4) if self.fault_prob_7d is not None else None,
            "probabilities": self.probabilities,
        }

        if self.state == "점검":
            result["ttr_hours"] = round(self.ttr_hours, 1) if self.ttr_hours is not None else None

        return result


# =========================================================
# 유틸
# =========================================================

def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


# =========================================================
# 개별 위험도 계산
# =========================================================

def calc_health_risk(health: float) -> float:
    """
    health 낮을수록 위험
    100 -> 0.0, 0 -> 1.0
    """
    health = clamp(health, 0.0, 100.0)
    return clamp((100.0 - health) / 100.0, 0.0, 1.0)


def calc_temp_risk(peak_temp: float, config: TTRConfig) -> float:
    """
    경고 시작 온도부터 위험 시작 온도까지 선형 증가
    """
    if peak_temp <= config.temp_warning_start:
        return 0.0
    if peak_temp >= config.temp_risk_start:
        return 1.0

    span = config.temp_risk_start - config.temp_warning_start
    return clamp((peak_temp - config.temp_warning_start) / span, 0.0, 1.0)


def calc_usage_risk(usage_hours: float, config: TTRConfig) -> float:
    """
    사용시간이 많을수록 위험
    """
    if usage_hours <= config.usage_warning_hours:
        return 0.0
    if usage_hours >= config.usage_risk_hours:
        return 1.0

    span = config.usage_risk_hours - config.usage_warning_hours
    return clamp((usage_hours - config.usage_warning_hours) / span, 0.0, 1.0)

def calc_voltage_risk(voltage: float) -> float:
    """
    정상권 218~222V 기준
    벗어날수록 위험 증가
    """
    if 218.0 <= voltage <= 222.0:
        return 0.0

    if voltage < 218.0:
        return clamp((218.0 - voltage) / 10.0, 0.0, 1.0)

    return clamp((voltage - 222.0) / 10.0, 0.0, 1.0)


def calc_current_risk(current: float) -> float:
    """
    충전 중 기준 대략 28~33A 정상권
    대기 중은 낮아도 됨
    여기서는 단순히 과전류 위주로 위험 계산
    """
    if current <= 33.0:
        return 0.0

    return clamp((current - 33.0) / 10.0, 0.0, 1.0)
# =========================================================
# 종합 위험점수
# =========================================================

def calculate_risk_score(features: InputFeatures, config: TTRConfig) -> float:
    health_risk = calc_health_risk(features.health)
    temp_risk = calc_temp_risk(features.peak_temp, config)
    usage_risk = calc_usage_risk(features.usage_hours, config)
    voltage_risk = calc_voltage_risk(features.voltage)
    current_risk = calc_current_risk(features.current)

    risk_score = (
        health_risk * 0.25 +
        temp_risk * 0.30 +
        usage_risk * 0.15 +
        voltage_risk * 0.15 +
        current_risk * 0.15
    )

    return clamp(risk_score, 0.0, 1.0)


# =========================================================
# 상태 확률 계산
# =========================================================

def calculate_probabilities(risk_score: float) -> ProbabilityResult:
    """
    risk_score를 정상/점검/위험 확률로 변환
    """
    risk_score = clamp(risk_score, 0.0, 1.0)

    normal_raw = max(0.0, 1.0 - risk_score) ** 2.2
    risk_raw = max(0.0, risk_score) ** 2.0
    inspect_raw = math.exp(-((risk_score - 0.55) ** 2) / 0.02) * 1.15

    total = normal_raw + inspect_raw + risk_raw
    if total == 0:
        return ProbabilityResult(normal=1.0, inspect=0.0, risk=0.0)

    return ProbabilityResult(
        normal=normal_raw / total,
        inspect=inspect_raw / total,
        risk=risk_raw / total,
    )


# =========================================================
# TTR 계산
# =========================================================

def calculate_ttr_hours(risk_score: float, config: TTRConfig) -> float:
    """
    위험점수가 높을수록 TTR은 짧아짐
    최대 336시간(14일)
    """
    risk_score = clamp(risk_score, 0.0, 1.0)
    ttr = (1.0 - risk_score) * config.max_ttr_hours
    return clamp(ttr, 0.0, config.max_ttr_hours)


# =========================================================
# 7일 내 고장확률 계산
# =========================================================

def calculate_fault_prob_7d(risk_score: float, state: str) -> float:
    """
    환경부/운영사 기준 구간으로 산출 (프론트에서 ×100으로 % 표기)
    정상: 미표기(0.0), 점검: 1%~5%, 위험: 5%~15%
    """
    risk_score = clamp(risk_score, 0.0, 1.0)

    if state == "정상":
        return 0.0

    if state == "점검":
        return clamp(0.05 + risk_score * 0.05, 0.05, 0.10)

    return clamp(0.10 + risk_score * 0.10, 0.10, 0.20)

# =========================================================
# 판단 사유 생성
# =========================================================

def build_reason(features: InputFeatures, config: TTRConfig) -> str:
    reasons = []

    if features.is_already_fault:
        reasons.append("이미 고장 또는 현장조치 필요 상태")

    if features.health <= config.health_risk_below:
        reasons.append("기기 수명 급락")
    elif features.health <= config.health_warning_below:
        reasons.append("기기 수명 감소")

    if features.peak_temp >= config.temp_risk_start:
        reasons.append("온도 과열")
    elif features.peak_temp >= config.temp_warning_start:
        reasons.append("내부 온도 상승")

    if features.usage_hours >= config.usage_risk_hours:
        reasons.append("과도한 사용시간")
    elif features.usage_hours >= config.usage_warning_hours:
        reasons.append("사용량 증가")

    if not reasons:
        return "특이 이상 없음"

    return ", ".join(reasons)


# =========================================================
# 상태 판정
# =========================================================

def classify_state(
    features: InputFeatures,
    probs: ProbabilityResult,
    ttr_hours: float,
    config: TTRConfig,
) -> str:
    """
    최종 상태:
    - 이미 고장 -> 위험
    - 위험 -> 확률 노출 안 함
    - 점검 -> 7일 내 고장확률 노출
    - 정상 -> 확률 노출 안 함
    """
    if features.is_already_fault:
        return "위험"

    if (
        ttr_hours <= config.risk_ttr_hours
        or probs.risk >= config.risk_prob_threshold
        or features.peak_temp >= config.temp_risk_start
        or features.health <= config.health_risk_below
        or features.usage_hours >= config.usage_risk_hours
    ):
        return "위험"

    if (
        ttr_hours <= config.inspect_ttr_hours
        or probs.inspect >= config.inspect_prob_threshold
        or features.peak_temp >= config.temp_warning_start
        or features.health <= config.health_warning_below
        or features.usage_hours >= config.usage_warning_hours
    ):
        return "점검"

    return "정상"


def build_action(state: str) -> Optional[str]:
    if state == "위험":
        return "즉시 점검 및 작동 중지"
    if state == "점검":
        return "점검 필요"
    return None


# =========================================================
# 최종 진단
# =========================================================

def diagnose(features: InputFeatures, config: Optional[TTRConfig] = None) -> DiagnosisResult:
    config = config or TTRConfig()
    print("USING TTR FILE:", __file__, flush=True)
    print("DIAG INPUT:", features, flush=True)

    if features.is_already_fault:
        print("DIAG RESULT:", "위험", 1.0, 1.0, flush=True)
        return DiagnosisResult(
            state="위험",
            action="즉시 점검 및 작동 중지",
            risk_score=1.0,
            reason=build_reason(features, config),
            ttr_hours=None,
            fault_prob_7d=1.0,
            probabilities=None,
        )

    risk_score = calculate_risk_score(features, config)
    probs = calculate_probabilities(risk_score)
    ttr_hours = calculate_ttr_hours(risk_score, config)
    reason = build_reason(features, config)

    state = classify_state(features, probs, ttr_hours, config)
    fault_prob_7d = calculate_fault_prob_7d(risk_score, state)
    action = build_action(state)

    print("DIAG RESULT:", state, risk_score, fault_prob_7d, flush=True)

    if state == "점검":
        return DiagnosisResult(
            state=state,
            action=action,
            risk_score=risk_score,
            reason=reason,
            ttr_hours=ttr_hours,
            fault_prob_7d=fault_prob_7d,
            probabilities=probs.to_dict(),
        )

    return DiagnosisResult(
        state=state,
        action=action,
        risk_score=risk_score,
        reason=reason,
        ttr_hours=None,
        fault_prob_7d=fault_prob_7d,
        probabilities=probs.to_dict(),
    )

# =========================================================
# 배치 처리용
# =========================================================

def diagnose_to_dict(features: InputFeatures, config: Optional[TTRConfig] = None) -> Dict:
    return diagnose(features, config).to_dict()


# =========================================================
# 테스트
# =========================================================

if __name__ == "__main__":
    samples = [
        InputFeatures(
            health=90.0,
            peak_temp=52.0,
            usage_hours=6.0,
            is_already_fault=False,
        ),
        InputFeatures(
            health=58.0,
            peak_temp=67.0,
            usage_hours=9.2,
            is_already_fault=False,
        ),
        InputFeatures(
            health=29.0,
            peak_temp=94.0,
            usage_hours=13.0,
            is_already_fault=False,
        ),
        InputFeatures(
            health=41.0,
            peak_temp=72.0,
            usage_hours=10.0,
            is_already_fault=True,
        ),
    ]

    for i, sample in enumerate(samples, start=1):
        result = diagnose(sample)
        print(f"[SAMPLE {i}]")
        print(result.to_dict())
        print("-" * 60)