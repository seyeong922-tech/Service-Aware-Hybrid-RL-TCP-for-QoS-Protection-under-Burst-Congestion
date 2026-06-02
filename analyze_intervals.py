import argparse
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------------
# 기본 경로 및 분석 구간 정의
# 결과 파일을 읽을 기본 경로와 주요 scenario interval을 정의합니다.
#
# startup: TCP 연결 초기화 및 slow start 영향이 포함되는 warm-up 구간
#
# pre: burst traffic 유입 전 정상 구간
#
# burst: S3 burst traffic이 유입되는 핵심 QoS 평가 구간
#
# post: burst 종료 이후 회복 구간
#
# steady: startup을 제외한 전체 steady-state 구간
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "results" / "topo3_rtt"

SCENARIOS = {
    "startup": (0.0, 20.0, "Startup / Warm-up"),
    "pre": (20.0, 50.0, "Pre-burst Normal"),
    "burst": (50.0, 100.0, "QoS Burst"),
    "post": (100.0, 160.0, "Post-burst Recovery"),
    "steady": (20.0, 160.0, "Steady Overall"),
}

DEFAULT_SCENARIO = "burst"


# -----------------------------------------------------------------------------
# 실행 인자 정의
# 분석할 결과 디렉토리와 분석 구간을 command-line option으로 받습니다.
#
# ex)
#   python3 scratch/analyze_intervals.py --data-dir scratch/results/topo3_rtt
#   python3 scratch/analyze_intervals.py --data-dir scratch/results/parkinglot_stress_rtt --start 50 --end 70
#
# --start와 --end를 지정하면 --scenario의 기본 구간보다 우선 적용됩니다.
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Directory containing baseline_*.txt and rl_*.txt result files. "
            "Default: scratch/results/topo3_rtt"
        ),
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default=DEFAULT_SCENARIO,
        choices=list(SCENARIOS.keys()),
        help="Analysis scenario interval.",
    )

    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Custom interval start time. Overrides --scenario start.",
    )

    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="Custom interval end time. Overrides --scenario end.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# 결과 디렉토리 결정
# --data-dir가 주어지지 않으면 main parking-lot 결과 경로를 사용합니다.
# -----------------------------------------------------------------------------

def resolve_data_dir(args):
    if args.data_dir is None:
        return DEFAULT_DATA_DIR

    return Path(args.data_dir)


# -----------------------------------------------------------------------------
# 시계열 로그 파일 로드
# topology 실행 결과로 생성된 txt 파일을 읽어 time/value 배열로 반환합니다.
#
# 입력 파일 형식:
#   time value
#
# 예:
#   baseline_goodput_s1.txt
#   rl_rtt_s2.txt
#
# 분석에 필요한 파일이 없거나 형식이 잘못된 경우에는 예외를 발생시켜
# 잘못된 표가 출력되지 않도록 합니다.
# -----------------------------------------------------------------------------

def load_series(data_dir, filename):
    path = data_dir / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    data = np.genfromtxt(path, invalid_raise=False)

    if data is None:
        raise ValueError(f"Empty file: {path}")

    data = np.asarray(data)

    if data.size == 0:
        raise ValueError(f"Empty file: {path}")

    data = np.atleast_2d(data)

    if data.shape[1] < 2:
        raise ValueError(f"Invalid format: {path}, expected at least 2 columns")

    return data[:, 0], data[:, 1]


# -----------------------------------------------------------------------------
# 분석 구간 선택
# time 배열과 value 배열에서 [start, end) 구간에 해당하는 값만 추출합니다.
#
# end를 포함하지 않는 half-open interval을 사용해 인접 구간 분석 시
# 같은 sample이 중복 집계되는 것을 피합니다.
# -----------------------------------------------------------------------------

def select_interval(t, y, start, end):
    t = np.asarray(t)
    y = np.asarray(y)

    mask = (t >= start) & (t < end)
    return y[mask]


# -----------------------------------------------------------------------------
# 안전한 통계 계산 함수
# 선택된 구간에 sample이 없는 경우 NaN을 반환합니다.
# 이렇게 하면 빈 구간으로 인해 분석 스크립트가 중단되는 것을 방지할 수 있습니다.
# -----------------------------------------------------------------------------

def mean_or_nan(values):
    values = np.asarray(values)

    if values.size == 0:
        return np.nan

    return float(np.mean(values))


def median_or_nan(values):
    values = np.asarray(values)

    if values.size == 0:
        return np.nan

    return float(np.median(values))


def percentile_or_nan(values, percentile):
    values = np.asarray(values)

    if values.size == 0:
        return np.nan

    return float(np.percentile(values, percentile))


# -----------------------------------------------------------------------------
# QoS ratio 계산 함수
# ratio_ge는 특정 threshold 이상인 sample 비율을 계산합니다.
# ratio_le는 특정 threshold 이하인 sample 비율을 계산합니다.
#
# 현재 사용 기준:
#   S1 survival: S1 goodput >= 1Mbps
#   S2 QoS:      S2 goodput >= 5Mbps
#   S2 RTT OK:   S2 RTT <= 120ms
# -----------------------------------------------------------------------------

def ratio_ge(values, threshold):
    values = np.asarray(values)

    if values.size == 0:
        return np.nan

    return float(np.mean(values >= threshold) * 100.0)


def ratio_le(values, threshold):
    values = np.asarray(values)

    if values.size == 0:
        return np.nan

    return float(np.mean(values <= threshold) * 100.0)


# -----------------------------------------------------------------------------
# Jain fairness index 계산
# S1/S2 평균 throughput을 기준으로 fairness index를 계산합니다.
#
# 주의:
# 이 값은 flow 간 균등성을 보는 보조 지표입니다.
# 본 프로젝트의 핵심 목표는 모든 flow의 균등 처리량 극대화가 아니라,
# S2 primary video-like flow의 QoS compliance 우선 보호입니다.
# -----------------------------------------------------------------------------

def jain_fairness(values):
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return np.nan

    denom = values.size * np.sum(values ** 2)

    if denom == 0:
        return np.nan

    return float((np.sum(values) ** 2) / denom)


# -----------------------------------------------------------------------------
# Prefix별 metric 수집
# baseline 또는 rl prefix에 해당하는 S1/S2 goodput, RTT 파일을 읽고,
# 지정된 interval에서 주요 성능 지표를 계산합니다.
#
# 주요 출력 metric:
# - S1/S2 mean throughput
# - S1/S2 mean RTT
# - Jain fairness index
# - S1/S2 median throughput
# - S1/S2 p95 RTT
# - S1 survival ratio
# - S2 5Mbps QoS compliance ratio
# - S2 120ms RTT compliance ratio
# -----------------------------------------------------------------------------

def collect_metrics(data_dir, prefix, start, end):
    s1_gp_t, s1_gp = load_series(data_dir, f"{prefix}_goodput_s1.txt")
    s2_gp_t, s2_gp = load_series(data_dir, f"{prefix}_goodput_s2.txt")
    s1_rtt_t, s1_rtt = load_series(data_dir, f"{prefix}_rtt_s1.txt")
    s2_rtt_t, s2_rtt = load_series(data_dir, f"{prefix}_rtt_s2.txt")

    s1_gp_v = select_interval(s1_gp_t, s1_gp, start, end)
    s2_gp_v = select_interval(s2_gp_t, s2_gp, start, end)
    s1_rtt_v = select_interval(s1_rtt_t, s1_rtt, start, end)
    s2_rtt_v = select_interval(s2_rtt_t, s2_rtt, start, end)

    s1_gp_mean = mean_or_nan(s1_gp_v)
    s2_gp_mean = mean_or_nan(s2_gp_v)

    metrics = {
        "s1_tp": s1_gp_mean,
        "s2_tp": s2_gp_mean,
        "s1_delay": mean_or_nan(s1_rtt_v),
        "s2_delay": mean_or_nan(s2_rtt_v),
        "jain": jain_fairness(np.array([s1_gp_mean, s2_gp_mean])),
        "s1_tp_median": median_or_nan(s1_gp_v),
        "s2_tp_median": median_or_nan(s2_gp_v),
        "s1_delay_p95": percentile_or_nan(s1_rtt_v, 95),
        "s2_delay_p95": percentile_or_nan(s2_rtt_v, 95),
        "s1_survival": ratio_ge(s1_gp_v, 1.0),
        "s2_qos": ratio_ge(s2_gp_v, 5.0),
        "s2_delay_ok": ratio_le(s2_rtt_v, 120.0),
        "sample_count_gp": min(len(s1_gp_v), len(s2_gp_v)),
    }

    return metrics


# -----------------------------------------------------------------------------
# 숫자 출력 형식 처리
# NaN 값은 그대로 "nan"으로 표시하고, 일반 숫자는 지정된 자리수로 포맷합니다.
# -----------------------------------------------------------------------------

def fmt(value, digits=2):
    if value is None or np.isnan(value):
        return "nan"

    return f"{value:.{digits}f}"


# -----------------------------------------------------------------------------
# 주요 성능 비교표 출력
# 지정된 scenario interval에서 S1/S2 throughput, S1/S2 delay, Jain fairness를
# Baseline Cubic과 Proposed RL로 나누어 출력합니다.
#
# Delay는 one-way delay가 아니라 TCP RTT입니다.
# -----------------------------------------------------------------------------

def print_main_table(scenario_label, start, end, data_dir, base, rl):
    title = "Primary Video QoS Scenario Analysis"

    print()
    print(f"{title}...")
    print()
    print(f"[DATA_DIR] {data_dir}")
    print("=" * 86)
    print(
        f"Scenario: {scenario_label} "
        f"({start:.0f}s ~ {end:.0f}s)"
    )
    print("=" * 86)
    print(f"{'Metric (QoS Scenario)':<30} | {'TCP Cubic (Base)':>18} | {'Proposed RL':>18}")
    print("-" * 86)

    rows = [
        ("S1 (FTP) TP (Mbps)", "s1_tp", 2),
        ("S2 (Video) TP (Mbps)", "s2_tp", 2),
        ("S1 (FTP) Delay (ms)", "s1_delay", 2),
        ("S2 (Video) Delay (ms)", "s2_delay", 2),
        ("Jain Fairness Index", "jain", 4),
    ]

    for label, key, digits in rows:
        print(
            f"{label:<30} | "
            f"{fmt(base[key], digits):>18} | "
            f"{fmt(rl[key], digits):>18}"
        )

    print("=" * 86)
    print()


# -----------------------------------------------------------------------------
# QoS 및 안정성 세부표 출력
# S2 primary video-like flow의 5Mbps compliance, 120ms RTT constraint,
# S1 FTP survival ratio, median throughput, p95 delay를 출력합니다.
#
# 이 표는 "S2를 얼마나 보호했는가"와 "S1이 완전히 starvation되지 않았는가"를
# 함께 확인하기 위한 보조 분석입니다.
# -----------------------------------------------------------------------------

def print_qos_table(base, rl):
    print("[QoS / Stability Detail]")
    print("-" * 86)
    print(f"{'Metric':<30} | {'TCP Cubic (Base)':>18} | {'Proposed RL':>18}")
    print("-" * 86)

    rows = [
        ("S2 >= 5Mbps Ratio (%)", "s2_qos", 1),
        ("S2 RTT <= 120ms (%)", "s2_delay_ok", 1),
        ("S1 >= 1Mbps Ratio (%)", "s1_survival", 1),
        ("S1 TP Median (Mbps)", "s1_tp_median", 2),
        ("S2 TP Median (Mbps)", "s2_tp_median", 2),
        ("S1 Delay p95 (ms)", "s1_delay_p95", 2),
        ("S2 Delay p95 (ms)", "s2_delay_p95", 2),
    ]

    for label, key, digits in rows:
        print(
            f"{label:<30} | "
            f"{fmt(base[key], digits):>18} | "
            f"{fmt(rl[key], digits):>18}"
        )

    print("-" * 86)

    s1_delta = rl["s1_tp"] - base["s1_tp"]
    s2_delta = rl["s2_tp"] - base["s2_tp"]
    s2_qos_delta = rl["s2_qos"] - base["s2_qos"]
    s2_delay_delta = rl["s2_delay"] - base["s2_delay"]

    print("[Key Difference]")
    print(f"  S1 FTP TP change       : {s1_delta:+.2f} Mbps")
    print(f"  S2 Video TP change     : {s2_delta:+.2f} Mbps")
    print(f"  S2 >= 5Mbps change     : {s2_qos_delta:+.1f}%p")
    print(f"  S2 Delay mean change   : {s2_delay_delta:+.2f} ms")
    print("=" * 86)
    print("Note: Delay is measured as TCP RTT, not one-way delay.")
    print("=" * 86)


# -----------------------------------------------------------------------------
# Main 분석 흐름
# command-line 인자를 해석해 분석 구간을 결정하고, baseline/RL 결과를 읽어
# 주요 성능 비교표와 QoS 세부표를 출력합니다.
#
# 기본적으로는 50~100초 burst 구간을 분석합니다.
# parking-lot stress처럼 구간을 나누어 보고 싶을 때는 --start, --end를 사용합니다.
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    data_dir = resolve_data_dir(args)

    scenario_start, scenario_end, scenario_label = SCENARIOS[args.scenario]

    if args.start is not None:
        scenario_start = args.start

    if args.end is not None:
        scenario_end = args.end

    if scenario_start >= scenario_end:
        raise ValueError(
            f"Invalid interval: start={scenario_start}, end={scenario_end}"
        )

    base = collect_metrics(data_dir, "baseline", scenario_start, scenario_end)
    rl = collect_metrics(data_dir, "rl", scenario_start, scenario_end)

    print_main_table(
        scenario_label,
        scenario_start,
        scenario_end,
        data_dir,
        base,
        rl,
    )

    print_qos_table(base, rl)


if __name__ == "__main__":
    main()