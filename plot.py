import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 기본 경로 및 QoS 기준 정의
# 결과 파일을 읽을 기본 경로와 분석에 사용할 QoS threshold를 정의합니다.
# MIN_GP: S2 primary video-like flow의 최소 goodput 기준
# TARGET_GP: S2가 안정적이라고 볼 수 있는 목표 goodput 기준
# RTT_LIMIT: S2 delay constraint로 사용하는 TCP RTT 기준
# STEADY_START: TCP slow start 및 초기 연결 구간을 제외하기 위한 분석 시작 시점
# BURST_START / BURST_END: S3 burst traffic이 유입되는 기본 구간
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_DATA_DIR = SCRIPT_DIR / "results" / "topo3_rtt"

MIN_GP = 5.0
TARGET_GP = 7.0
RTT_LIMIT = 120.0

STEADY_START = 20.0
BURST_START = 50.0
BURST_END = 100.0


# -----------------------------------------------------------------------------
# 실행 인자 정의
# 분석할 결과 디렉토리, 저장할 figure 이름, figure title을 command-line option으로 받습니다.
# 예시로 python3 scratch/plot.py --data-dir scratch/results/topo3_rtt
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
        "--save-name",
        type=str,
        default=None,
        help=(
            "Output image path. "
            "Default: <data-dir>/topo3_stable_report.png"
        ),
    )

    parser.add_argument(
        "--title",
        type=str,
        default="Primary Video QoS Performance",
        help="Figure title prefix.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# 입력/출력 경로 결정
# --data-dir가 주어지지 않으면 main parking-lot 결과 경로를 사용합니다.
# --save-name이 주어지지 않으면 해당 data-dir 아래에 topo3_stable_report.png로 저장합니다.
# -----------------------------------------------------------------------------

def resolve_paths(args):
    if args.data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    else:
        data_dir = Path(args.data_dir)

    if args.save_name is None:
        save_name = data_dir / "topo3_stable_report.png"
    else:
        save_name = Path(args.save_name)

    return data_dir, save_name


# -----------------------------------------------------------------------------
# 시계열 로그 파일 로드
# topology 실행 결과로 생성된 txt 파일을 읽고, steady_start 이후 데이터만 반환합니다.
# 예시: baseline_goodput_s2.txt, rl_rtt_s2.txt 등
# 파일이 없거나 형식이 잘못된 경우 None을 반환하여 이후 require_data()에서 중단
# -----------------------------------------------------------------------------

def load_series(data_dir, filename, steady_start=STEADY_START):
    path = data_dir / filename

    if not path.exists():
        print(f"[Missing] {path}")
        return None, None

    try:
        data = np.genfromtxt(path, invalid_raise=False)
    except Exception as e:
        print(f"[Load Error] {path}: {e}")
        return None, None

    if data is None:
        print(f"[Empty] {path}")
        return None, None

    data = np.asarray(data)

    if data.size == 0:
        print(f"[Empty] {path}")
        return None, None

    data = np.atleast_2d(data)

    if data.shape[1] < 2:
        print(f"[Invalid Format] {path}: expected at least 2 columns")
        return None, None

    steady = data[data[:, 0] > steady_start]

    if steady.shape[0] == 0:
        print(f"[No Steady Data] {path}: no samples after {steady_start}s")
        return None, None

    return steady[:, 0], steady[:, 1]


# -----------------------------------------------------------------------------
# 필수 데이터 확인
# S2 goodput/RTT의 baseline과 RL 결과는 figure 및 summary 계산에 필수이므로, 하나라도 없으면 분석을 중단합니다.
# -----------------------------------------------------------------------------

def require_data(named_series):
    missing = []

    for name, series in named_series.items():
        t, y = series
        if t is None or y is None:
            missing.append(name)

    if missing:
        print("[Abort] Required data is missing:")
        for name in missing:
            print(f"  - {name}")
        return False

    return True


# -----------------------------------------------------------------------------
# 구간 필터링
# 특정 시간 구간에 해당하는 값만 추출 - burst 구간 성능을 계산할 때 사용합니다.
# -----------------------------------------------------------------------------

def filter_interval(t, y, start, end):
    t = np.asarray(t)
    y = np.asarray(y)

    mask = (t >= start) & (t <= end)

    return y[mask]


# -----------------------------------------------------------------------------
# 안전한 percentage 계산
# 조건 배열이 비어 있을 경우 NaN을 반환하여 빈 구간 계산 오류를 방지합니다.
# -----------------------------------------------------------------------------

def safe_percent(condition):
    condition = np.asarray(condition)

    if condition.size == 0:
        return np.nan

    return float(np.mean(condition) * 100.0)


# -----------------------------------------------------------------------------
# 기본 통계량 계산
# mean, median, p95를 반환합니다. 값이 비어 있으면 NaN을 반환합니다.
# -----------------------------------------------------------------------------

def summarize(values):
    values = np.asarray(values)

    if values.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "p95": np.nan,
        }

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


# -----------------------------------------------------------------------------
# S2 QoS metric 계산
# steady 구간 전체와 burst 구간에 대해 goodput/RTT 통계와 QoS compliance를 계산합니다.
# 주요 지표:
# - Overall S2 goodput >= 5Mbps ratio
# - Burst S2 goodput >= 5Mbps ratio
# - Overall S2 RTT <= 120ms ratio
# - Burst S2 RTT <= 120ms ratio
# -----------------------------------------------------------------------------

def compute_metrics(gp_t, gp, rtt_t, rtt):
    burst_gp = filter_interval(gp_t, gp, BURST_START, BURST_END)
    burst_rtt = filter_interval(rtt_t, rtt, BURST_START, BURST_END)

    gp_summary = summarize(gp)
    rtt_summary = summarize(rtt)
    burst_gp_summary = summarize(burst_gp)
    burst_rtt_summary = summarize(burst_rtt)

    return {
        "gp_mean": gp_summary["mean"],
        "gp_median": gp_summary["median"],
        "gp_p95": gp_summary["p95"],
        "rtt_mean": rtt_summary["mean"],
        "rtt_median": rtt_summary["median"],
        "rtt_p95": rtt_summary["p95"],
        "gp_qos_ratio": safe_percent(gp >= MIN_GP),
        "rtt_qos_ratio": safe_percent(rtt <= RTT_LIMIT),
        "burst_gp_mean": burst_gp_summary["mean"],
        "burst_gp_median": burst_gp_summary["median"],
        "burst_gp_p95": burst_gp_summary["p95"],
        "burst_gp_qos_ratio": safe_percent(burst_gp >= MIN_GP),
        "burst_rtt_mean": burst_rtt_summary["mean"],
        "burst_rtt_median": burst_rtt_summary["median"],
        "burst_rtt_p95": burst_rtt_summary["p95"],
        "burst_rtt_qos_ratio": safe_percent(burst_rtt <= RTT_LIMIT),
    }


# -----------------------------------------------------------------------------
# S2 성능 요약 출력
# Baseline Cubic 또는 Proposed RL의 S2 goodput/RTT/QoS compliance를 터미널에 출력합니다.
# -----------------------------------------------------------------------------

def print_flow_summary(label, metrics):
    print(f"[{label}] QoS Compliance")
    print(
        f"  S2 Goodput mean/median/p95: "
        f"{metrics['gp_mean']:.2f}/"
        f"{metrics['gp_median']:.2f}/"
        f"{metrics['gp_p95']:.2f} Mbps"
    )
    print(
        f"  S2 RTT mean/median/p95: "
        f"{metrics['rtt_mean']:.2f}/"
        f"{metrics['rtt_median']:.2f}/"
        f"{metrics['rtt_p95']:.2f} ms"
    )
    print(
        f"  Overall S2 Goodput >= {MIN_GP:.1f} Mbps ratio: "
        f"{metrics['gp_qos_ratio']:.1f}%"
    )
    print(
        f"  Overall S2 RTT <= {RTT_LIMIT:.0f} ms ratio: "
        f"{metrics['rtt_qos_ratio']:.1f}%"
    )
    print(
        f"  Burst S2 Goodput mean/median: "
        f"{metrics['burst_gp_mean']:.2f}/"
        f"{metrics['burst_gp_median']:.2f} Mbps"
    )
    print(
        f"  Burst S2 Goodput >= {MIN_GP:.1f} Mbps ratio: "
        f"{metrics['burst_gp_qos_ratio']:.1f}%"
    )
    print(
        f"  Burst S2 RTT mean/median/p95: "
        f"{metrics['burst_rtt_mean']:.2f}/"
        f"{metrics['burst_rtt_median']:.2f}/"
        f"{metrics['burst_rtt_p95']:.2f} ms"
    )
    print(
        f"  Burst S2 RTT <= {RTT_LIMIT:.0f} ms ratio: "
        f"{metrics['burst_rtt_qos_ratio']:.1f}%"
    )


# -----------------------------------------------------------------------------
# S2 goodput time-series plot
# Baseline Cubic과 Proposed RL의 S2 goodput 변화를 시간에 따라 비교합니다.
# 5Mbps minimum line, 7Mbps target line, burst period를 함께 표시합니다.
# -----------------------------------------------------------------------------

def plot_goodput_timeseries(ax, b_gp_t, b_gp, r_gp_t, r_gp):
    ax.plot(
        b_gp_t,
        b_gp,
        label="Baseline Cubic",
        alpha=0.45,
        linewidth=1.4,
    )

    ax.plot(
        r_gp_t,
        r_gp,
        label="Proposed RL",
        linewidth=1.4,
    )

    ax.axhline(
        y=MIN_GP,
        linestyle="--",
        linewidth=1.3,
        label=f"Minimum {MIN_GP:.0f} Mbps",
    )

    ax.axhline(
        y=TARGET_GP,
        linestyle=":",
        linewidth=1.5,
        label=f"Target {TARGET_GP:.0f} Mbps",
    )

    ax.axvspan(
        BURST_START,
        BURST_END,
        alpha=0.12,
        label="Burst Period",
    )

    ax.set_title("S2 Primary Video-like Goodput over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Goodput (Mbps)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


# -----------------------------------------------------------------------------
# S2 RTT CDF plot
# Baseline Cubic과 Proposed RL의 S2 RTT 분포를 CDF로 비교합니다.
# 120ms RTT limit을 함께 표시해 delay constraint 만족 여부를 확인합니다.
# -----------------------------------------------------------------------------

def plot_rtt_cdf(ax, b_rtt, r_rtt):
    for values, label in [
        (b_rtt, "Baseline Cubic"),
        (r_rtt, "Proposed RL"),
    ]:
        values = np.asarray(values)

        if values.size < 2:
            continue

        sorted_values = np.sort(values)
        yvals = np.arange(sorted_values.size) / float(sorted_values.size - 1)

        ax.plot(
            sorted_values,
            yvals,
            label=label,
            linewidth=2.0,
        )

    ax.axvline(
        x=RTT_LIMIT,
        linestyle="--",
        linewidth=1.3,
        label=f"{RTT_LIMIT:.0f} ms Limit",
    )

    ax.set_title("S2 RTT CDF")
    ax.set_xlabel("RTT (ms)")
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


# -----------------------------------------------------------------------------
# S2 QoS compliance bar plot
# Overall 및 burst 구간에서 S2 goodput이 5Mbps 이상인 비율을 bar chart로 비교합니다.
# primary video-like flow 보호 효과 요약을 위한 다이어그램**
# -----------------------------------------------------------------------------

def plot_qos_compliance_bar(ax, baseline_metrics, rl_metrics):
    labels = [
        "Overall\nGoodput ≥ 5 Mbps",
        "Burst\nGoodput ≥ 5 Mbps",
    ]

    baseline_values = [
        baseline_metrics["gp_qos_ratio"],
        baseline_metrics["burst_gp_qos_ratio"],
    ]

    rl_values = [
        rl_metrics["gp_qos_ratio"],
        rl_metrics["burst_gp_qos_ratio"],
    ]

    x = np.arange(len(labels))
    width = 0.34

    bars_baseline = ax.bar(
        x - width / 2,
        baseline_values,
        width,
        label="Baseline Cubic",
        alpha=0.65,
    )

    bars_rl = ax.bar(
        x + width / 2,
        rl_values,
        width,
        label="Proposed RL",
        alpha=0.85,
    )

    ax.set_title("S2 QoS Compliance")
    ax.set_ylabel("Compliance Ratio (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")

    for bars in [bars_baseline, bars_rl]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )


# -----------------------------------------------------------------------------
# Main 분석 흐름
# 결과 파일을 로드하고, S2 metric을 계산한 뒤, figure와 터미널 summary를 생성합니다.
# 출력 figure:
# - S2 goodput time-series
# - S2 RTT CDF
# - S2 QoS compliance bar
# 터미널 summary:
# - Baseline Cubic S2 QoS
# - Proposed RL S2 QoS
# - S1 FTP impact
# - Key improvements
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    data_dir, save_name = resolve_paths(args)

    print("Analyzing primary video QoS performance...")
    print(f"[DATA_DIR] {data_dir}")
    print(f"[SAVE_NAME] {save_name}")

    os.makedirs(data_dir, exist_ok=True)

    b_gp1_t, b_gp1 = load_series(data_dir, "baseline_goodput_s1.txt")
    b_gp2_t, b_gp2 = load_series(data_dir, "baseline_goodput_s2.txt")
    b_rtt2_t, b_rtt2 = load_series(data_dir, "baseline_rtt_s2.txt")

    r_gp1_t, r_gp1 = load_series(data_dir, "rl_goodput_s1.txt")
    r_gp2_t, r_gp2 = load_series(data_dir, "rl_goodput_s2.txt")
    r_rtt2_t, r_rtt2 = load_series(data_dir, "rl_rtt_s2.txt")

    needed = {
        "baseline_goodput_s2": (b_gp2_t, b_gp2),
        "baseline_rtt_s2": (b_rtt2_t, b_rtt2),
        "rl_goodput_s2": (r_gp2_t, r_gp2),
        "rl_rtt_s2": (r_rtt2_t, r_rtt2),
    }

    if not require_data(needed):
        return

    baseline_metrics = compute_metrics(b_gp2_t, b_gp2, b_rtt2_t, b_rtt2)
    rl_metrics = compute_metrics(r_gp2_t, r_gp2, r_rtt2_t, r_rtt2)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(args.title, fontsize=16, fontweight="bold", y=0.995)

    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])

    ax_goodput = fig.add_subplot(gs[0, :])
    ax_rtt_cdf = fig.add_subplot(gs[1, 0])
    ax_qos = fig.add_subplot(gs[1, 1])

    plot_goodput_timeseries(ax_goodput, b_gp2_t, b_gp2, r_gp2_t, r_gp2)
    plot_rtt_cdf(ax_rtt_cdf, b_rtt2, r_rtt2)
    plot_qos_compliance_bar(ax_qos, baseline_metrics, rl_metrics)

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)

    print("=" * 60)
    print("[Summary after steady start]")
    print_flow_summary("Baseline Cubic", baseline_metrics)
    print("-" * 60)
    print_flow_summary("Proposed RL", rl_metrics)

    if b_gp1 is not None and r_gp1 is not None:
        b_s1 = summarize(b_gp1)
        r_s1 = summarize(r_gp1)

        s1_delta = r_s1["mean"] - b_s1["mean"]
        s1_delta_ratio = (
            (s1_delta / b_s1["mean"]) * 100.0
            if b_s1["mean"] != 0
            else np.nan
        )

        print("-" * 60)
        print("[S1 FTP Impact]")
        print(
            f"  Baseline S1 Goodput mean/median/p95: "
            f"{b_s1['mean']:.2f}/"
            f"{b_s1['median']:.2f}/"
            f"{b_s1['p95']:.2f} Mbps"
        )
        print(
            f"  Proposed RL S1 Goodput mean/median/p95: "
            f"{r_s1['mean']:.2f}/"
            f"{r_s1['median']:.2f}/"
            f"{r_s1['p95']:.2f} Mbps"
        )
        print(
            f"  S1 mean goodput change: "
            f"{s1_delta:+.2f} Mbps ({s1_delta_ratio:+.1f}%)"
        )

    print("-" * 60)
    print("[Key Improvements]")
    print(
        f"  Overall S2 Goodput compliance improvement: "
        f"{rl_metrics['gp_qos_ratio'] - baseline_metrics['gp_qos_ratio']:+.1f}%p"
    )
    print(
        f"  Burst S2 Goodput compliance improvement: "
        f"{rl_metrics['burst_gp_qos_ratio'] - baseline_metrics['burst_gp_qos_ratio']:+.1f}%p"
    )
    print(
        f"  S2 mean RTT change: "
        f"{rl_metrics['rtt_mean'] - baseline_metrics['rtt_mean']:+.2f} ms"
    )

    print("=" * 60)
    print(f"Report saved to: {save_name}")

    plt.show()


if __name__ == "__main__":
    main()
