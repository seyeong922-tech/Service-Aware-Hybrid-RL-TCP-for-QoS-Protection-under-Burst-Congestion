#!/usr/bin/env python3

"""
analyze_parkinglot_validation.py

parking-lot validation 결과를 분석합니다.

작성되는 report는
    multiseed:
        Baseline Cubic vs Full RL mean ± std

    ablation:
        Baseline Cubic vs Full RL vs No-Projection

    all:
        multiseed와 ablation을 모두 출력

입력 구조:
    scratch/results/parkinglot_validation/
    ├── baseline/
    ├── full_rl/seed_1 ...
    └── no_projection/seed_1 ...
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 기본 경로 및 QoS 기준 정의
# parking-lot scenario 기준으로 분석합니다.
# -----------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path("scratch/results/parkinglot_validation")

STEADY_START = 20.0
BURST_START = 50.0
BURST_END = 100.0

MIN_GP = 5.0
TARGET_GP = 7.0
RTT_LIMIT = 120.0
MIN_FTP_GP = 1.0


# -----------------------------------------------------------------------------
# 실행 인자 정의
# 분석할 결과 디렉토리, report 종류, seed 수, figure 표시 여부를 받습니다.
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze parking-lot validation results."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing baseline/, full_rl/, no_projection/.",
    )

    parser.add_argument(
        "--report",
        choices=["multiseed", "ablation", "all"],
        default="all",
        help="Report type to generate.",
    )

    parser.add_argument(
        "--n-seeds",
        "--n_seeds",
        dest="n_seeds",
        type=int,
        default=5,
        help="Number of seed folders to analyze.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show matplotlib windows after saving figures.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# 시계열 로그 파일 로드
# time/value 형식의 txt 파일을 읽고, steady_start 이후 데이터만 반환합니다.
# -----------------------------------------------------------------------------

def load_series(path):
    path = Path(path)

    if not path.exists():
        return None, None

    try:
        data = np.genfromtxt(path, invalid_raise=False)
    except Exception:
        return None, None

    if data is None:
        return None, None

    data = np.asarray(data)

    if data.size == 0:
        return None, None

    data = np.atleast_2d(data)

    if data.shape[1] < 2:
        return None, None

    mask = data[:, 0] > STEADY_START

    if np.sum(mask) == 0:
        return None, None

    return data[mask, 0], data[mask, 1]


# -----------------------------------------------------------------------------
# Burst 구간 필터링
# 50~100초 구간의 값을 추출하기 위한 용도입니다.
# -----------------------------------------------------------------------------

def filter_burst(t, y):
    t = np.asarray(t)
    y = np.asarray(y)

    mask = (t >= BURST_START) & (t <= BURST_END)

    return y[mask]


# -----------------------------------------------------------------------------
# 안전한 통계 및 QoS ratio 계산
# 값이 비어 있으면 NaN을 반환합니다.
# -----------------------------------------------------------------------------

def safe_mean(values):
    values = np.asarray(values)

    if values.size == 0:
        return np.nan

    return float(np.mean(values))


def safe_ratio_ge(values, threshold):
    values = np.asarray(values)

    if values.size == 0:
        return np.nan

    return float(np.mean(values >= threshold) * 100.0)


def safe_ratio_le(values, threshold):
    values = np.asarray(values)

    if values.size == 0:
        return np.nan

    return float(np.mean(values <= threshold) * 100.0)


# -----------------------------------------------------------------------------
# 단일 run metric 계산
# S2 primary video-like QoS와 S1 FTP impact를 함께 계산합니다.
# -----------------------------------------------------------------------------

def compute_metrics(gp2_t, gp2, rtt2_t, rtt2, gp1_t=None, gp1=None):
    burst_gp2 = filter_burst(gp2_t, gp2)
    burst_rtt2 = filter_burst(rtt2_t, rtt2)

    metrics = {
        "s2_gp_mean": safe_mean(gp2),
        "s2_gp_qos": safe_ratio_ge(gp2, MIN_GP),
        "s2_rtt_mean": safe_mean(rtt2),
        "s2_rtt_qos": safe_ratio_le(rtt2, RTT_LIMIT),
        "s2_burst_gp_mean": safe_mean(burst_gp2),
        "s2_burst_gp_qos": safe_ratio_ge(burst_gp2, MIN_GP),
        "s2_burst_rtt_mean": safe_mean(burst_rtt2),
        "s2_burst_rtt_qos": safe_ratio_le(burst_rtt2, RTT_LIMIT),
    }

    if gp1 is not None and gp1_t is not None:
        burst_gp1 = filter_burst(gp1_t, gp1)

        metrics.update(
            {
                "s1_gp_mean": safe_mean(gp1),
                "s1_burst_gp_mean": safe_mean(burst_gp1),
                "s1_survival": safe_ratio_ge(gp1, MIN_FTP_GP),
            }
        )

    return metrics


# -----------------------------------------------------------------------------
# Baseline 결과 로드
# baseline/ 폴더에서 baseline_*.txt 파일을 읽어옵니다.
# -----------------------------------------------------------------------------

def load_baseline(data_dir):
    base_dir = Path(data_dir) / "baseline"

    gp2_t, gp2 = load_series(base_dir / "baseline_goodput_s2.txt")
    rtt2_t, rtt2 = load_series(base_dir / "baseline_rtt_s2.txt")
    gp1_t, gp1 = load_series(base_dir / "baseline_goodput_s1.txt")

    if gp2 is None or rtt2 is None:
        return None

    return compute_metrics(gp2_t, gp2, rtt2_t, rtt2, gp1_t, gp1)


# -----------------------------------------------------------------------------
# Variant seed 결과 수집
# full_rl/seed_N 경로와 no_projection/seed_N에서 rl_*.txt를 읽어 metric list를 만듭니다.
# -----------------------------------------------------------------------------

def collect_variant(data_dir, variant, n_seeds):
    records = []
    valid_seeds = []

    for seed in range(1, n_seeds + 1):
        seed_dir = Path(data_dir) / variant / f"seed_{seed}"

        gp2_t, gp2 = load_series(seed_dir / "rl_goodput_s2.txt")
        rtt2_t, rtt2 = load_series(seed_dir / "rl_rtt_s2.txt")
        gp1_t, gp1 = load_series(seed_dir / "rl_goodput_s1.txt")

        if gp2 is None or rtt2 is None:
            print(f"[Warning] {variant}/seed_{seed}: missing S2 data, skipped.")
            continue

        records.append(compute_metrics(gp2_t, gp2, rtt2_t, rtt2, gp1_t, gp1))
        valid_seeds.append(seed)

    return records, valid_seeds


# -----------------------------------------------------------------------------
# 평균 ± 표준편차 집계
# 각 metric별 mean, std, n을 계산합니다.
# -----------------------------------------------------------------------------

def aggregate(records):
    if not records:
        return {}

    keys = sorted({key for record in records for key in record.keys()})
    output = {}

    for key in keys:
        values = [
            record[key]
            for record in records
            if key in record and not np.isnan(record[key])
        ]

        output[key] = {
            "mean": float(np.mean(values)) if values else np.nan,
            "std": float(np.std(values)) if len(values) > 1 else 0.0,
            "n": len(values),
        }

    return output


# -----------------------------------------------------------------------------
# 출력 formatting
# Baseline 단일값, RL mean ± std, delta 값을 문자열로 변환합니다.
# -----------------------------------------------------------------------------

def fmt_value(value, digits=2):
    if value is None or np.isnan(value):
        return "N/A"

    return f"{value:.{digits}f}"


def fmt_agg(agg, key, digits=2):
    if key not in agg:
        return "N/A"

    mean = agg[key]["mean"]
    std = agg[key]["std"]

    if np.isnan(mean):
        return "N/A"

    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def fmt_delta(base, agg, key, digits=2):
    base_value = base.get(key, np.nan)
    variant_value = agg.get(key, {}).get("mean", np.nan)

    if np.isnan(base_value) or np.isnan(variant_value):
        return "N/A"

    return f"{variant_value - base_value:+.{digits}f}"


def get_agg_mean(agg, key):
    return agg.get(key, {}).get("mean", np.nan)


def get_agg_std(agg, key):
    return agg.get(key, {}).get("std", 0.0)


# -----------------------------------------------------------------------------
# Multi-seed 비교표 출력
# Baseline Cubic과 Full RL seed 평균 ± 표준편차를 비교합니다.
# -----------------------------------------------------------------------------

def print_multiseed_table(baseline, full_agg, full_seeds):
    print()
    print("=" * 78)
    print(
        "  Multi-Seed Comparison [Parking-lot Single-Burst] "
        f"(Full RL n={len(full_seeds)} seeds)"
    )
    print("=" * 78)
    print(f"{'Metric':<42} {'Baseline':>10} {'Full RL':>18} {'Δ':>8}")
    print("-" * 78)

    rows = [
        ("S2 Overall Goodput mean (Mbps)", "s2_gp_mean", 2),
        ("S2 Overall ≥5Mbps ratio (%)", "s2_gp_qos", 2),
        ("S2 Overall RTT mean (ms)", "s2_rtt_mean", 2),
        ("S2 Overall RTT ≤120ms ratio (%)", "s2_rtt_qos", 2),
        ("─── Burst Period 50~100s ───", None, None),
        ("S2 Burst Goodput mean (Mbps)", "s2_burst_gp_mean", 2),
        ("S2 Burst ≥5Mbps ratio (%)", "s2_burst_gp_qos", 2),
        ("S2 Burst RTT mean (ms)", "s2_burst_rtt_mean", 2),
        ("S2 Burst RTT ≤120ms ratio (%)", "s2_burst_rtt_qos", 2),
        ("S1 FTP Overall Goodput mean (Mbps)", "s1_gp_mean", 2),
        ("S1 FTP Burst Goodput mean (Mbps)", "s1_burst_gp_mean", 2),
        ("S1 FTP ≥1Mbps ratio (%)", "s1_survival", 2),
    ]

    for label, key, digits in rows:
        if key is None:
            print(f"  {label}")
            continue

        print(
            f"  {label:<40} "
            f"{fmt_value(baseline.get(key, np.nan), digits):>10} "
            f"{fmt_agg(full_agg, key, digits):>18} "
            f"{fmt_delta(baseline, full_agg, key, digits):>8}"
        )

    print("=" * 78)
    print(f"  Valid Full RL seeds: {full_seeds}")
    print("=" * 78)


# -----------------------------------------------------------------------------
# Ablation 비교표 출력
# Baseline, Full RL, No-Projection을 3-way로 비교합니다.
# -----------------------------------------------------------------------------

def print_ablation_table(baseline, full_agg, no_proj_agg, full_seeds, no_proj_seeds):
    width = 96

    print()
    print("=" * width)
    print("  Ablation Study: Baseline vs Full RL vs No-Projection")
    print(
        f"  Full RL n={len(full_seeds)} seeds | "
        f"No-Projection n={len(no_proj_seeds)} seeds | values: mean ± std"
    )
    print("=" * width)
    print(
        f"{'Metric':<38} {'Baseline':>10} {'Full RL':>18} "
        f"{'No-Proj':>18} {'Δ Full':>8} {'Δ NoP':>8}"
    )
    print("-" * width)

    rows = [
        ("S2 Overall Goodput mean (Mbps)", "s2_gp_mean", 2),
        ("S2 Overall ≥5Mbps ratio (%)", "s2_gp_qos", 2),
        ("S2 Overall RTT mean (ms)", "s2_rtt_mean", 2),
        ("S2 Overall RTT ≤120ms (%)", "s2_rtt_qos", 2),
        ("─── Burst Period 50~100s ───", None, None),
        ("S2 Burst Goodput mean (Mbps)", "s2_burst_gp_mean", 2),
        ("S2 Burst ≥5Mbps ratio (%)", "s2_burst_gp_qos", 2),
        ("S2 Burst RTT mean (ms)", "s2_burst_rtt_mean", 2),
        ("S1 FTP Overall mean (Mbps)", "s1_gp_mean", 2),
        ("S1 FTP Burst mean (Mbps)", "s1_burst_gp_mean", 2),
    ]

    for label, key, digits in rows:
        if key is None:
            print(f"  {label}")
            continue

        print(
            f"  {label:<36} "
            f"{fmt_value(baseline.get(key, np.nan), digits):>10} "
            f"{fmt_agg(full_agg, key, digits):>18} "
            f"{fmt_agg(no_proj_agg, key, digits):>18} "
            f"{fmt_delta(baseline, full_agg, key, digits):>8} "
            f"{fmt_delta(baseline, no_proj_agg, key, digits):>8}"
        )

    print("=" * width)

    base_burst_qos = baseline.get("s2_burst_gp_qos", np.nan)
    full_burst_qos = get_agg_mean(full_agg, "s2_burst_gp_qos")
    no_proj_burst_qos = get_agg_mean(no_proj_agg, "s2_burst_gp_qos")

    if not any(np.isnan(v) for v in [base_burst_qos, full_burst_qos, no_proj_burst_qos]):
        print()
        print("[Ablation Interpretation]")
        print(
            f"  Full RL - No-Projection burst QoS gain: "
            f"{full_burst_qos - no_proj_burst_qos:+.1f}%p"
        )
        print(
            f"  Baseline {base_burst_qos:.1f}% | "
            f"No-Projection {no_proj_burst_qos:.1f}% | "
            f"Full RL {full_burst_qos:.1f}%"
        )
        print(
            "  해석: No-Projection 대비 Full RL이 더 높다면, "
            "action projection이 S2 보호에 추가적으로 기여한 것으로 볼 수 있습니다."
        )

    print()


# -----------------------------------------------------------------------------
# Multi-seed 요약 그래프 생성
# Baseline 단일값과 Full RL mean ± std를 bar chart로 비교합니다.
# -----------------------------------------------------------------------------

def plot_multiseed_summary(baseline, full_agg, save_path, show=False):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    fig.suptitle(
        "Multi-Seed Comparison — Parking-lot Single-Burst\n"
        f"(Full RL n={full_agg.get('s2_gp_qos', {}).get('n', '?')} seeds, burst: 50~100s)",
        fontsize=13,
        y=1.01,
    )

    def bar_compare(ax, key, title, ylabel, unit="", ylim=None):
        base_value = baseline.get(key, np.nan)
        rl_mean = get_agg_mean(full_agg, key)
        rl_std = get_agg_std(full_agg, key)

        x = np.arange(2)
        values = [base_value, rl_mean]
        errors = [0.0, rl_std]
        labels = ["Baseline\nCubic", "Full\nRL"]

        bars = ax.bar(
            x,
            values,
            yerr=errors,
            capsize=6,
            width=0.55,
            alpha=0.82,
            error_kw={"linewidth": 1.5},
        )

        for bar, value, error in zip(bars, values, errors):
            if np.isnan(value):
                continue

            text = f"{value:.1f}{unit}"

            if error > 0:
                text += f"\n±{error:.1f}"

            ax.annotate(
                text,
                xy=(bar.get_x() + bar.get_width() / 2, value + error),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.3)

        if ylim is not None:
            ax.set_ylim(*ylim)

    bar_compare(
        axes[0, 0],
        "s2_gp_qos",
        "Overall S2 Goodput ≥5Mbps",
        "Ratio (%)",
        "%",
        (0, 110),
    )

    bar_compare(
        axes[0, 1],
        "s2_burst_gp_qos",
        "Burst S2 Goodput ≥5Mbps",
        "Ratio (%)",
        "%",
        (0, 110),
    )

    bar_compare(
        axes[1, 0],
        "s2_burst_gp_mean",
        "Burst S2 Goodput",
        "Goodput (Mbps)",
    )
    axes[1, 0].axhline(
        MIN_GP,
        linestyle="--",
        linewidth=1.3,
        label=f"Min {MIN_GP:.0f}Mbps",
    )
    axes[1, 0].axhline(
        TARGET_GP,
        linestyle=":",
        linewidth=1.3,
        label=f"Target {TARGET_GP:.0f}Mbps",
    )
    axes[1, 0].legend(fontsize=9)

    bar_compare(
        axes[1, 1],
        "s2_burst_rtt_mean",
        "Burst S2 RTT",
        "RTT (ms)",
    )
    axes[1, 1].axhline(
        RTT_LIMIT,
        linestyle="--",
        linewidth=1.3,
        label=f"Limit {RTT_LIMIT:.0f}ms",
    )
    axes[1, 1].legend(fontsize=9)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[Plot saved] {save_path}")

    if show:
        plt.show()

    plt.close(fig)


# -----------------------------------------------------------------------------
# Ablation 요약 그래프 생성
# Baseline, Full RL, No-Projection을 3-way bar chart로 비교합니다.
# -----------------------------------------------------------------------------

def plot_ablation_summary(baseline, full_agg, no_proj_agg, save_path, show=False):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    fig.suptitle(
        "Ablation Study — Baseline vs Full RL vs No-Projection\n"
        "(burst period 50~100s, bars = mean ± std)",
        fontsize=12,
    )

    def triple_bar(ax, key, title, ylabel, ylim=None, threshold=None):
        base_value = baseline.get(key, np.nan)
        full_mean = get_agg_mean(full_agg, key)
        full_std = get_agg_std(full_agg, key)
        no_proj_mean = get_agg_mean(no_proj_agg, key)
        no_proj_std = get_agg_std(no_proj_agg, key)

        x = np.arange(3)
        values = [base_value, full_mean, no_proj_mean]
        errors = [0.0, full_std, no_proj_std]
        labels = ["Baseline\nCubic", "Full RL\n(w/ Proj)", "No-Proj\nAblation"]

        bars = ax.bar(
            x,
            values,
            yerr=errors,
            capsize=6,
            width=0.55,
            alpha=0.82,
            error_kw={"linewidth": 1.5},
        )

        for bar, value, error in zip(bars, values, errors):
            if np.isnan(value):
                continue

            text = f"{value:.1f}"

            if error > 0:
                text += f"\n±{error:.1f}"

            ax.annotate(
                text,
                xy=(bar.get_x() + bar.get_width() / 2, value + error),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        if threshold is not None:
            ax.axhline(
                threshold,
                linestyle="--",
                linewidth=1.2,
                label=f"threshold={threshold}",
            )
            ax.legend(fontsize=8)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.3)

        if ylim is not None:
            ax.set_ylim(*ylim)

    triple_bar(
        axes[0],
        "s2_burst_gp_qos",
        "Burst S2 ≥5Mbps",
        "Compliance (%)",
        ylim=(0, 110),
    )

    triple_bar(
        axes[1],
        "s2_burst_gp_mean",
        "Burst S2 Goodput",
        "Goodput (Mbps)",
        threshold=MIN_GP,
    )

    triple_bar(
        axes[2],
        "s1_gp_mean",
        "S1 FTP Overall Goodput",
        "Goodput (Mbps)",
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[Plot saved] {save_path}")

    if show:
        plt.show()

    plt.close(fig)


# -----------------------------------------------------------------------------
# Main 분석 흐름
# baseline, full_rl, no_projection 결과를 읽고 선택한 report를 생성합니다.
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    data_dir = Path(args.data_dir)

    print()
    print("[analyze_parkinglot_validation]")
    print(f"Data dir: {data_dir}")
    print(f"Report  : {args.report}")
    print(f"Seeds   : 1~{args.n_seeds}")

    baseline = load_baseline(data_dir)

    if baseline is None:
        print("[Error] Baseline data not found.")
        print(f"Expected: {data_dir}/baseline/baseline_goodput_s2.txt")
        return

    full_records = []
    full_seeds = []
    full_agg = {}

    no_proj_records = []
    no_proj_seeds = []
    no_proj_agg = {}

    if args.report in ("multiseed", "ablation", "all"):
        full_records, full_seeds = collect_variant(
            data_dir,
            "full_rl",
            args.n_seeds,
        )
        full_agg = aggregate(full_records)

    if args.report in ("ablation", "all"):
        no_proj_records, no_proj_seeds = collect_variant(
            data_dir,
            "no_projection",
            args.n_seeds,
        )
        no_proj_agg = aggregate(no_proj_records)

    if args.report in ("multiseed", "all"):
        if not full_records:
            print("[Error] Full RL data not found. Run full RL validation first.")
        else:
            print_multiseed_table(baseline, full_agg, full_seeds)
            plot_multiseed_summary(
                baseline,
                full_agg,
                data_dir / "multiseed_report.png",
                show=args.show,
            )

    if args.report in ("ablation", "all"):
        if not full_records:
            print("[Error] Full RL data not found. Ablation report requires full_rl data.")
        elif not no_proj_records:
            print("[Error] No-Projection data not found. Run no-projection validation first.")
        else:
            print_ablation_table(
                baseline,
                full_agg,
                no_proj_agg,
                full_seeds,
                no_proj_seeds,
            )
            plot_ablation_summary(
                baseline,
                full_agg,
                no_proj_agg,
                data_dir / "ablation_report.png",
                show=args.show,
            )


if __name__ == "__main__":
    main()