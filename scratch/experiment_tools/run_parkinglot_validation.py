#!/usr/bin/env python3

"""
run_parkinglot_validation.py

parking-lot single-burst scenario에서 다음 검증 실험을 실행합니다.

1. Baseline Cubic
2. Full RL multi-seed
3. No-Projection ablation multi-seed

사용 위치:
    ns-3.38 최상위 폴더에서 실행

터미널에서는 이렇게 실행:
    python3 scratch/experiment_tools/run_parkinglot_validation.py --task both --n-seeds 5

결과:
    scratch/results/parkinglot_validation/
    ├── baseline/
    ├── full_rl/seed_1 ...
    ├── no_projection/seed_1 ...
    └── logs/
"""

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# -----------------------------------------------------------------------------
# 기본 실행 경로 및 실험 설정
# 기존 main topology와 train script를 그대로 사용합니다. (parkinglot topology와 train_agent)
# -----------------------------------------------------------------------------

DEFAULT_NS3_SCRIPT = "scratch/qos_rl_tcp/topo3_parkinglot"
DEFAULT_FULL_TRAIN_SCRIPT = "scratch/rl_algorithm/train_agent.py"
DEFAULT_NOPROJ_TRAIN_SCRIPT = "scratch/rl_algorithm/train_agent_ablation.py"

DEFAULT_WORK_RESULT_DIR = Path("scratch/results/topo3_rtt")
DEFAULT_OUT_DIR = Path("scratch/results/parkinglot_validation")

DEFAULT_TIMESTEPS = 1536
DEFAULT_STARTUP_WAIT = 3.0
DEFAULT_NS3_TIMEOUT = 120.0


# -----------------------------------------------------------------------------
# 실행 인자 정의
# 실행할 task, seed 수, timesteps, 결과 경로 등을 설정합니다.
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run parking-lot validation: baseline, full RL, no-projection."
    )

    parser.add_argument(
        "--task",
        choices=["baseline", "full", "no-proj", "both"],
        default="both",
        help=(
            "baseline: run Baseline only, "
            "full: run Baseline + Full RL seeds, "
            "no-proj: run Baseline if needed + No-Projection seeds, "
            "both: run Baseline + Full RL + No-Projection seeds."
        ),
    )

    parser.add_argument(
        "--n-seeds",
        "--n_seeds",
        dest="n_seeds",
        type=int,
        default=5,
        help="Number of seeds to run.",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="PPO timesteps per seed.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Directory to store validation results.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing baseline/seed result folders.",
    )

    parser.add_argument(
        "--startup-wait",
        type=float,
        default=DEFAULT_STARTUP_WAIT,
        help="Seconds to wait after launching ns-3 RL before starting Python agent.",
    )

    parser.add_argument(
        "--ns3-timeout",
        type=float,
        default=DEFAULT_NS3_TIMEOUT,
        help="Seconds to wait for ns-3 after Python agent exits.",
    )

    parser.add_argument(
        "--ns3-script",
        type=str,
        default=DEFAULT_NS3_SCRIPT,
        help="ns-3 scratch target to run.",
    )

    parser.add_argument(
        "--full-train-script",
        type=str,
        default=DEFAULT_FULL_TRAIN_SCRIPT,
        help="Python training script for Full RL.",
    )

    parser.add_argument(
        "--no-proj-train-script",
        type=str,
        default=DEFAULT_NOPROJ_TRAIN_SCRIPT,
        help="Python training script for No-Projection ablation.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# 실행 환경 검증
# ns-3 최상위 폴더에서 실행 중인지와 필요한 Python script 존재 여부를 확인합니다.
# -----------------------------------------------------------------------------

def validate_environment(args):
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be positive.")

    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive.")

    if not Path("./ns3").is_file():
        raise FileNotFoundError(
            "./ns3 not found. Run this script from the ns-3 root directory."
        )

    if args.task in ("full", "both") and not Path(args.full_train_script).is_file():
        raise FileNotFoundError(
            f"Full RL train script not found: {args.full_train_script}"
        )

    if args.task in ("no-proj", "both") and not Path(args.no_proj_train_script).is_file():
        raise FileNotFoundError(
            f"No-Projection train script not found: {args.no_proj_train_script}"
        )


# -----------------------------------------------------------------------------
# 결과 디렉토리 준비
# validation 결과와 로그를 저장할 폴더를 생성합니다.
# -----------------------------------------------------------------------------

def prepare_output_dirs(out_dir):
    out_dir = Path(out_dir)

    (out_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (out_dir / "full_rl").mkdir(parents=True, exist_ok=True)
    (out_dir / "no_projection").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    return out_dir


# -----------------------------------------------------------------------------
# 기존 작업 결과 백업
# topo3_parkinglot.cc는 scratch/results/topo3_rtt에 로그를 저장하도록 되어있어서 해당 실험에서는 기존 main 결과 보존을 위해 실행 전에 백업을 진행했습니다.
# -----------------------------------------------------------------------------

def backup_work_results(work_dir, out_dir):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    files = []

    for pattern in ("*.txt", "*.png"):
        files.extend(work_dir.glob(pattern))

    if not files:
        print("[Backup] No existing topo3_rtt files to backup.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(out_dir) / f"_work_backup_topo3_rtt_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        shutil.copy2(file_path, backup_dir / file_path.name)

    print(f"[Backup] Existing topo3_rtt files copied to: {backup_dir}")

    return backup_dir


# -----------------------------------------------------------------------------
# 작업 결과 복원
# validation 실행 중 생성된 topo3_rtt 임시 결과를 제거하고, 실행 전 백업 결과가 있으면 복원하게끔 했습니다.
# -----------------------------------------------------------------------------

def restore_work_results(work_dir, backup_dir):
    work_dir = Path(work_dir)

    for pattern in ("baseline_*.txt", "rl_*.txt", "*.png"):
        for file_path in work_dir.glob(pattern):
            file_path.unlink(missing_ok=True)

    if backup_dir is None:
        print("[Restore] No previous topo3_rtt files to restore.")
        return

    backup_dir = Path(backup_dir)

    for file_path in backup_dir.glob("*"):
        if file_path.is_file():
            shutil.copy2(file_path, work_dir / file_path.name)

    print(f"[Restore] Previous topo3_rtt files restored from: {backup_dir}")


# -----------------------------------------------------------------------------
# 파일 처리 helper
# 작업 디렉토리에서 생성된 result 파일을 결과 디렉토리로 복사합니다. (topo3_rtt 폴더에서 parkinglot_validation 하우ㅣ폴더로)
# ----------------------------------------------------------------------------- 

def remove_matching_files(directory, patterns):
    directory = Path(directory)

    for pattern in patterns:
        for file_path in directory.glob(pattern):
            file_path.unlink(missing_ok=True)


def copy_matching_files(src_dir, dst_dir, pattern, required=True):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob(pattern))

    if required and not files:
        raise FileNotFoundError(f"No files matched: {src_dir}/{pattern}")

    for file_path in files:
        shutil.copy2(file_path, dst_dir / file_path.name)

    return files


def has_required_files(directory, patterns):
    directory = Path(directory)

    return all(any(directory.glob(pattern)) for pattern in patterns)


# -----------------------------------------------------------------------------
# ns-3 process 종료 helper
# Python agent 실패 등으로 ns-3 process가 남아 있을 경우 안전하게 종료되도록합니다.
# -----------------------------------------------------------------------------

def terminate_process(proc):
    if proc is None:
        return

    if proc.poll() is not None:
        return

    proc.terminate()

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


# -----------------------------------------------------------------------------
# Baseline Cubic 실행
# Baseline은 seed와 무관하기에 1회만 실행합니다.
# -----------------------------------------------------------------------------

def run_baseline(args, work_dir, out_dir):
    baseline_dir = Path(out_dir) / "baseline"

    if (
        not args.overwrite
        and has_required_files(baseline_dir, ["baseline_goodput_s2.txt", "baseline_rtt_s2.txt"])
    ):
        print(f"[Baseline] Existing result found. Skipping: {baseline_dir}")
        return

    if args.overwrite and baseline_dir.exists():
        shutil.rmtree(baseline_dir)

    baseline_dir.mkdir(parents=True, exist_ok=True)

    remove_matching_files(work_dir, ["baseline_*.txt"])

    print()
    print("[Step] Running Baseline Cubic...")

    subprocess.run(
        ["./ns3", "run", f"{args.ns3_script} --mode=Baseline"],
        check=True,
    )

    copied = copy_matching_files(
        work_dir,
        baseline_dir,
        "baseline_*.txt",
        required=True,
    )

    print(f"[Baseline done] {len(copied)} files copied to: {baseline_dir}")


# -----------------------------------------------------------------------------
# 단일 seed RL 실행
# ns-3 RL simulation을 백그라운드로 실행한 뒤, Python PPO agent를 연결합니다.
# variant에 따라 Full RL 또는 No-Projection train script를 사용합니다.
# -----------------------------------------------------------------------------

def run_single_seed(args, variant, seed, work_dir, out_dir):
    if variant == "full_rl":
        train_script = args.full_train_script
        variant_label = "Full RL"
    elif variant == "no_projection":
        train_script = args.no_proj_train_script
        variant_label = "No-Projection"
    else:
        raise ValueError(f"Unknown variant: {variant}")

    seed_dir = Path(out_dir) / variant / f"seed_{seed}"

    if (
        not args.overwrite
        and has_required_files(seed_dir, ["rl_goodput_s2.txt", "rl_rtt_s2.txt"])
    ):
        print(f"[{variant_label}][Seed {seed}] Existing result found. Skipping.")
        return

    if args.overwrite and seed_dir.exists():
        shutil.rmtree(seed_dir)

    seed_dir.mkdir(parents=True, exist_ok=True)

    remove_matching_files(work_dir, ["rl_*.txt"])

    logs_dir = Path(out_dir) / "logs"
    ns3_log_path = logs_dir / f"{variant}_seed{seed}_ns3.log"
    train_log_path = logs_dir / f"{variant}_seed{seed}_train.log"

    print("----------------------------------------------------")
    print(f"[{variant_label}][Seed {seed}/{args.n_seeds}] Starting ns-3 RL simulation...")

    ns3_proc = None

    try:
        with ns3_log_path.open("w") as ns3_log:
            ns3_proc = subprocess.Popen(
                ["./ns3", "run", f"{args.ns3_script} --mode=RL"],
                stdout=ns3_log,
                stderr=subprocess.STDOUT,
            )

        time.sleep(args.startup_wait)

        model_save_path = seed_dir / f"model_seed{seed}"

        print(f"[{variant_label}][Seed {seed}] Starting Python agent...")

        with train_log_path.open("w") as train_log:
            subprocess.run(
                [
                    sys.executable,
                    train_script,
                    f"--seed={seed}",
                    f"--timesteps={args.timesteps}",
                    f"--save-path={model_save_path}",
                ],
                stdout=train_log,
                stderr=subprocess.STDOUT,
                check=True,
            )

        try:
            ns3_return_code = ns3_proc.wait(timeout=args.ns3_timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"ns-3 did not exit within {args.ns3_timeout}s "
                f"after Python agent finished. Check log: {ns3_log_path}"
            )

        ns3_proc = None

        if ns3_return_code != 0:
            raise RuntimeError(
                f"ns-3 RL simulation failed for {variant_label} seed {seed}. "
                f"Check log: {ns3_log_path}"
            )

        copied = copy_matching_files(
            work_dir,
            seed_dir,
            "rl_*.txt",
            required=True,
        )

        print(f"[{variant_label}][Seed {seed}] Done.")
        print(f"  Results: {seed_dir}")
        print(f"  Model:   {model_save_path}.zip")
        print(f"  Logs:    {ns3_log_path}, {train_log_path}")

    except Exception:
        terminate_process(ns3_proc)
        raise


# -----------------------------------------------------------------------------
# Variant별 seed 반복 실행
# Full RL / No-Projection seed_N 실행을 반복합니다.
# -----------------------------------------------------------------------------

def run_variant(args, variant, work_dir, out_dir):
    print()
    print(f"[Step] Running variant: {variant}")

    for seed in range(1, args.n_seeds + 1):
        run_single_seed(args, variant, seed, work_dir, out_dir)
        time.sleep(2)


# -----------------------------------------------------------------------------
# Main 실행 흐름
# task에 따라 Baseline, Full RL, No-Projection ablation을 실행하고, 작업 경로를 실행 전 상태로 복원합니다.
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    validate_environment(args)

    work_dir = DEFAULT_WORK_RESULT_DIR
    out_dir = prepare_output_dirs(args.out_dir)
    work_backup_dir = backup_work_results(work_dir, out_dir)

    print("========================================================")
    print("[run_parkinglot_validation]")
    print(f"Task        : {args.task}")
    print(f"Seeds       : 1~{args.n_seeds}")
    print(f"Timesteps   : {args.timesteps}")
    print(f"ns-3 script : {args.ns3_script}")
    print(f"Work dir    : {work_dir}")
    print(f"Output dir  : {out_dir}")
    print("========================================================")

    try:
        if args.task in ("baseline", "full", "no-proj", "both"):
            run_baseline(args, work_dir, out_dir)

        if args.task in ("full", "both"):
            run_variant(args, "full_rl", work_dir, out_dir)

        if args.task in ("no-proj", "both"):
            run_variant(args, "no_projection", work_dir, out_dir)

    finally:
        restore_work_results(work_dir, work_backup_dir)

    print()
    print("========================================================")
    print("[run_parkinglot_validation] Complete.")
    print(f"Results in: {out_dir}")
    print()
    print("Next step:")
    print(
        "  python3 scratch/experiment_tools/analyze_parkinglot_validation.py "
        f"--data-dir {out_dir} --report all --n-seeds {args.n_seeds}"
    )
    print("========================================================")


if __name__ == "__main__":
    main()
