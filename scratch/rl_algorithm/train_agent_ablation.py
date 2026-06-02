import argparse
import time
from pathlib import Path

import numpy as np

# ns3-gym 또는 일부 구버전 dependency에서 np.float를 참조하는 경우를 대비합니다.
if not hasattr(np, "float"):
    np.float = float

from ns3gym import ns3env
from stable_baselines3 import PPO

from qos_reward_env_ablation import QosRewardWrapperNoProjection


# -----------------------------------------------------------------------------
# 기본 경로 및 시뮬레이션 상수 정의
# No-Projection ablation 학습에 사용할 기본 모델 저장 경로, ns-3 제어 주기, 전체 시뮬레이션 시간, PPO rollout 길이, ns3-gym 포트를 정의합니다.
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ABLATION_MODEL = SCRIPT_DIR / "qos_tcp_ablation_model"

STEP_TIME = 0.1
SIM_TIME = 160.0
MAX_EPISODE_STEPS = int(SIM_TIME / STEP_TIME)

PPO_N_STEPS = 256
NS3GYM_PORT = 5555


# -----------------------------------------------------------------------------
# 실행 인자 정의
# No-Projection ablation은 기존 모델을 불러오는 eval이 아니라 seed별 fresh training입니다.
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the No-Projection ablation PPO agent. "
            "The ns-3 simulation must be launched manually with --mode=RL."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for PPO training.",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=1536,
        help=(
            "Training timesteps. "
            "This must stay below 1600 because the ns-3 episode length is 160s / 0.1s."
        ),
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default=str(DEFAULT_ABLATION_MODEL),
        help=(
            "Path to save the trained ablation PPO model. "
            "The .zip suffix may be omitted."
        ),
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# 실행 인자 검증
# ns-3 episode 길이를 초과하는 timesteps 설정을 방지합니다.
# -----------------------------------------------------------------------------

def validate_args(args):
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive.")

    if args.timesteps >= MAX_EPISODE_STEPS:
        raise ValueError(
            "TOTAL_TIMESTEPS must be smaller than one ns-3 episode length "
            f"when startSim=False. Got {args.timesteps}, "
            f"episode limit is {MAX_EPISODE_STEPS}."
        )


# -----------------------------------------------------------------------------
# 모델 저장 경로 처리
# save-path의 상위 폴더가 없으면 생성합니다.
# -----------------------------------------------------------------------------

def ensure_parent_dir(path_str):
    path = Path(path_str)
    parent = path.parent

    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 실행 설정 출력
# ablation 조건과 PPO 설정을 터미널에 출력해 실험 로그 확인용으로 사용했습니다.
# -----------------------------------------------------------------------------

def print_run_configuration(args):
    print("=" * 60, flush=True)
    print("[Ablation Configuration]", flush=True)
    print("Wrapper: QosRewardWrapperNoProjection", flush=True)
    print("Projection: OFF", flush=True)
    print("Reward function: same as Full RL", flush=True)
    print(f"Simulation time: {SIM_TIME:.0f}s", flush=True)
    print(f"Control interval: {STEP_TIME:.1f}s", flush=True)
    print(f"Max episode steps: {MAX_EPISODE_STEPS}", flush=True)
    print(f"Timesteps: {args.timesteps}", flush=True)
    print(f"PPO n_steps: {PPO_N_STEPS}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"Save path: {args.save_path}", flush=True)
    print("startSim: False (manual ns-3 launch mode)", flush=True)
    print("=" * 60, flush=True)


# -----------------------------------------------------------------------------
# ns3-gym 환경 생성
# 수동으로 실행된 ns-3 RL topology에 Python PPO agent를 연결하고, action projection이 제거된 wrapper를 적용합니다.
# -----------------------------------------------------------------------------

def make_env():
    raw_env = ns3env.Ns3Env(
        port=NS3GYM_PORT,
        stepTime=STEP_TIME,
        startSim=False,
    )

    return QosRewardWrapperNoProjection(raw_env)


# -----------------------------------------------------------------------------
# PPO 모델 생성
# Full RL과 같은 PPO hyperparameter를 사용하고, wrapper만 No-Projection ablation 버전으로 둡니다.
# -----------------------------------------------------------------------------

def create_model(env, seed):
    return PPO(
        "MlpPolicy",
        env,
        n_steps=PPO_N_STEPS,
        batch_size=64,
        learning_rate=0.0002,
        seed=seed,
        verbose=1,
    )


# -----------------------------------------------------------------------------
# No-Projection PPO 학습 실행
# PPO learning을 수행하고 학습된 ablation model을 지정된 save-path에 저장합니다.
# -----------------------------------------------------------------------------

def run_training(model, args):
    start_time = time.time()

    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=True,
    )

    elapsed = time.time() - start_time
    avg_latency_ms = (elapsed / args.timesteps) * 1000.0

    ensure_parent_dir(args.save_path)
    model.save(args.save_path)

    control_budget_ms = STEP_TIME * 1000.0
    cpu_headroom = (1.0 - avg_latency_ms / control_budget_ms) * 100.0

    print("=" * 60, flush=True)
    print("[Ablation Training Report]", flush=True)
    print(f"Training timesteps: {args.timesteps}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"Elapsed wall time: {elapsed:.2f}s", flush=True)
    print(f"Avg. step processing latency: {avg_latency_ms:.2f} ms", flush=True)
    print(f"Control interval budget: {control_budget_ms:.2f} ms", flush=True)
    print(f"CPU headroom estimate: {cpu_headroom:.1f}%", flush=True)
    print(f"Model saved to: {args.save_path}", flush=True)
    print("=" * 60, flush=True)


# -----------------------------------------------------------------------------
# Main 실행 흐름
# 인자 파싱, 검증, seed 설정, ns3-gym 환경 생성, No-Projection PPO 학습, 모델 저장, 환경 종료를 순서대로 수행합니다.
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    validate_args(args)

    np.random.seed(args.seed)

    print("\n[Ablation] No-Projection RL-TCP training started", flush=True)
    print_run_configuration(args)

    env = make_env()

    try:
        model = create_model(env, args.seed)
        run_training(model, args)

    finally:
        env.close()


if __name__ == "__main__":
    main()