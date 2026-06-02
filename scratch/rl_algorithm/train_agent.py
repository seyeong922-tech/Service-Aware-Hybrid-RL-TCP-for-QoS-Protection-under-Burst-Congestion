import argparse
import time
from pathlib import Path

import numpy as np

# ns3-gym 또는 일부 구버전 dependency에서 np.float를 참조하는 경우를 대비합니다.
if not hasattr(np, "float"):
    np.float = float

from ns3gym import ns3env
from stable_baselines3 import PPO

from qos_reward_env import QosRewardWrapper


# -----------------------------------------------------------------------------
# 기본 경로 및 시뮬레이션 상수 정의
# 모델 저장 기본 경로, ns-3 제어 주기(step time=0.1), 전체 시뮬레이션 시간(160초), PPO rollout 길이(256), ns3-gym 연결 포트를 정의합니다.
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STABLE_MODEL = SCRIPT_DIR / "qos_tcp_stable_model"

STEP_TIME = 0.1
SIM_TIME = 160.0
MAX_EPISODE_STEPS = int(SIM_TIME / STEP_TIME)

PPO_N_STEPS = 256
NS3GYM_PORT = 5555


# -----------------------------------------------------------------------------
# 실행 인자 정의
# fresh training, continue training, eval-only 실행에 필요한 command-line option을 정의합니다. (어떤 argument가 터미널에 출력될지 정의)
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train or evaluate the PPO-based Hybrid RL-TCP agent. "
            "The ns-3 simulation must be launched manually with --mode=RL."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for PPO training/evaluation.",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=1536,
        help=(
            "Timesteps for one manual ns-3 episode. "
            "This must stay below 1600 because the ns-3 episode length is 160s / 0.1s."
        ),
    )

    parser.add_argument(
        "--continue-train",
        action="store_true",
        help="Load an existing PPO model and continue training.",
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load an existing PPO model and run deterministic evaluation without learning.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Path to PPO model for --continue-train or --eval-only. "
            "The .zip suffix may be omitted. "
            "Example: scratch/rl_algorithm/qos_tcp_general_model"
        ),
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default=str(DEFAULT_STABLE_MODEL),
        help=(
            "Path to save the trained PPO model. "
            "The .zip suffix may be omitted. "
            "Default: scratch/rl_algorithm/qos_tcp_stable_model"
        ),
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# 실행 인자 검증
# 서로 충돌하는 실행 mode를 막고, 이어 학습/eval-only에 필요한 model path가 주어졌는지 검증하는 부분입니다.
# +) ns-3 episode 길이를 넘는 timesteps 설정을 방지합니다.
# -----------------------------------------------------------------------------

def validate_args(args):
    if args.continue_train and args.eval_only:
        raise ValueError("--continue-train and --eval-only cannot be used together.")

    if (args.continue_train or args.eval_only) and args.model_path is None:
        raise ValueError(
            "--model-path is required when using --continue-train or --eval-only."
        )

    if args.timesteps >= MAX_EPISODE_STEPS:
        raise ValueError(
            "TOTAL_TIMESTEPS must be smaller than one ns-3 episode length "
            f"when startSim=False. Got {args.timesteps}, "
            f"episode limit is {MAX_EPISODE_STEPS}."
        )


# -----------------------------------------------------------------------------
# 모델 경로 처리
# 여기서는 Stable-Baselines3 모델 파일을 불러오거나 저장할 때 필요한 경로 처리를 수행합니다. 
# .zip 확장자를 생략해도 기존 모델을 찾을 수 있도록 처리합니다.
# -----------------------------------------------------------------------------

def resolve_existing_model_path(model_path):
    if model_path is None:
        return None

    path = Path(model_path)

    if path.exists():
        return path

    if path.suffix != ".zip":
        zip_path = Path(str(path) + ".zip")
        if zip_path.exists():
            return zip_path

    raise FileNotFoundError(
        f"Model file not found: {model_path} or {model_path}.zip"
    )


def ensure_parent_dir(path_str):
    path = Path(path_str)
    parent = path.parent

    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Gym / Gymnasium API 호환 처리
# Gym과 Gymnasium의 reset(), step() 반환 형식 차이를 정의해둡니다.
# ns3-gym 또는 dependency 버전에 따라 반환 형식이 달라져도 동일한 방식으로 학습/eval 코드를 실행할 수 있도록 합니다.
# -----------------------------------------------------------------------------

def reset_env(env):
    result = env.reset()

    if isinstance(result, tuple):
        return result[0]

    return result


def step_env(env, action):
    result = env.step(action)

    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
        return obs, reward, done, info

    obs, reward, done, info = result
    return obs, reward, done, info


# -----------------------------------------------------------------------------
# 실행 mode 및 로그 출력
# 현재 실행이 fresh training, continue training, eval-only 중 어느 mode인지 판별하고, 실험 설정을 터미널에 출력하도록 합니다.
# -----------------------------------------------------------------------------

def get_run_mode(args):
    if args.eval_only:
        return "eval-only"

    if args.continue_train:
        return "continue-train"

    return "fresh-train"


def print_run_configuration(args, mode):
    print("=" * 60, flush=True)
    print("[Run Configuration]", flush=True)
    print(f"Mode: {mode}", flush=True)
    print(f"Simulation time: {SIM_TIME:.0f}s", flush=True)
    print(f"Control interval: {STEP_TIME:.1f}s", flush=True)
    print(f"Max episode steps: {MAX_EPISODE_STEPS}", flush=True)
    print(f"Timesteps: {args.timesteps}", flush=True)
    print(f"PPO n_steps: {PPO_N_STEPS}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"Model path: {args.model_path}", flush=True)
    print(f"Save path: {args.save_path}", flush=True)
    print("startSim: False (manual ns-3 launch mode)", flush=True)
    print("=" * 60, flush=True)


# -----------------------------------------------------------------------------
# ns3-gym 환경 생성
# 수동으로 실행된 ns-3 RL topology에 Python PPO agent를 연결하고, QoS reward shaping 및 action projection을 포함하는 QosRewardWrapper를 적용합니다.
# -----------------------------------------------------------------------------

def make_env():
    raw_env = ns3env.Ns3Env(
        port=NS3GYM_PORT,
        stepTime=STEP_TIME,
        startSim=False,
    )

    return QosRewardWrapper(raw_env)


# -----------------------------------------------------------------------------
# PPO 모델 생성 또는 불러오기
# 실행 mode에 따라 새 PPO model을 만들거나 기존 model을 불러옵니다.
# fresh-train은 새 모델을 생성하고, continue-train/eval-only는 model-path의 기존 모델을 불러옵니다.
# -----------------------------------------------------------------------------

def create_or_load_model(args, env):
    if args.continue_train or args.eval_only:
        loaded_path = resolve_existing_model_path(args.model_path)

        print(f"[Model] Loading existing model: {loaded_path}", flush=True)

        return PPO.load(
            str(loaded_path),
            env=env,
            seed=args.seed,
        )

    print("[Model] Creating a new PPO model", flush=True)

    return PPO(
        "MlpPolicy",
        env,
        n_steps=PPO_N_STEPS,
        batch_size=64,
        learning_rate=0.0002,
        seed=args.seed,
        verbose=1,
    )


# -----------------------------------------------------------------------------
# Eval-only 실행
# 이 부분에서는 학습된 PPO policy를 고정한 상태로 deterministic evaluation을 수행하도록 정의해뒀습니다.
# 이 mode에서는 PPO 가중치가 업데이트되지 않습니다.
# 단, QosRewardWrapper는 그대로 적용되므로 eval-only 결과는 fixed PPO policy + action projection이 적용된 Hybrid RL-TCP 결과입니다.
# -----------------------------------------------------------------------------

def run_evaluation(model, env, total_timesteps):
    obs = reset_env(env)

    total_reward = 0.0
    actual_steps = 0

    start_time = time.time()

    for _ in range(total_timesteps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = step_env(env, action)

        total_reward += float(reward)
        actual_steps += 1

        if done:
            break

    duration = time.time() - start_time
    avg_latency_ms = (duration / max(actual_steps, 1)) * 1000.0

    print("=" * 60, flush=True)
    print("[Evaluation Report]", flush=True)
    print(f"Evaluation steps: {actual_steps}", flush=True)
    print(f"Total reward: {total_reward:.2f}", flush=True)
    print(f"Elapsed wall time: {duration:.2f}s", flush=True)
    print(f"Avg. step processing latency: {avg_latency_ms:.2f} ms", flush=True)
    print("=" * 60, flush=True)


# -----------------------------------------------------------------------------
# PPO 학습 실행
# 이 부분에서는 PPO learning을 수행하고 학습된 model을 지정된 save-path에 저장합니다.
# multi-topology sequential training에서는 첫 topology에서 fresh-train을 수행하고, 이후 topology에서는 continue-train으로 같은 모델을 이어서 학습합니다.
# -----------------------------------------------------------------------------

def run_training(model, args, mode):
    start_time = time.time()

    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=not args.continue_train,
    )

    duration = time.time() - start_time
    avg_latency_ms = (duration / args.timesteps) * 1000.0

    ensure_parent_dir(args.save_path)
    model.save(args.save_path)

    control_budget_ms = STEP_TIME * 1000.0
    cpu_headroom = (1.0 - avg_latency_ms / control_budget_ms) * 100.0

    print("=" * 60, flush=True)
    print("[Training Report]", flush=True)
    print(f"Mode: {mode}", flush=True)
    print(f"Training timesteps: {args.timesteps}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"Elapsed wall time: {duration:.2f}s", flush=True)
    print(f"Avg. step processing latency: {avg_latency_ms:.2f} ms", flush=True)
    print(f"Control interval budget: {control_budget_ms:.2f} ms", flush=True)
    print(f"CPU headroom estimate: {cpu_headroom:.1f}%", flush=True)
    print(f"Model saved to: {args.save_path}", flush=True)
    print("=" * 60, flush=True)


# -----------------------------------------------------------------------------
# Main 실행 흐름
# 인자 파싱, 인자 검증, seed 설정, ns3-gym 환경 생성, PPO 모델 준비, 학습 또는 평가 실행, 환경 종료를 순서대로 수행합니다.
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    validate_args(args)

    np.random.seed(args.seed)

    mode = get_run_mode(args)

    print("\n[RL-TCP] Primary Video QoS agent started", flush=True)
    print_run_configuration(args, mode)

    env = make_env()

    try:
        model = create_or_load_model(args, env)

        if args.eval_only:
            run_evaluation(model, env, args.timesteps)
            return

        run_training(model, args, mode)

    finally:
        env.close()


if __name__ == "__main__":
    main()
