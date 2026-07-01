# Service-Aware Hybrid RL-TCP for QoS Protection under Burst Congestion

종합설계 프로젝트에서 사용한 `ns-3` / `ns3-gym` 기반 Hybrid RL-TCP 실험 코드입니다.

프로젝트의 핵심 목표는 여러 flow가 병목 링크를 공유하는 상황에서, TCP Cubic 구조를 그대로 둔 상태에서 RL agent가 CWND를 보조적으로 조절해 `S2 primary video-like flow`의 QoS를 보호할 수 있는지 확인하는 것으로 설정했습니다.

프로젝트를 진행하며 직접 작성하거나 수정한 코드를 중심으로 정리했습니다. 

## 실험 환경

실험은 아래 환경에서 진행했습니다.

```text
Ubuntu 22.04 LTS
ns-3.38
ns3-gym
Python 3
Stable-Baselines3
PyTorch
NumPy
Matplotlib
```

기본 실험 설정은 다음과 같습니다.

```text
RL algorithm       : PPO
TCP baseline       : TCP Cubic
Bottleneck rate    : 15 Mbps
Queue              : DropTail 150p
Control interval   : 0.1 s
Simulation time    : 160 s
Burst interval     : 50 ~ 100 s
S2 QoS 기준        : goodput ≥ 5 Mbps, RTT ≤ 120 ms
```
## Goodput 기준

본 프로젝트에서 goodput은 수신 측 application이 실제로 받은 유효 데이터량을 기준으로 계산했습니다. 각 flow의 `PacketSink`에서 0.1초 동안 수신한 byte 수를 누적하고, 이를 Mbps로 변환했습니다.

```text
goodput (Mbps) = received bytes × 8 / 0.1 / 1e6
```

이처럼, TCP가 보낸 전체 전송량이 아니라 수신 application 기준 useful data rate라고 볼 수 있는 지표입니다. S2 primary video-like flow의 품질을 평가할 때, S2 goodput이 5 Mbps 이상을 만족한 비율을 주요 QoS compliance 지표로 사용했습니다.


## 코드 구조

```text
scratch/
├── analyze_intervals.py
├── plot.py
│
├── qos_rl_tcp/
│   ├── CMakeLists.txt
│   ├── gym_tcp_env.h
│   ├── gym_tcp_env.cc
│   ├── topo3_parkinglot.cc
│   ├── topo3_dumbbell.cc
│   ├── topo3_dumbbell_asym.cc
│   ├── topo3_star_gateway.cc
│   └── topo3_parkinglot_stress.cc
│
├── rl_algorithm/
│   ├── train_agent.py
│   ├── qos_reward_env.py
│   ├── train_agent_ablation.py
│   └── qos_reward_env_ablation.py
│
└── experiment_tools/
    ├── run_parkinglot_validation.py
    └── analyze_parkinglot_validation.py
```

## 파일 설명

### 1. ns3-gym 환경 인터페이스

```text
scratch/qos_rl_tcp/gym_tcp_env.h
scratch/qos_rl_tcp/gym_tcp_env.cc
```

`gym_tcp_env`는 ns-3의 TCP socket 상태를 Python RL agent와 연결하는 역할을 합니다.

현재 실험에서는 S1과 S2만 RL 제어 대상으로 보기에 이 둘을 등록합니다.

```text
S1: FTP/background flow
S2: primary video-like flow
S3: burst cross traffic
```

agent가 받는 observation은 S1과 S2 각각에 대해 다음 정보를 포함합니다.

```text
cwnd, RTT, RTT ratio, received bytes, loss signal, service type
```

S1과 S2가 각각 6개 feature를 가지기에 전체 observation은 12차원입니다.
action은 S1/S2 각각에 대해 CWND [감소, 유지, 증가] 중 하나를 선택하는 방식이며, 두 flow의 action을 합쳐 총 9개의 경우가 발생할 수 있도록 구성했습니다.

### 2. ns-3 topology 코드

```text
scratch/qos_rl_tcp/topo3_parkinglot.cc
scratch/qos_rl_tcp/topo3_dumbbell.cc
scratch/qos_rl_tcp/topo3_star_gateway.cc
scratch/qos_rl_tcp/topo3_parkinglot_stress.cc
scratch/qos_rl_tcp/topo3_dumbbell_asym.cc
```

각 topology의 역할은 다음과 같습니다.

| 파일                           | 역할                                            |
| ---------------------------- | --------------------------------------------- |
| `topo3_parkinglot.cc`        | main parking-lot 실험 topology                  |
| `topo3_dumbbell.cc`          | shared bottleneck 형태의 추가 training topology |
| `topo3_star_gateway.cc`      | 학습에 포함하지 않은 unseen evaluation topology        |
| `topo3_parkinglot_stress.cc` | S4 burst를 추가한 stress 평가 topology              |
| `topo3_dumbbell_asym.cc`     | dumbbell 구조를 기반으로, 노드별 delay 차이를 둔 실험용 training variant          |

<img width="600" height="380" alt="Basic Topology" src="https://github.com/user-attachments/assets/c1b31005-8735-4dee-8b2a-7d01ba723cc2" /> <br /> 
[stress 시나리오를 제외한 여타 topology에서 개입하는 요소들은 위 다이어그램처럼 동작합니다]

<img width="600" height="572" alt="stress_condition" src="https://github.com/user-attachments/assets/202e4eac-baa1-409e-a2bc-0197783a0455" /> <br />
[stress 시나리오에서는 위처럼 하나의 추가 burst traffic이 중간에 끼어들도록 설정했습니다]

`topo3_dumbbell_asym.cc`는 학습 topology를 다양화해보는 과정에서 시도한 코드입니다. 다만, 최종 성능 개선에는 도움이 되지 않아 최종 결과물에는 포함하지 않았습니다.

각 topology는 두 가지 mode에 대해서 돌아가게끔 설정했습니다.

```text
Baseline : TCP Cubic만 실행
RL       : ns3-gym을 통해 Python PPO agent와 연결
```

실험이 끝나면 goodput, RTT, CWND 로그가 `scratch/results/` 하위 폴더에 저장됩니다.

### 3. RL agent 및 reward wrapper

```text
scratch/rl_algorithm/train_agent.py
scratch/rl_algorithm/qos_reward_env.py
```

`train_agent.py`는 Stable-Baselines3의 PPO를 이용해 학습과 저장된 model을 불러와 평가하는 코드입니다.
`qos_reward_env.py`는 ns3-gym raw environment를 감싸서 reward 계산과 action projection을 수행합니다.

전체 제어 흐름은 아래 다이어그램처럼 이루어집니다.
<img width="5394" height="2090" alt="flow_diagram" src="https://github.com/user-attachments/assets/2fa18339-c32f-434e-b4b2-4fbaeebcc5ac" />

흐름을 따라가며 짚어보면, <br />
(1) ns-3 simulation은 GymTcpEnv를 통해 현재 TCP 상태를 Python 쪽으로 넘겨줍니다. <br />
(2) Python agent는 이 observation을 바탕으로 action을 선택하고, <br />
(3) reward wrapper는 선택된 action이 S2 QoS를 심하게 해치지 않도록 필요한 경우에 보정합니다. <br />
(4) 이후 최종 action이 다시 ns-3으로 전달되고, S1/S2의 CWND에 적용됩니다. <br />

종합적으로, 실험 전반에서 사용한 Full RL 구조는 아래와 같습니다.

```text
PPO policy + service-aware reward shaping + action projection
```

reward는 S2 primary video-like flow가 5 Mbps goodput 기준을 가능한 만족하도록 설계했습니다. 동시에, S2를 무조건 보호하기만 하 S1 FTP flow가 아예 끊길 수 있기에 S1이 최소한의 전송 성능을 유지하도록 penalty도 함께 두었습니다.

action projection은 PPO가 선택한 action을 적용하기 전에 보정이 들어가는 부분입니다. 예를 들어, S2 goodput이 5 Mbps 아래로 떨어진 상황에서는 S1이 계속 CWND를 늘리는 action을 제한하거나, S2의 CWND 감소 action을 막는 식으로 동작합니다.

### 4. Ablation 코드

```text
scratch/rl_algorithm/train_agent_ablation.py
scratch/rl_algorithm/qos_reward_env_ablation.py
```

ablation에서는 reward shaping은 그대로 두고 action projection을 제거해봤습니다.

비교 구조는 다음과 같습니다.

```text
Full RL       : PPO policy + reward shaping + action projection
No-Projection : PPO policy + reward shaping
```

이런 비교를 통해 성능 개선이 reward에서만 비롯된 것인지, action projection도 실제로 도움이 되었는지 나눠 확인했습니다.

### 5. Validation 및 분석 코드

```text
scratch/experiment_tools/run_parkinglot_validation.py
scratch/experiment_tools/analyze_parkinglot_validation.py
scratch/analyze_intervals.py
scratch/plot.py
```

`run_parkinglot_validation.py`는 기본 토폴로지+시나리오인 parking-lot single-burst 환경에서 Baseline, Full RL, No-Projection ablation을 seed별로 반복 실행하는 코드입니다.

`analyze_parkinglot_validation.py`는 위 실험 결과를 모아 multi-seed 평균과 ablation 결과를 출력합니다.

`analyze_intervals.py`는 특정 시간 구간만 잘라서 goodput, RTT, QoS ratio 등을 계산하고 터미널에 출력하는 용도입니다.

`plot.py`는 S2 goodput time-series, S2 RTT CDF, QoS compliance bar plot을 생성합니다.

## 파일 배치

ns-3.38 최상위 폴더에 맞춰 배치하는 것을 전제로 파일들을 구성했습니다. 

ns-3 경로가 아래와 같다면,

```text
~/ns-allinone-3.38/ns-3.38/
```

해당 repository의 `scratch/` 폴더를 ns-3.38 내부에 위치시키면 됩니다.

최종적으로는 아래와 같은 구조가 됩니다.

```text
ns-3.38/
├── ns3
├── scratch/
│   ├── qos_rl_tcp/
│   ├── rl_algorithm/
│   ├── experiment_tools/
│   ├── analyze_intervals.py
│   └── plot.py
```

이후 ns-3 최상위 폴더에서 빌드합니다. (터미널 활용)

```bash
./ns3 build
```

## 실행 방법

### Baseline 실행

주축이 되는 parking-lot topology에서 TCP Cubic baseline을 실행하려면 다음과 같이 터미널에 입력하면 됩니다.

```bash
./ns3 run "scratch/qos_rl_tcp/topo3_parkinglot --mode=Baseline"
```

별도의 명시가 없으면 결과는 아래 경로에 저장됩니다.

```text
scratch/results/topo3_rtt/
```

### RL 학습 실행

RL mode는 ns-3 simulation과 Python agent를 서로 다른 터미널에서 실행하는 방식입니다.

먼저 Terminal 1에서 ns-3 simulation을 실행합니다.

```bash
./ns3 run "scratch/qos_rl_tcp/topo3_parkinglot --mode=RL"
```

이후, Terminal 2에서 PPO agent를 실행합니다.

```bash
python3 scratch/rl_algorithm/train_agent.py --seed=1 --timesteps=1536
```

기본 model 저장 경로는 아래와 같습니다.

```text
scratch/rl_algorithm/qos_tcp_stable_model.zip
```

### 이어서 학습하기

기존 model을 불러와서 이어서 학습시킬 수 있습니다. 

```bash
python3 scratch/rl_algorithm/train_agent.py \
  --continue-train \
  --model-path scratch/rl_algorithm/qos_tcp_stable_model \
  --save-path scratch/rl_algorithm/qos_tcp_stable_model \
  --seed=1 \
  --timesteps=1536
```

multi-topology sequential training 시에는 parking-lot에서 학습한 model을 저장하고, 이후 dumbbell topology에서 같은 model을 이어서 학습하는 방식으로 사용했습니다.

### Eval-only 실행

학습된 model을 고정한 상태로 평가만 수행하는 방법입니다. (Unseen topology 검증에 활용)

```bash
python3 scratch/rl_algorithm/train_agent.py \
  --eval-only \
  --model-path scratch/rl_algorithm/qos_tcp_stable_model \
  --seed=1 \
  --timesteps=1536
```

eval-only에서는 PPO model이 업데이트되지 않습니다.
다만 `qos_reward_env.py`의 action projection은 그대로 적용되기에, 최종 결과는 fixed PPO policy와 action projection이 함께 적용된 Hybrid RL-TCP 결과입니다.

### No-Projection ablation 실행

No-Projection ablation도 ns-3 simulation을 먼저 실행한 뒤 Python agent를 연결합니다. (Projection 유무에 따른 성능 비교에 활용)

Terminal 1:

```bash
./ns3 run "scratch/qos_rl_tcp/topo3_parkinglot --mode=RL"
```

Terminal 2:

```bash
python3 scratch/rl_algorithm/train_agent_ablation.py \
  --seed=1 \
  --timesteps=1536
```

## Parking-lot validation 실행

main으로 보고 있는 parking-lot single-burst 환경에서 Baseline, Full RL, No-Projection을 seed별로 반복 실행하기 위해서는 아래와 같은 스크립트를 터미널에 입력하면 됩니다.

```bash
python3 scratch/experiment_tools/run_parkinglot_validation.py --task both --n-seeds 5
```

결과는 아래 경로에 정리됩니다.

```text
scratch/results/parkinglot_validation/
```

분석은 다음과 같은 코드를 통해 실행시킬 수 있습니다.

```bash
python3 scratch/experiment_tools/analyze_parkinglot_validation.py \
  --data-dir scratch/results/parkinglot_validation \
  --report all \
  --n-seeds 5
```

## 결과 분석 스크립트

그래프가 필요하지 않고, 특정 구간(burst traffic)만 분석하기 위한 경우 `analyze_intervals.py`를 사용합니다.

지금 실험에서 발생하는 기본 burst 구간인 50~100초를 분석하려면 다음과 같이 실행합니다.

```bash
python3 scratch/analyze_intervals.py \
  --data-dir scratch/results/topo3_rtt \
  --scenario burst
```

필요하다면 직접 구간을 지정할 수도 있습니다.

```bash
python3 scratch/analyze_intervals.py \
  --data-dir scratch/results/parkinglot_stress_rtt \
  --start 70 \
  --end 100
```

그래프는 아래와 같은 명령어로 생성합니다.

```bash
python3 scratch/plot.py --data-dir scratch/results/topo3_rtt
```

기본 출력 파일은 다음 경로에 저장됩니다.

```text
scratch/results/topo3_rtt/topo3_stable_report.png
```

## 결과 요약

Parking-lot, single-burst 환경에서 S2 flow의 goodput, RTT, QoS compliance를 비교해본 결과 다음과 같은 출력 그래프를 확인할 수 있었습니다. <br /> <br />
<img width="4800" height="3000" alt="parkinglot" src="https://github.com/user-attachments/assets/75b4d9fa-ec92-413a-b582-f8976fa81ff5" />

최종 보고서에서는 위와 같은 단일 실행 결과에서 나아가, 5개 seed 평균을 중심으로 결과를 정리했습니다.

위와 동일한 환경에서 제안된 모델(Full RL)은 S2 burst 구간의 5 Mbps 이상 goodput compliance를 TCP Cubic baseline 대비 개선했습니다.

```text
Baseline Cubic : 18.2%
Full RL        : 32.2% ± 6.8%
Improvement    : +14.1%p
```

ablation 결과에서는 reward shaping만 사용한 No-Projection도 일정 수준 개선을 보였고, action projection을 추가한 Full RL에서는 더 높은 개선이 나타났습니다.

```text
Baseline Cubic : 18.2%
No-Projection  : 27.6% ± 3.6%
Full RL        : 32.2% ± 6.8%
```

parking-lot과 dumbbell에서 순차 학습한 general model은 학습에 사용하지 않은 star-like shared gateway topology에서도 S2 burst compliance를 개선했습니다.

```text
Star-like gateway unseen evaluation
Baseline Cubic : 45.5%
Full RL        : 71.7%
Improvement    : +26.1%p
```

상당한 개선이 이루어진 것으로 보이지만, 모든 flow의 성능을 동시에 높인 결과는 아닙니다.
S2 primary video-like flow를 보호하는 대신 S1 FTP/background flow의 goodput이 일부 감소하는 trade-off가 있었습니다. 그렇기에 전체 처리량이 개선되었다고 보기보다, RTT 120 ms constraint를 크게 벗어나지 않는 범위에서 S2 QoS를 우선 보호하는 service-aware control 방식이라고 이해해주시면 감사하겠습니다.

stress 평가에서는 S3 burst에 S4 burst가 추가로 겹치는 조건을 두었습니다. 이 경우에도 Full RL이 S2 degradation을 일부 줄이는데는 성공했지만, 5 Mbps 기준을 안정적으로 만족시키는 수준까지는 이르지 못했습니다.

```text
Parking-lot stress
Baseline Cubic : 13.2%
Full RL        : 21.2%
Improvement    : +8.0%p
```

이처럼, 설정한 조건보다 많고 다양한 bursty traffic이 발생하는 환경에서는 지금 제시된 구조만으로는 안정적인 QoS 보장을 하기는 어려움이 있었습니다. 이후 burst traffic 자체를 예측하거나 제어하는 방식까지 함께 고려한다면 더욱 개선이 가능할 것이라고 기대됩니다.

## Output log 형식

simulation을 실행하면 각 mode별로 다음과 같은 로그가 생성됩니다.

```text
baseline_goodput_s1.txt
baseline_goodput_s2.txt
baseline_rtt_s1.txt
baseline_rtt_s2.txt
baseline_cwnd_s1.txt
baseline_cwnd_s2.txt

rl_goodput_s1.txt
rl_goodput_s2.txt
rl_rtt_s1.txt
rl_rtt_s2.txt
rl_cwnd_s1.txt
rl_cwnd_s2.txt
```

각 파일은 기본적으로 시간에 따라 각 value를 기록하는 형식의 text log입니다.

분석 스크립트들은 각각 대응하는 로그 파일을 읽어 goodput, RTT, QoS compliance, burst 구간 성능 등을 계산합니다.

## 참고 사항

최종 보고서에 대응하는 모든 내용을 기록 및 정리하기보다, 실험에 사용된 코드 구조와 실행 방법을 기반으로 정리했습니다.
실험 배경, 결과표, 해석 등은 보고서를 참고해주시면 감사하겠습니다.
