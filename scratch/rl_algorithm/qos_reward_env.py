import gym
import numpy as np


# -----------------------------------------------------------------------------
# QoS reward wrapper 정의
# ns3-gym raw environment를 감싸서 service-aware reward shaping과 action projection을 적용합니다. 
# PPO policy가 선택한 raw action은 이 wrapper를 거친 뒤 ns-3 TCP CWND 제어 action으로 전달됩니다.
# -----------------------------------------------------------------------------

class QosRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.count = 0
        self.prev_action = None

        # ns-3 topology의 GymStep 주기와 맞춰야 합니다.
        # 현재 실험에서는 0.1초마다 observation/action interaction이 발생합니다.
        self.CONTROL_INTERVAL = 0.1

        # ---------------------------------------------------------------------
        # S2 primary video-like flow QoS 기준
        # 여기서는 보호 대상인 S2 flow의 최소 goodput, 목표 goodput, deep-drop 기준, RTT constraint를 정의합니다.
        # ---------------------------------------------------------------------

        self.MIN_GP = 5.0 * 1e6        # 5 Mbps: 1080p-like minimum QoS threshold
        self.TARGET_GP = 7.0 * 1e6     # 7 Mbps: preferred target goodput
        self.DEEP_DROP_GP = 4.5 * 1e6  # 4.5 Mbps: severe goodput drop guard
        self.MAX_RTT = 120.0           # 120 ms: RTT QoS constraint

        # ---------------------------------------------------------------------
        # S1 FTP/background flow 보호 기준
        # 여기서는 S2가 위험할 때 S1이 일부 양보하도록 유도하되, S1 FTP가 완전히 starvation되지 않도록 최소 기준을 둡니다.
        # ---------------------------------------------------------------------

        self.MIN_FTP_GP = 1.0 * 1e6
        self.FTP_PROJECTION_FLOOR = 2.5 * 1e6
        self.FTP_CONCESSION_TARGET = 3.0 * 1e6

        # 0.1초 단위 instant goodput은 변동성이 크기에 EWMA (Exponentially weighted average로 완화합니다.
        # alpha=0.3은 burst 변화에는 반응하면서도 순간 spike 영향을 줄이기 위한 값입니다.
        self.EWMA_ALPHA = 0.3

        self.s1_gp_ewma = None
        self.s2_gp_ewma = None

        # action projection은 직전 step의 S1/S2 상태를 바탕으로 수행됩니다.
        self.prev_s1_gp = None
        self.prev_s2_gp = None
        self.prev_s2_rtt = None


    # -----------------------------------------------------------------------------
    # Episode 초기화
    # 여기서는 episode 시작 시 누적 상태, EWMA goodput, 직전 상태 정보를 초기화합니다.
    # Gym 버전 차이로 reset(**kwargs)가 실패하는 경우도 처리합니다.
    # -----------------------------------------------------------------------------

    def reset(self, **kwargs):
        self.count = 0
        self.prev_action = None

        self.s1_gp_ewma = None
        self.s2_gp_ewma = None

        self.prev_s1_gp = None
        self.prev_s2_gp = None
        self.prev_s2_rtt = None

        try:
            return self.env.reset(**kwargs)
        except TypeError:
            return self.env.reset()


    # -----------------------------------------------------------------------------
    # EWMA goodput 갱신
    # 여기서는 현재 step에서 관측된 raw goodput을 EWMA로 smoothing합니다.
    # reward와 projection은 raw instant goodput 대신 EWMA goodput을 기준으로 계산됩니다.
    # -----------------------------------------------------------------------------

    def _update_ewma(self, prev_value, current_value):
        if prev_value is None:
            return current_value

        return (1.0 - self.EWMA_ALPHA) * prev_value + self.EWMA_ALPHA * current_value


    # -----------------------------------------------------------------------------
    # Joint action encoding / decoding
    # 여기서는 9개 discrete action을 S1/S2 각각의 3개 action으로 분해하거나
    # 다시 하나의 action id로 결합합니다.
    #
    # action value:
    #   0 = decrease CWND
    #   1 = hold CWND
    #   2 = increase CWND
    #
    # joint action:
    #   action_id = s1_action + 3 * s2_action
    # -----------------------------------------------------------------------------

    def _decode_action(self, action_id):
        s1_action = action_id % 3
        s2_action = (action_id // 3) % 3
        return s1_action, s2_action

    def _encode_action(self, s1_action, s2_action):
        return int(s1_action + 3 * s2_action)


    # -----------------------------------------------------------------------------
    # Constraint-aware action projection
    # 여기서는 PPO policy가 선택한 raw action을 QoS 제약에 맞게 보정합니다.
    #
    # 목적:
    # - S2 primary video-like flow가 5Mbps 미만이면 S1 FTP가 먼저 양보하도록 유도
    # - S2가 위험한 상태에서 S2 CWND를 줄이는 action 방지
    # - S2가 4.5Mbps 미만으로 깊게 떨어지면 S2 increase action 우선
    # - 단, S1 FTP가 이미 낮으면 S1 추가 감소를 강제하지 않음
    #
    # 따라서 eval-only에서도 결과는 "fixed PPO policy 단독"이 아니라
    # "fixed PPO policy + action projection"이 적용된 Hybrid RL-TCP 결과입니다.
    # -----------------------------------------------------------------------------

    def _project_action(self, raw_action_id):
        raw_s1_action, raw_s2_action = self._decode_action(raw_action_id)

        s1_action = raw_s1_action
        s2_action = raw_s2_action
        projected = False

        if self.prev_s2_gp is not None:
            prev_s2_unsafe = self.prev_s2_gp < self.MIN_GP
            prev_s2_deep_drop = self.prev_s2_gp < self.DEEP_DROP_GP

            if prev_s2_unsafe:
                if self.prev_s1_gp is not None:
                    if self.prev_s1_gp > self.FTP_PROJECTION_FLOOR:
                        if s1_action != 0:
                            s1_action = 0
                            projected = True

                    elif self.prev_s1_gp < self.MIN_FTP_GP * 1.3:
                        if s1_action == 0:
                            s1_action = 1
                            projected = True

                if s2_action == 0:
                    s2_action = 1
                    projected = True

                if prev_s2_deep_drop and s2_action != 2:
                    s2_action = 2
                    projected = True

            # S2가 이미 target보다 충분히 높고 RTT도 안정적이면 과도한 증가를 억제합니다.
            if (
                self.prev_s2_gp > self.TARGET_GP + 0.8 * 1e6
                and self.prev_s2_rtt is not None
                and self.prev_s2_rtt <= self.MAX_RTT
                and s2_action == 2
            ):
                s2_action = 1
                projected = True

        projected_action_id = self._encode_action(s1_action, s2_action)

        return (
            projected_action_id,
            raw_s1_action,
            raw_s2_action,
            s1_action,
            s2_action,
            projected,
        )


    # -----------------------------------------------------------------------------
    # Environment step 및 reward 계산
    # 여기서는 PPO action을 projection한 뒤 ns-3 environment에 전달하고, 반환된 observation을 바탕으로 service-aware reward를 계산합니다.
    #
    # observation layout:
    #   S1: [0]cwnd [1]rtt [2]rttRatio [3]bytes [4]loss [5]serviceType
    #   S2: [6]cwnd [7]rtt [8]rttRatio [9]bytes [10]loss [11]serviceType
    #
    # reward 설계 방향:
    # - S2 primary video-like flow의 5Mbps/120ms QoS compliance 우선
    # - S2가 안전하면 S1 FTP 활용도 허용
    # - S2가 위험하면 S1 FTP가 일부 양보하도록 유도
    # - loss와 과도한 action oscillation은 penalty 부여
    # -----------------------------------------------------------------------------

    def step(self, action):
        raw_action_id = int(np.asarray(action).item())

        (
            action_id,
            raw_s1_action,
            raw_s2_action,
            s1_action,
            s2_action,
            projected,
        ) = self._project_action(raw_action_id)

        obs, _, done, info = self.env.step(action_id)
        obs = np.asarray(obs, dtype=np.float32)

        if obs.size < 12:
            raise ValueError(
                f"Observation size mismatch: expected 12 values, got {obs.size}"
            )

        s1_gp_raw = float(obs[3]) * 8.0 / self.CONTROL_INTERVAL
        s2_gp_raw = float(obs[9]) * 8.0 / self.CONTROL_INTERVAL
        s2_rtt = float(obs[7])

        self.s1_gp_ewma = self._update_ewma(self.s1_gp_ewma, s1_gp_raw)
        self.s2_gp_ewma = self._update_ewma(self.s2_gp_ewma, s2_gp_raw)

        s1_gp = self.s1_gp_ewma
        s2_gp = self.s2_gp_ewma

        s1_mbps = s1_gp / 1e6
        s2_mbps = s2_gp / 1e6

        common_loss_count = max(float(obs[4]), float(obs[10]))

        min_safe = (
            s2_gp >= self.MIN_GP
            and s2_rtt <= self.MAX_RTT
        )

        target_safe = (
            s2_gp >= self.TARGET_GP
            and s2_rtt <= self.MAX_RTT
        )

        reward = 0.0


        # -------------------------------------------------------------------------
        # 1. S2 primary video goodput reward
        # S2 goodput이 높을수록 보상을 주고, 5Mbps 최소 기준 및 7Mbps target 기준을 만족하면 추가 보상을 부여합니다.
        # -------------------------------------------------------------------------

        reward += min(s2_mbps, 7.0) * 65.0

        if min_safe:
            reward += 380.0

        if target_safe:
            reward += 220.0


        # -------------------------------------------------------------------------
        # 2. S2 goodput deficit penalty
        # S2가 5Mbps 미만으로 내려가면 큰 penalty를 부여하고, 4.5Mbps 미만의 deep drop에 대해서는 추가 penalty를 부여합니다.
        # 5Mbps 이상이지만 7Mbps target에는 부족한 경우에는 약한 penalty만 둡니다.
        # -------------------------------------------------------------------------

        min_deficit_mbps = max(0.0, (self.MIN_GP - s2_gp) / 1e6)
        target_deficit_mbps = max(0.0, (self.TARGET_GP - s2_gp) / 1e6)

        if min_deficit_mbps > 0.0:
            reward -= 950.0
            reward -= min_deficit_mbps * 800.0

            deep_drop_mbps = max(0.0, (self.DEEP_DROP_GP - s2_gp) / 1e6)
            reward -= deep_drop_mbps * 650.0

        else:
            reward -= target_deficit_mbps * 75.0


        # -------------------------------------------------------------------------
        # 3. S2 RTT penalty
        # 여기서는 S2 RTT가 120ms를 초과할 때만 penalty를 부여합니다.
        # 현재 돌려본 결과에서는 RTT를 직접 줄이는 것보다, 120ms delay constraint를 유지하는 범위 내에서 S2 goodput QoS를 개선하는 구조로 보는게 좋을 것 같습니다.
        # -------------------------------------------------------------------------

        if s2_rtt > self.MAX_RTT:
            rtt_excess = s2_rtt - self.MAX_RTT
            reward -= 1000.0
            reward -= rtt_excess * 10.0


        # -------------------------------------------------------------------------
        # 4. S1 FTP utilization and concession
        # 여기서는 S1 FTP가 완전히 굶지 않도록 최소 goodput을 보호하면서, S2가 unsafe한 경우에는 S1이 일정 수준 이상 자원을 가져가는 것을 양보 부족으로 보고 penalty를 부여합니다.
        # -------------------------------------------------------------------------

        ftp_deficit_mbps = max(0.0, (self.MIN_FTP_GP - s1_gp) / 1e6)

        if ftp_deficit_mbps > 0.0:
            reward -= 300.0
            reward -= ftp_deficit_mbps * 300.0

        if target_safe:
            reward += min(s1_mbps, 6.0) * 18.0

            if self.prev_action is not None and action_id != self.prev_action:
                reward -= 40.0

        elif min_safe:
            reward += min(s1_mbps, 3.0) * 6.0

            if self.prev_action is not None and action_id != self.prev_action:
                reward -= 20.0

        else:
            ftp_excess_mbps = max(
                0.0,
                (s1_gp - self.FTP_CONCESSION_TARGET) / 1e6
            )
            reward -= ftp_excess_mbps * 230.0


        # -------------------------------------------------------------------------
        # 5. Action shaping
        # 여기서는 현재 S2 QoS 상태에 맞지 않는 action에 penalty를 주고, S2가 unsafe할 때 S1 decrease 또는 S2 increase와 같은 유리한 방향의 action에는 보상을 부여합니다.
        # -------------------------------------------------------------------------

        if not min_safe:
            if s1_action == 2:
                reward -= 180.0

            if s2_action == 0:
                reward -= 260.0

            if s2_action == 2:
                reward += 90.0

            if s1_action == 0:
                reward += 40.0

        if target_safe:
            if s2_action == 2:
                reward -= 35.0


        # -------------------------------------------------------------------------
        # 6. Loss penalty
        # 여기서는 S1/S2 중 더 큰 loss count를 공통 혼잡 신호로 보고 penalty를 부여합니다. 과도한 loss count 영향은 3으로 cap을 씌워 제한합니다.
        # -------------------------------------------------------------------------

        capped_loss = min(common_loss_count, 3.0)
        reward -= capped_loss * 300.0


        # -------------------------------------------------------------------------
        # 다음 step을 위한 상태 저장 및 주기적 로그 출력
        # projection과 reward 계산에 필요한 현재 상태를 저장하고, 20 step마다 action projection 여부, S1/S2 goodput, RTT, loss, reward를 출력해 학습/평가 진행 상태를 확인합니다.
        # -------------------------------------------------------------------------

        self.prev_s1_gp = s1_gp
        self.prev_s2_gp = s2_gp
        self.prev_s2_rtt = s2_rtt

        self.prev_action = action_id
        self.count += 1

        if self.count % 20 == 0:
            status = "TARGET_SAFE" if target_safe else ("MIN_SAFE" if min_safe else "ADJUSTING")
            projection_text = "Y" if projected else "N"

            print(
                f"[{self.count}] "
                f"RAW_A:{raw_action_id}(S1:{raw_s1_action},S2:{raw_s2_action}) | "
                f"ACT_A:{action_id}(S1:{s1_action},S2:{s2_action}) | "
                f"PROJ:{projection_text} | "
                f"S1:{s1_mbps:.2f}Mbps | "
                f"S2_RAW:{s2_gp_raw / 1e6:.2f}Mbps | "
                f"S2_EWMA:{s2_mbps:.2f}Mbps | "
                f"S2_RTT:{s2_rtt:.1f}ms | "
                f"LOSS:{common_loss_count:.0f} | "
                f"RWD:{reward:.1f} | "
                f"State:{status}",
                flush=True,
            )

        return obs, reward, done, info