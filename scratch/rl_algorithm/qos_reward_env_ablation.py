"""
qos_reward_env_ablation.py

No-Projection ablation wrapper.

Full RL의 reward shaping은 그대로 사용하고, action projection만 제거합니다.

Full RL: PPO policy + reward shaping + action projection
No-Projection: PPO policy + reward shaping
"""

from qos_reward_env import QosRewardWrapper


class QosRewardWrapperNoProjection(QosRewardWrapper):
    """
    Action projection을 제거한 ablation wrapper.
    QosRewardWrapper의 step(), reward 계산, EWMA goodput, S1/S2 상태 갱신, action shaping reward는 그대로 받아옵니다.
    여기서 바꾸는 건 _project_action() 만입니다.
    """

    def __init__(self, env):
        super().__init__(env)

        print(
            "[NO-PROJ] Action projection disabled. "
            "Reward shaping is inherited from QosRewardWrapper.",
            flush=True,
        )

    def _project_action(self, raw_action_id):
        """
        Full RL에서는 S2 QoS 상태에 따라 PPO raw action을 보정합니다.
        No-Projection ablation에서는 PPO raw action을 그대로 사용합니다.
        Return되는 tuple 형식은 원본 QosRewardWrapper.step()과 호환되도록 유지합니다.
        """

        raw_action_id = int(raw_action_id)

        raw_s1_action, raw_s2_action = self._decode_action(raw_action_id)

        projected_action_id = raw_action_id
        s1_action = raw_s1_action
        s2_action = raw_s2_action
        projected = False

        return (
            projected_action_id,
            raw_s1_action,
            raw_s2_action,
            s1_action,
            s2_action,
            projected,
        )
