import numpy as np
import gymnasium as gym

from control.base_controller import BaseController


class ActionAdapter(BaseController):
    """
    Direct joint controller.

    Modes:
        - incremental=True  → action = delta_q
        - incremental=False → action = absolute q

    rescale_to_limits=True maps an action in [-1, 1] to [lower, upper] joint limits
    (absolute mode only). The limits are provided by Go2Env via set_joint_limits().
    """

    def __init__(self,
                 action_dim=12,
                 kp=50.0,
                 kd=3.0,
                 incremental=False,
                 action_scale=1.0,
                 rescale_to_limits=False,
                 comp_grav=False):

        super().__init__()

        self.action_dim = action_dim
        self.incremental = incremental
        self.action_scale = action_scale
        self.rescale_to_limits = rescale_to_limits
        self.joint_limits = None  # (12, 2), set by Go2Env

        # read by Go2Sim._low_level_control (PD only by default)
        self.comp_grav = comp_grav

        self.Kp_vec = np.ones(action_dim) * kp
        self.Kd_vec = np.ones(action_dim) * kd

    def set_joint_limits(self, joint_limits):
        self.joint_limits = np.asarray(joint_limits)

    # ======================================================
    # ACTION SPACE
    # ======================================================
    def get_action_space(self):

        # Normalmente RL usa [-1, 1]
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

    # ======================================================
    # BEFORE STEP
    # ======================================================
    def before_step(self, state, action):

        if action is None:
            action = np.zeros(self.action_dim)

        action = np.asarray(action).reshape(self.action_dim)

        if self.rescale_to_limits and not self.incremental:
            lower, upper = self.joint_limits[:, 0], self.joint_limits[:, 1]
            action = lower + (np.clip(action, -1.0, 1.0) + 1.0) * 0.5 * (upper - lower)
        else:
            # Escala (útil para RL)
            action = action * self.action_scale

        q_current = state.robot.q  # ⚠️ ajuste se seu nome for diferente

        if self.incremental:
            qr = q_current + action
        else:
            qr = action

        # Atualiza estado (seguindo seu BaseController)
        state.low_level.qr = qr
        state.low_level.Kp = self.Kp_vec
        state.low_level.Kd = self.Kd_vec

    # ======================================================
    def reset(self):
        pass
