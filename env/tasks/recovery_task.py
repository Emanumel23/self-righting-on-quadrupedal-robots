import time
import numpy as np
from env.tasks.base_task import BaseTask
from sim.utils.orientation import gravity_body


class RecoveryTask(BaseTask):
    """
    Self-recovery task from "Self-Recovery of Quadrupedal Robot Using Deep RL" (PPO, joint position actions).

    Observation (30): q (12) + dq (12) + gravity in body frame (3) + base angular velocity (3)
    Reward: Eq. (9) of the paper, weights from Table III, every term mapped to [0, 1].
    Termination: only the step limit (no early termination), as in the paper.
    """

    # Table III

    #Colocar random para os pesos no intervalor de [-1,1]
    W1, W2, W3, W4, W5, W6, W7, W8 = 0.3, 0.3, 0.3, 0.6, 0.1, 0.05, 0.005, 0.05

    # standing joint configuration (keyframe "home" in go2.xml)
    Q_STAND = np.array([0.0, 0.9, -1.8] * 4)

    def __init__(self, normalizer, tpe=None, h_target=0.27, noise=False, noise_std=(0.1, 1.0, 0.2)):
        super().__init__(step_limit=1600, normalizer=normalizer)

        self.tpe = tpe  # unused, kept so create_env can build every task the same way

        self.obs_dim = 30
        self.h_target = h_target  # height (m) above which R_h = 1; paper: 0.31 for the A1 (0.34 max)
        self.noise = noise
        self.noise_std = noise_std  # (q [rad], dq [rad/s], omega [rad/s])

        self._last_qr_norm = None

    # ------------------------------------------------------
    def get_obs_dim(self):
        return self.obs_dim

    def set_step_limit(self, step_limit):
        self.step_limit = step_limit

    def reset(self):
        self._last_qr_norm = None
        return super().reset()

    def compute_initial_obs(self, state):
        self.compute_features(state)
        return self.get_obs()

    # ------------------------------------------------------
    def compute_features(self, state):

        self.state_copy = state
        rs = state.robot
        cs = state.low_level

        q, dq, omega_b = rs.q, rs.dq, rs.omega_b

        if self.noise:
            q = q + np.random.uniform(-1, 1, 12) * self.noise_std[0]
            dq = dq + np.random.uniform(-1, 1, 12) * self.noise_std[1]
            omega_b = omega_b + np.random.uniform(-1, 1, 3) * self.noise_std[2]

        self._features = {
            "q_norm": self.normalizer.normalize_q(q),
            "dq_norm": self.normalizer.normalize_dq(dq),
            "g_body": gravity_body(rs.epsilon),
            "omega_norm": np.tanh(omega_b / (self.normalizer.omega_scale + self.normalizer.eps)),
            "qr_norm": self.normalizer.normalize_q(cs.qr),
        }

    def get_obs(self):
        f = self._features
        obs = np.concatenate([f["q_norm"], f["dq_norm"], f["g_body"], f["omega_norm"]])
        return obs.astype(np.float32)

    # ------------------------------------------------------
    def evaluate_reward(self):
        f = self._features
        rs = self.state_copy.robot
        
        h_altura = rs.b_pos[2] 
        # height: [0, 1), saturates at h_target
        r_h = float(np.clip(h_altura /self.h_target , 0.0, 1.0))

        # orientation: paper error ||[0,0,-1] - theta_B|| in [0, 2] -> reward in [0, 1]
        r_g = 1.0 - float(np.linalg.norm(np.array([0.0, 0.0, -1.0]) - f["g_body"])) / 2.0

        # joint velocity: mean(1 - dq^2) with tanh-normalized dq
        r_v = float(np.mean(1.0 - f["dq_norm"]**2))

        # joint position close to the standing pose (radians, clipped to [0, 1])
        r_jp = float(np.mean(1.0 - np.clip((rs.q - self.Q_STAND)**2, 0.0, 1.0)))

        # action difference (actions are the normalized joint targets, max diff = 2)
        qr_norm = f["qr_norm"]
        if self._last_qr_norm is None:
            r_ad = 1.0
        else:
            r_ad = float(np.mean(1.0 - ((qr_norm - self._last_qr_norm) / 2.0)**2))
        self._last_qr_norm = qr_norm

        # base linear velocity (target 0): RBF, Eq. (8)
        r_vb = float(np.exp(-2.0 * np.sum(rs.b_vel**2)))

        # foot contact: 0.25 per foot in contact
        r_fc = 0.25 * float(np.sum(rs.foot_touching))

        # Eq. (9) as printed (R_h appears in both the w1 term and the gated w3 term)
        cl = 1.0 if r_g > 0.7 else 0.0

        r = (self.W1 * r_h + self.W2 * r_g + cl * (self.W3 * r_h + self.W4 * r_jp + self.W5 * r_fc) + self.W6 * r_ad +
             self.W7 * r_v + self.W8 * r_vb)

        return float(r), h_altura

    # ------------------------------------------------------
    def check_termination(self, current_step, clock, foot_values):
        rs = self.state_copy.robot
        centro = (foot_values.rs.foot_contacts[0, :] + foot_values.rs.foot_contacts[0, :] + foot_values.rs.foot_contacts[0, :] + foot_values.rs.foot_contacts[0, :])/4
        
        altura_robo = rs.b_pos[2] - centro[2]
        
        
        print(f"altura = {altura_robo}")
        terminate = False
        if(altura_robo >= 0.24 and altura_robo <= 0.28):
        #if ((rs.b_pos[2] > 0.27 and rs.b_pos[2] < 0.29) or (rs.b_pos[2] > 0.29 and rs.b_pos[2] < 0.32)):
            if clock.time_started == False:
                clock.time_started = True 
                clock.start_time = time.time()
            current_time = time.time()
            clock.duracao = current_time - clock.start_time 
            #print(f"tempo ={clock.duracao}")
            if (clock.duracao > 2.0):
                clock.time_started = False
                clock.start_time = 0.0
                clock.duracao = 0.0
                terminate = True
        else:
            clock.time_started = False
            clock.start_time = 0.0
            clock.duracao = 0.0
            terminate = False
        

           
        return  terminate, current_step >= self.step_limit

    def gen_info(self, env_id=None, step_now=0):
        return {"env_id": env_id, "sim_step": step_now}
