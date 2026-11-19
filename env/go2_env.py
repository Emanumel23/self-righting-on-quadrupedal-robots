import gymnasium as gym
import numpy as np
from sim.go2_sim import Go2Sim
from sim.engine.pinocchio_engine import PinocchioEngine
import pinocchio as pin


class Go2Env(gym.Env):

    def __init__(self, env_id=None, max_step=2000, shared_state=None, **kwargs):

        super().__init__()
        self.estado = shared_state

        controller = kwargs.get("controller", None)
        task = kwargs.get("task", None)

        if controller is None:
            raise ValueError("Controller must be provided to Go2Env")

        if task is None:
            raise ValueError("Task must be provided to Go2Env")

        self.controller = controller
        self.task = task
        self.viewer = kwargs.get("viewer", None)
        self.scenario = kwargs.get("scenario", None)

        urdf_path = kwargs.get("urdf_path")

        root_joint = pin.JointModelFreeFlyer()
        self.pin_model = pin.buildModelFromUrdf(urdf_path, root_joint)
        self.pin_engine = PinocchioEngine(self.pin_model)

        # --------------------------------------
        # SIMULATION
        # --------------------------------------
        self.sim = Go2Sim(
            mj_model=kwargs.get("mj_model"),
            mj_data=kwargs.get("mj_data"),
            controller=self.controller,
            pin_engine=self.pin_engine,
            con_dt=kwargs.get("con_dt", 0.01),
            dyn_dt=kwargs.get("dyn_dt", 0.001),
            viewer=self.viewer,
        )

        self.task.normalizer.set_robot_limits(joint_limits=self.sim.joint_limits, torque_limits=self.sim.torque_limits)

        # controllers with discrete modes (self-righting schedulers) expose get_num_modes()
        if hasattr(self.controller, "get_num_modes"):
            self.task.normalizer.set_n_actions(self.controller.get_num_modes())

        # direct joint controllers (PPO) need the joint limits to rescale [-1, 1] actions
        if hasattr(self.controller, "set_joint_limits"):
            self.controller.set_joint_limits(self.sim.joint_limits)

        # --------------------------------------
        # ENV STATE
        # --------------------------------------
        self.max_step = max_step
        self.current_step = 0
        self.ep_reward = 0.0

        # --------------------------------------
        # SPACES
        # --------------------------------------
        self.action_space = self.controller.get_action_space()

        obs_dim = self.task.get_obs_dim()

        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        # --------------------------------------
        # META
        # --------------------------------------
        self.env_id = env_id

    def step(self, action, clock):

        self.current_step += 1

        self.sim.simulation_loop(action)

        state = self.sim.state
        self._attach_task_state(state)

        # --------------------------------------
        # Task block
        # --------------------------------------

        self.task.compute_features(state)

        obs = self.task.get_obs()

        reward, height = self.task.evaluate_reward()

        terminated, truncated = self.task.check_termination(self.current_step, clock, self.sim)
        
        info = self.task.gen_info(env_id=self.env_id, step_now=self.current_step)

        if info.get("success_flag") is True:
            reward += info["sucess_extra_reward"]

        # --------------------------------------

        self.ep_reward += reward
    
      
        self.estado["tempo"].append(self.current_step)
        self.estado["altura"].append(height)
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # SB3 calls reset() without options -> sample the initial pose from the scenario
        if not options and self.scenario is not None:
            options = self.scenario.sample()["options"]

        options = options or {}
        q0 = options.get("q0")
        b0 = options.get("b0")
        r0 = options.get("r0")
        diff = options.get("task_gain", 1)

        self.sim.reset_robot_pose(q0=q0, b0=b0, r0=r0)

        state = self.sim.state
        self._attach_task_state(state)

        if self.task.normalizer is not None:
            self.task.normalizer.reset_reference()

        obs = self.task.compute_initial_obs(state)
        self.task.reset()
        self.task.set_step_limit(self.max_step)
        self.task.set_difficulty(diff)

        self.current_step = 0
        self.ep_reward = 0.0

        info = {"env_id": self.env_id}

        return obs, info

    def _attach_task_state(self, state):
        """Attach optional mode/TPE state used only by the self-righting task."""
        task_state = getattr(self.controller, "task_state", None)
        if task_state is not None:
            state.task_state = task_state

        tpe = getattr(self.task, "tpe", None)
        if tpe is not None:
            state.tpe = tpe.state

    def close(self):

        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

        # liberar referências
        self.sim = None
