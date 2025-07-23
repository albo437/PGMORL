from . import VecEnvWrapper
import numpy as np

class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=False, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, use_tf=False, obj_rms=False):
        VecEnvWrapper.__init__(self, venv)
        if use_tf:
            from baselines.common.running_mean_std import TfRunningMeanStd
            self.ob_rms = TfRunningMeanStd(shape=self.observation_space.shape, scope='ob_rms') if ob else None
            self.ret_rms = TfRunningMeanStd(shape=(), scope='ret_rms') if ret else None
            self.obj_rms = TfRunningMeanStd(shape=(), scope='obj_rms') if ret else None
        else:
            from baselines.common.running_mean_std import RunningMeanStd
            self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
            self.ret_rms = RunningMeanStd(shape=()) if ret else None
            self.obj_rms = RunningMeanStd(shape=()) if ret and obj_rms else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.obj = np.array([None] * self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, terminated, truncated, infos = self.venv.step_wait()
        news = terminated | truncated  # Combine for compatibility

        # Handle multi-objective rewards (2D arrays) for PGMORL
        if isinstance(rews[0], np.ndarray) and rews[0].ndim > 0:
            # Multi-objective rewards - don't normalize, just pass through
            # Initialize ret with correct shape if needed
            if self.ret.ndim == 1 and rews[0].ndim > 1:
                self.ret = np.zeros((self.num_envs, rews[0].shape[0]))
            # Only update ret if shapes match
            if self.ret.shape == rews.shape:
                self.ret = self.ret * self.gamma + rews
        else:
            # Single-objective rewards - normal normalization
            self.ret = self.ret * self.gamma + rews
            
        if 'obj' in infos[0]:
            for info in infos:
                info['obj_raw'] = info['obj']
            obj = np.array([info['obj'] for info in infos])
            # For PGMORL: Accumulate objectives properly for episode totals (no gamma discounting for objectives)
            self.obj = self.obj + obj if self.obj[0] is not None else obj

        obs = self._obfilt(obs)

        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        # Still track objective variance for normalization purposes, but don't modify the original objectives
        if self.obj_rms:
            self.obj_rms.update(self.obj)
            # Don't modify the original obj values - PGMORL needs them raw

        self.ret[news] = 0.
        if 'obj' in infos[0]:
            self.obj[news] = np.zeros_like(self.obj[news])

        return obs, rews, terminated, truncated, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)
