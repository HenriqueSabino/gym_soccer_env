from pathlib import Path
import numpy as np
import imageio
import torch

class VideoRecorder:
    def __init__(self, fps=30):
        self.fps = fps
        self.frames = []

    def reset(self):
        self.frames = []

    def record_frame(self, env):
        self.frames.append(env.render())

    def save(self, filename):
        imageio.mimsave(f"{filename}", self.frames, fps=self.fps)


def record_episodes(act_func, n_timesteps, path, env_params: dict, make_fn: callable):
    recorder = VideoRecorder()
    done = True

    env = make_fn(env_params)

    for _ in range(n_timesteps):
        if done or truncated:
            env.reset()
            obs, r, done, truncated, infos = env.last()
            recorder.record_frame(env)
        else:
            with torch.no_grad():
                act = act_func(obs)
                if isinstance(act, int) or \
                    isinstance(act, np.int64) or \
                    isinstance(act, np.int32):
                    # print(f"act: {act} | type {type(act)}")
                    env.step(act)
                else:
                    # print(f"act: {act} | len {len(act)}")
                    for a in act:
                        env.step(a)
                obs, r, done, truncated, infos = env.last()
            recorder.record_frame(env)

    env.close()
    Path(path).parents[0].mkdir(parents=True, exist_ok=True)
    recorder.save(path)
