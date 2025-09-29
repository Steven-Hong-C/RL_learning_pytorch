import gymnasium as gym
from gymnasium import Wrapper


class ControlledRenderingWrapper(Wrapper):
    def __init__(self, env, render_interval=100):
        super().__init__(env)
        self.render_interval = render_interval
        self.step_count = 0
        self.should_render = False

    def render(self):
        # 只在标志为True时渲染
        if self.should_render:
            return self.env.render()
        return None

    def reset(self, **kwargs):
        self.step_count = 0
        self.should_render = False
        return self.env.reset(**kwargs)


# 使用包装器
env = gym.make("CartPole-v1", render_mode="human")
env = ControlledRenderingWrapper(env, render_interval=50)

observation, info = env.reset()

for i in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if i % 100 == 0:
        env.should_render = True

    # 手动调用render，但包装器会控制实际渲染
    env.render()

    if terminated or truncated:
        observation, info = env.reset()

env.close()