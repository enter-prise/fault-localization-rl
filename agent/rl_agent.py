# agent/rl_agent.py

class RLAgent:
    def __init__(self, policy):
        self.policy = policy

    def run_episode(self, env):
        state = env.reset()
        total_reward = 0

        while True:
            action = self.policy.select_action(state)

            next_state, reward, done = env.step(action)

            total_reward += reward
            state = next_state

            if done:
                break

        return total_reward