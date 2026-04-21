# agent/policy.py

import random


class RandomPolicy:
    def select_action(self, state):
        return random.choice(["jump", "retrieve", "submit"])