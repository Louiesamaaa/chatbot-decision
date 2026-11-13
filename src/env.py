import gymnasium as gym
import numpy as np
from gymnasium import spaces

ANSWER, CLARIFY, HANDOFF = 0, 1, 2
ACTION_NAMES = ["answer", "clarify", "handoff"]

def customer_msg(conversations):
    return  [text for speaker, text in conversations['original'] if speaker == "customer"]

def is_bot_solvable(conversations):

    buttons = [
        turn["targets"][2]
        for turn in conversations["delexed"]
        if turn["speaker"] == "action" and len(turn["targets"]) > 2
    ]

    return bool(buttons) and all(b.startswith("search-") or b == "select-faq" for b in buttons)

def build_scenarios(conversations, classifier, max_msgs = 4):

    scenarios = []
    for convo in conversations:
        msgs = customer_msg(convo)
        if not msgs:
            continue
        n = min(len(msgs), max_msgs)
        texts = [" ".join(msgs[:k]) for k in range(1, n+1)]
        probs = classifier.model.predict_proba(texts)
        scenarios.append({
            "convo_id": convo["convo_id"],
            "flow": convo["scenario"]["flow"],
            "bot_solvable": is_bot_solvable(convo),
            "probs": probs,
            "messages": msgs[:n]
        })

    return scenarios

class Handoffenv(gym.Env):
    metadata = {"render_mode": ["human"]}

    def  __init__(self, scenarios , intent_names, solvable_ratio = 0.5, 
                  max_turns = 6, answer_success_prob = 0.9, seed = None):
        super().__init__()
        self.scenarios = scenarios
        self.intent_names = list(intent_names)
        self.n_intents = len(self.intent_names)
        self.max_turns = max_turns
        self.answer_success_prob = answer_success_prob


        self.solvable_ratio = solvable_ratio
        self.solvable = [s for s in scenarios if s["bot_solvable"]]
        self.unsolvable = [s for s in scenarios if not s["bot_solvable"]]

        self.action_space = spaces.Discrete(3)

        obs_size = 2 + self.n_intents + 4
        self.observation_space = spaces.Box(0.0,1.0, shape = (obs_size,), dtype = np.float32)

        self.reset(seed= seed)

    def _current_probs(self):
        return self.s["probs"][self.n_seen - 1]
 
    def _predicted_intent(self):
        return self.intent_names[int(np.argmax(self._current_probs()))]
 
    def _obs(self):
        p = self._current_probs()
        top2 = np.sort(p)[-2:]
        one_hot = np.zeros(self.n_intents)
        one_hot[int(np.argmax(p))] = 1.0
        return np.concatenate([
            [top2[1], top2[1] - top2[0]],                      # confidence, margin
            one_hot,                                            # which intent it thinks
            [self.turn / self.max_turns,                        # how long the chat is
             self.n_seen / 4,                                   # how many messages seen
             min(self.frustration, 4) / 4,                      # how annoyed the customer is
             float(self.asked_human)],                          # did they ask for a human?
        ]).astype(np.float32)
 
    def _maybe_ask_for_human(self):
        """Frustrated customers may ask for a human (more likely as frustration grows)."""
        if not self.asked_human and self.np_random.random() < 0.15 * self.frustration:
            self.asked_human = True
 
    def action_masks(self):
        """Which actions are allowed right now."""
        if self.asked_human:
            return np.array([False, False, True])
        return np.array([True, True, True])
 
    # Gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.solvable_ratio is None:
            pool = self.scenarios
        else:
            pool = self.solvable if self.np_random.random() < self.solvable_ratio else self.unsolvable
        self.s = pool[self.np_random.integers(len(pool))]
        self.n_seen = 1          # the bot has seen the customer's first message
        self.turn = 0
        self.frustration = 0
        self.asked_human = False
        return self._obs(), {}
 
    def step(self, action):
        action = int(action)
        reward, terminated, truncated = 0.0, False, False
        info = {"flow": self.s["flow"], "bot_solvable": self.s["bot_solvable"],
                "ignored_human": False, "outcome": None}
 
        # Hard rule for agents without action masking
        if self.asked_human and action != HANDOFF:
            info["ignored_human"] = True
            reward -= 10
            action = HANDOFF
 
        if action == HANDOFF:
            reward += -3
            terminated = True
            info["outcome"] = "handoff"
 
        elif action == CLARIFY:
            reward += -1
            self.turn += 1
            if self.n_seen < len(self.s["probs"]):
                self.n_seen += 1          # customer reveals their next real message
            self.frustration += 0.5       # asking questions is a little annoying
            self._maybe_ask_for_human()
 
        elif action == ANSWER:
            reward += -1
            self.turn += 1
            correct_intent = self._predicted_intent() == self.s["flow"]
            if (self.s["bot_solvable"] and correct_intent
                    and self.np_random.random() < self.answer_success_prob):
                reward += 10
                terminated = True
                info["outcome"] = "resolved"
            else:
                reward += -15
                self.frustration += 2
                info["wrong_answer"] = True
                self._maybe_ask_for_human()
 
        # Customer gives up if the chat drags on too long
        if not terminated and self.turn >= self.max_turns:
            reward += -10
            truncated = True
            info["outcome"] = "abandoned"
 
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        msg = self.s["messages"][self.n_seen - 1]
        print(f"[turn {self.turn}] customer: {msg!r} | bot thinks: {self._predicted_intent()} "
              f"({self._current_probs().max():.2f}) | frustration={self.frustration} "
              f"| asked_human={self.asked_human}")