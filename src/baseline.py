import numpy as np
from src.env import HANDOFF, CLARIFY, ANSWER
from src.classifier import IntentClassifier
from gymnasium.utils.env_checker import check_env

clf = IntentClassifier.load("../models/solvable_classifier.joblib")


def always_handoff(env):
    return HANDOFF

def never_handoff(env, n):
    if env.n_seen - 1 < n:
        action = CLARIFY
    else:
        action = ANSWER
    return action

def confidence_handoff(env, threshold = 0.8, n = 3):
    if env.asked_human:
        return HANDOFF
    if env._current_probs().max() >= threshold:
        return ANSWER
    if env.n_seen - 1 < n:
        return CLARIFY
    return HANDOFF

def classifier_handoff(env,solvable_clf, high_threshold = 0.8, low_threshold = 0.1, n =3):
    probs = solvable_clf.model.predict_proba([" ".join(env.s["messages"][:env.n_seen])])[0]
    if env.asked_human:
        return HANDOFF
    if probs[1] >= high_threshold:
        return ANSWER
    if probs[1] < high_threshold and probs[1] >= low_threshold:
        if env.n_seen - 1 < n:
            return CLARIFY
        else:
            return HANDOFF
    if probs[1] < low_threshold:
        return HANDOFF

    return HANDOFF  