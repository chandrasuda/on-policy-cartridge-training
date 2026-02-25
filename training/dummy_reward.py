"""Dummy reward function for cartridge distillation.

Cartridge training uses KL distillation loss (actor vs ref), not
reward-based RL. This returns 0 for all responses so veRL's GRPO
pipeline doesn't crash, but the actual training signal comes from
the KL loss between actor (with cartridge) and ref (without).
"""


def compute_score(data_source, solution_str, ground_truth, **kwargs):
    """Always return 0 — distillation doesn't use rewards."""
    return 0.0
