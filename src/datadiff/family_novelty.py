from __future__ import annotations


def candidate_family_novelty_reward(
    previous_hits: int,
    *,
    enable_family_saturation: bool = True,
    family_saturation_threshold: int = 8,
    saturated_family_reward: float = 0.02,
) -> float:
    if enable_family_saturation and family_saturation_threshold > 0 and previous_hits >= family_saturation_threshold:
        return max(0.0, saturated_family_reward)
    if previous_hits <= 0:
        return 4.0
    if previous_hits <= 2:
        return 1.5
    if previous_hits <= 8:
        return 0.75 / max(1.0, previous_hits**0.5)
    return 0.10


def family_key_matches_known_family(
    candidate_family: str,
    known_bug_families: list[str] | tuple[str, ...],
) -> bool:
    candidate_root, candidate_backends = split_family_key(candidate_family)
    for known_family in known_bug_families:
        known_root, known_backends = split_family_key(known_family)
        if known_root != candidate_root:
            continue
        if not known_backends or not candidate_backends or known_backends & candidate_backends:
            return True
    return False


def split_family_key(family_key: str) -> tuple[str, set[str]]:
    root, _, backend_part = str(family_key).partition("@")
    backends = {backend.strip() for backend in backend_part.split(",") if backend.strip()}
    return root.strip(), backends
