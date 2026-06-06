from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import math
import random
from typing import Any


OPERATOR_SWARM_SCHEMA_VERSION = "operator-swarm-v1"


@dataclass(slots=True)
class OperatorParticle:
    particle_id: int
    weights: dict[str, float]
    velocity: dict[str, float]
    personal_best_reward: float = 0.0
    personal_best_weights: dict[str, float] = field(default_factory=dict)
    pulls: int = 0
    reward_total: float = 0.0
    last_reward: float = 0.0

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "particle_id": int(self.particle_id),
            "weights": dict(self.weights),
            "velocity": dict(self.velocity),
            "personal_best_reward": float(self.personal_best_reward),
            "personal_best_weights": dict(self.personal_best_weights),
            "pulls": int(self.pulls),
            "reward_total": float(self.reward_total),
            "last_reward": float(self.last_reward),
        }

    @classmethod
    def from_state_dict(
        cls,
        data: Mapping[str, Any],
        *,
        operator_names: tuple[str, ...],
    ) -> "OperatorParticle":
        weights = _normalize_distribution(
            {
                str(operator): float(weight or 0.0)
                for operator, weight in (data.get("weights", {}) or {}).items()
            },
            operator_names,
        )
        velocity = {
            operator: float((data.get("velocity", {}) or {}).get(operator, 0.0) or 0.0)
            for operator in operator_names
        }
        personal_best_weights = _normalize_distribution(
            {
                str(operator): float(weight or 0.0)
                for operator, weight in (data.get("personal_best_weights", {}) or {}).items()
            },
            operator_names,
            fallback=weights,
        )
        return cls(
            particle_id=int(data.get("particle_id", 0) or 0),
            weights=weights,
            velocity=velocity,
            personal_best_reward=float(data.get("personal_best_reward", 0.0) or 0.0),
            personal_best_weights=personal_best_weights,
            pulls=int(data.get("pulls", 0) or 0),
            reward_total=float(data.get("reward_total", 0.0) or 0.0),
            last_reward=float(data.get("last_reward", 0.0) or 0.0),
        )


@dataclass(slots=True)
class OperatorSwarm:
    particles: list[OperatorParticle]
    global_best_weights: dict[str, float] = field(default_factory=dict)
    global_best_reward: float = -math.inf
    inertia: float = 0.70
    cognitive: float = 1.35
    social: float = 1.35
    operator_names: tuple[str, ...] = ()
    operator_pulls: dict[str, int] = field(default_factory=dict)
    operator_rewards: dict[str, float] = field(default_factory=dict)

    @classmethod
    def init(
        cls,
        operator_names: Iterable[Any] = (),
        *,
        n_particles: int = 16,
        seed: int = 0,
        inertia: float = 0.70,
        cognitive: float = 1.35,
        social: float = 1.35,
    ) -> "OperatorSwarm":
        names = _unique_operator_names(operator_names)
        if not names:
            return cls(
                particles=[],
                global_best_weights={},
                global_best_reward=-math.inf,
                inertia=float(inertia),
                cognitive=float(cognitive),
                social=float(social),
                operator_names=(),
            )
        rnd = random.Random(seed)
        particles: list[OperatorParticle] = []
        for particle_id in range(max(1, int(n_particles or 1))):
            weights = _initial_particle_weights(names, particle_id, rnd)
            particles.append(
                OperatorParticle(
                    particle_id=particle_id,
                    weights=weights,
                    velocity={operator: 0.0 for operator in names},
                    personal_best_reward=0.0,
                    personal_best_weights=dict(weights),
                )
            )
        return cls(
            particles=particles,
            global_best_weights=dict(particles[0].weights),
            global_best_reward=0.0,
            inertia=float(inertia),
            cognitive=float(cognitive),
            social=float(social),
            operator_names=names,
        )

    def ensure_operators(self, operator_names: Iterable[Any]) -> None:
        names = _unique_operator_names(operator_names)
        if names == self.operator_names:
            return
        if not names:
            self.operator_names = ()
            self.global_best_weights = {}
            return
        if not self.particles:
            replacement = OperatorSwarm.init(
                names,
                n_particles=16,
                inertia=self.inertia,
                cognitive=self.cognitive,
                social=self.social,
            )
            self.particles = replacement.particles
            self.global_best_weights = replacement.global_best_weights
            self.global_best_reward = replacement.global_best_reward
            self.operator_names = replacement.operator_names
            return
        for particle in self.particles:
            particle.weights = _normalize_distribution(particle.weights, names)
            particle.velocity = {
                operator: float(particle.velocity.get(operator, 0.0) or 0.0)
                for operator in names
            }
            particle.personal_best_weights = _normalize_distribution(
                particle.personal_best_weights,
                names,
                fallback=particle.weights,
            )
        self.global_best_weights = _normalize_distribution(
            self.global_best_weights,
            names,
            fallback=self.particles[0].weights,
        )
        self.operator_names = names
        self.operator_pulls = {
            operator: int(self.operator_pulls.get(operator, 0) or 0)
            for operator in names
            if int(self.operator_pulls.get(operator, 0) or 0) > 0
        }
        self.operator_rewards = {
            operator: float(self.operator_rewards.get(operator, 0.0) or 0.0)
            for operator in names
            if operator in self.operator_rewards
        }

    def select_particle(self, bandit_choice: str | int | None) -> OperatorParticle:
        if not self.particles:
            return OperatorParticle(particle_id=-1, weights={}, velocity={})
        try:
            particle_id = int(str(bandit_choice).strip())
        except (TypeError, ValueError):
            particle_id = self.particles[0].particle_id
        for particle in self.particles:
            if particle.particle_id == particle_id:
                return particle
        return self.particles[particle_id % len(self.particles)]

    def sample_operator(self, particle: OperatorParticle, rnd: random.Random) -> str:
        if not particle.weights:
            return ""
        threshold = rnd.random()
        cumulative = 0.0
        last_operator = ""
        for operator, weight in particle.weights.items():
            last_operator = operator
            cumulative += max(0.0, float(weight))
            if threshold <= cumulative:
                return operator
        return last_operator

    def score_bonus(self, operator: str, *, particle_id: int | None = None, scale: float = 0.035) -> float:
        operator = str(operator).strip()
        if not operator or operator not in self.operator_names:
            return 0.0
        pulls = int(self.operator_pulls.get(operator, 0) or 0)
        particle_weights = (
            self.select_particle(particle_id).weights
            if particle_id is not None
            else self.global_best_weights
        )
        if not particle_weights:
            return 0.0
        uniform = 1.0 / max(1, len(self.operator_names))
        centered = (float(particle_weights.get(operator, uniform) or 0.0) - uniform) / uniform
        reward_signal = (
            _bounded_confident_mean_reward(
                float(self.operator_rewards.get(operator, 0.0) or 0.0),
                pulls,
                max_abs=2.0,
            )
            if pulls > 0
            else 0.0
        )
        confidence = min(1.0, math.log1p(pulls) / math.log(4.0)) if pulls > 0 else 0.25
        bonus = float(scale) * confidence * (0.70 * centered + 0.30 * reward_signal)
        return max(-float(scale), min(float(scale), bonus))

    def update(
        self,
        particle_id: int | str | None,
        reward: float,
        *,
        operator: str = "",
        rnd: random.Random | None = None,
    ) -> None:
        if not self.particles:
            return
        particle = self.select_particle(particle_id)
        if particle.particle_id < 0:
            return
        reward_value = float(reward or 0.0)
        particle.pulls += 1
        particle.reward_total += reward_value
        particle.last_reward = reward_value
        operator = str(operator).strip()
        if operator:
            self.ensure_operators((*self.operator_names, operator))
            self.operator_pulls[operator] = int(self.operator_pulls.get(operator, 0) or 0) + 1
            self.operator_rewards[operator] = float(self.operator_rewards.get(operator, 0.0) or 0.0) + reward_value
        update_rnd = rnd or random.Random(_stable_update_seed(particle.particle_id, operator, reward_value, particle.pulls))
        self._advance_particle(particle, operator=operator, reward=reward_value, rnd=update_rnd)
        if particle.pulls == 1 or reward_value > particle.personal_best_reward:
            particle.personal_best_reward = reward_value
            particle.personal_best_weights = dict(particle.weights)
        if reward_value > self.global_best_reward or not self.global_best_weights:
            self.global_best_reward = reward_value
            self.global_best_weights = dict(particle.weights)

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OPERATOR_SWARM_SCHEMA_VERSION,
            "operator_names": list(self.operator_names),
            "particles": [particle.to_state_dict() for particle in self.particles],
            "global_best_weights": dict(self.global_best_weights),
            "global_best_reward": float(self.global_best_reward),
            "inertia": float(self.inertia),
            "cognitive": float(self.cognitive),
            "social": float(self.social),
            "operator_pulls": dict(self.operator_pulls),
            "operator_rewards": dict(self.operator_rewards),
        }

    @classmethod
    def from_state_dict(
        cls,
        data: Mapping[str, Any] | None,
        *,
        operator_names: Iterable[Any] = (),
    ) -> "OperatorSwarm":
        requested_names = _unique_operator_names(operator_names)
        if not isinstance(data, Mapping):
            return cls.init(requested_names)
        stored_names = _unique_operator_names(data.get("operator_names", ()) or ())
        names = _unique_operator_names((*stored_names, *requested_names))
        swarm = cls(
            particles=[
                OperatorParticle.from_state_dict(item, operator_names=names)
                for item in (data.get("particles", []) or [])
                if isinstance(item, Mapping)
            ],
            global_best_weights=_normalize_distribution(
                {
                    str(operator): float(weight or 0.0)
                    for operator, weight in (data.get("global_best_weights", {}) or {}).items()
                },
                names,
            ),
            global_best_reward=_float_or_default(data.get("global_best_reward", -math.inf), -math.inf),
            inertia=float(data.get("inertia", 0.70) or 0.70),
            cognitive=float(data.get("cognitive", 1.35) or 1.35),
            social=float(data.get("social", 1.35) or 1.35),
            operator_names=names,
            operator_pulls={
                str(operator): int(count or 0)
                for operator, count in (data.get("operator_pulls", {}) or {}).items()
                if str(operator).strip() and int(count or 0) > 0
            },
            operator_rewards={
                str(operator): float(reward or 0.0)
                for operator, reward in (data.get("operator_rewards", {}) or {}).items()
                if str(operator).strip()
            },
        )
        if names and not swarm.particles:
            return cls.init(names)
        swarm.ensure_operators(names)
        return swarm

    def _advance_particle(
        self,
        particle: OperatorParticle,
        *,
        operator: str,
        reward: float,
        rnd: random.Random,
    ) -> None:
        if not particle.weights:
            return
        bounded_reward = max(-1.0, min(1.0, float(reward) / 3.0))
        updated: dict[str, float] = {}
        for name in self.operator_names:
            current = float(particle.weights.get(name, 0.0) or 0.0)
            direct = 0.0
            if operator and name == operator:
                direct = 0.10 * bounded_reward * (1.0 - current if bounded_reward >= 0 else current)
            personal = float(particle.personal_best_weights.get(name, current) or 0.0)
            global_best = float(self.global_best_weights.get(name, current) or 0.0)
            old_velocity = float(particle.velocity.get(name, 0.0) or 0.0)
            velocity = (
                self.inertia * old_velocity
                + self.cognitive * rnd.random() * (personal - current)
                + self.social * rnd.random() * (global_best - current)
                + direct
            )
            particle.velocity[name] = max(-0.20, min(0.20, velocity))
            updated[name] = current + particle.velocity[name]
        particle.weights = _normalize_distribution(updated, self.operator_names)


def _initial_particle_weights(
    names: tuple[str, ...],
    particle_id: int,
    rnd: random.Random,
) -> dict[str, float]:
    if not names:
        return {}
    weights: dict[str, float] = {}
    focus_index = particle_id % len(names)
    for index, operator in enumerate(names):
        jitter = rnd.uniform(0.85, 1.15)
        if index == focus_index:
            jitter *= 1.35
        weights[operator] = jitter
    return _normalize_distribution(weights, names)


def _normalize_distribution(
    weights: Mapping[str, float],
    operator_names: tuple[str, ...],
    *,
    fallback: Mapping[str, float] | None = None,
) -> dict[str, float]:
    if not operator_names:
        return {}
    cleaned: dict[str, float] = {}
    for operator in operator_names:
        value = float(weights.get(operator, 0.0) or 0.0)
        if value <= 0.0 and fallback is not None:
            value = float(fallback.get(operator, 0.0) or 0.0)
        cleaned[operator] = max(0.0001, value)
    total = sum(cleaned.values())
    if total <= 0.0:
        uniform = 1.0 / len(operator_names)
        return {operator: uniform for operator in operator_names}
    return {operator: value / total for operator, value in cleaned.items()}


def _unique_operator_names(values: Iterable[Any]) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        names.append(text)
    return tuple(names)


def _stable_update_seed(particle_id: int, operator: str, reward: float, pulls: int) -> int:
    operator_hash = sum((index + 1) * ord(char) for index, char in enumerate(operator))
    reward_bucket = int(round(abs(float(reward)) * 1000.0))
    return (
        (int(particle_id) + 1) * 1_000_003
        + (int(pulls) + 1) * 97_409
        + operator_hash * 131
        + reward_bucket
    )


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bounded_confident_mean_reward(total_reward: float, pulls: float, *, max_abs: float) -> float:
    if pulls <= 0:
        return 0.0
    mean_reward = float(total_reward) / float(pulls)
    bounded_mean_reward = max(-max_abs, min(max_abs, mean_reward))
    confidence = min(1.0, math.log1p(float(pulls)) / math.log(4.0))
    return bounded_mean_reward * confidence
