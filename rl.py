"""Training the same creatures with gradients instead of evolution.

Everywhere else in this project, behaviour comes from evolution: creatures that survive to
breed pass on mutated weights, and nobody ever specifies what good behaviour looks like.
This module does the other thing. It trains one shared policy network by policy gradient,
where good behaviour is whatever the reward function says it is.

The comparison is the point, and the interesting part is not which method reaches a higher
number. It is that the two methods optimise different things.

Evolution optimises reproductive success, because that is the only thing that determines
whose weights survive. Parental care therefore costs an individual nothing in the currency
evolution counts: a creature that abandons its young simply has no descendants, and the
question never arises again.

Policy gradient optimises the reward we write down. Parental care is expensive to the
individual: sitting beside a pup means not eating. So unless the reward itself accounts for
offspring, there is no gradient anywhere that points towards caring for them.

That gives a sharp prediction. Train on survival alone and care should not appear at all.
Add a term for young surviving and it should. The two reward functions here differ by
exactly that one term, so whatever separates the results is that term and nothing else.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from agent import Agent, draw_lifespan
from brain import ACTIONS, Brain
from population import draw_founders
from world import World

# Reward for being alive one more tick. Both objectives include it, so both learn to eat.
SURVIVAL_REWARD = 0.01

# Reward for one of your young reaching independence. Only the "offspring" objective has it.
OFFSPRING_REWARD = 1.0


class PolicyNet(nn.Module):
    """The same shape of network the evolved creatures use, built so gradients can flow.

    Deliberately identical to Brain: twelve inputs, one hidden layer of twelve with tanh,
    six action scores. Keeping the architectures the same means any difference in behaviour
    comes from how the weights were found, not from what the network could represent.
    """

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(Brain.N_INPUTS, Brain.N_HIDDEN)
        self.out = nn.Linear(Brain.N_HIDDEN, Brain.N_OUTPUTS)

    def forward(self, senses: torch.Tensor) -> torch.Tensor:
        return self.out(torch.tanh(self.hidden(senses)))

    def to_brain(self) -> Brain:
        """Repackage the trained weights as a Brain.

        This lets a trained policy be measured with exactly the same behaviour probes used
        on the evolved populations, rather than a separate set of measurements that might
        quietly differ. Torch stores its weight matrices transposed relative to the plain
        numpy version, which is why they are flipped on the way across.
        """
        with torch.no_grad():
            parts = [
                self.hidden.weight.T.flatten(),
                self.hidden.bias,
                self.out.weight.T.flatten(),
                self.out.bias,
            ]
            flat = torch.cat([p.detach().reshape(-1) for p in parts]).numpy()
        return Brain(flat.astype(np.float64))


def _build_world(cfg, rng):
    world = World(
        width=cfg.width, height=cfg.height, n_food=cfg.n_food,
        food_unlimited=cfg.food_unlimited, food_respawn_rate=cfg.food_respawn_rate,
        wrap_edges=cfg.wrap_edges, rng=rng,
    )
    if cfg.nests_enabled:
        world.scatter_nests(cfg.n_nests)
    return world


def run_episode(cfg, policy: PolicyNet, reward_offspring: bool, seed: int, founders=None):
    """Run one simulation where every creature is driven by the shared policy.

    Returns the log probability and reward of every choice every creature made, along with
    some statistics about how the population did. Dependent pups are skipped: they cannot
    act, so they make no choices to learn from.
    """
    rng = random.Random(seed)
    world = _build_world(cfg, rng)

    brains = (
        draw_founders(founders, cfg.n_initial_agents, rng) if founders
        else [Brain.random(rng) for _ in range(cfg.n_initial_agents)]
    )
    world.agents = []
    for brain in brains:
        x, y = world.random_square()
        a = Agent(x, y, cfg.energy_start, brain, lifespan=draw_lifespan(cfg, rng))
        if cfg.ageing_enabled:
            a.age = rng.randrange(0, max(1, cfg.fertility_end_age))
        world.agents.append(a)

    # One list of log probabilities and one of rewards per creature, in step order.
    log_probs: dict = {}
    rewards: dict = {}
    births = 0

    for _ in range(cfg.n_ticks):
        world.rebuild_occupancy(cfg.crowding_radius)

        movers = [a for a in world.agents if a.alive and not a.is_dependent]
        pups_in_care = [(a, a.parent) for a in world.agents
                        if a.alive and a.is_dependent and a.parent is not None]

        chosen: dict = {}
        if movers:
            # Every creature's senses in one tensor, one forward pass for the whole
            # population. Asking the network once per creature would be far slower and
            # would make training on populations of this size impractical.
            senses = torch.tensor(
                np.array([a.sense(world, cfg, neighbours=a.neighbours) for a in movers]),
                dtype=torch.float32,
            )
            dist = Categorical(logits=policy(senses))
            picks = dist.sample()
            step_log_probs = dist.log_prob(picks)
            for i, a in enumerate(movers):
                chosen[a] = ACTIONS[picks[i].item()]
                log_probs.setdefault(a, []).append(step_log_probs[i])
                rewards.setdefault(a, []).append(0.0)

        newborns = []
        for a in movers:
            child = a.act(world, cfg, rng, action=chosen.get(a))
            if child is not None:
                if len(world.agents) + len(newborns) < cfg.max_population:
                    newborns.append(child)
                    births += 1
                else:
                    if child.nest is not None:
                        world.release_nest(*child.nest)
                        child.nest = None
                    a.refund_birth(cfg)

        # Pups act too, but only to be fed or neglected. They make no choices.
        for a in world.agents:
            if a.alive and a.is_dependent:
                a.act(world, cfg, rng)

        # Staying alive is worth a little, to both objectives.
        for a in movers:
            if a.alive and rewards.get(a):
                rewards[a][-1] += SURVIVAL_REWARD

        # A pup reaching independence pays its parent, but only under the objective that
        # counts offspring. This single term is the whole difference between the two.
        if reward_offspring:
            for pup, parent in pups_in_care:
                if pup.alive and not pup.is_dependent and rewards.get(parent):
                    rewards[parent][-1] += OFFSPRING_REWARD

        survivors = []
        for a in world.agents:
            if a.alive:
                survivors.append(a)
                continue
            if a.nest is not None:
                world.release_nest(*a.nest)
                a.nest = None
            if a.parent is not None:
                a._leave_the_parent()
        world.agents = survivors + newborns

        world.respawn_step()
        if world.population == 0:
            break

    stats = {
        "final_population": world.population,
        "births": births,
        "mean_return": float(np.mean([sum(r) for r in rewards.values()])) if rewards else 0.0,
    }
    return log_probs, rewards, stats


def train(cfg, reward_offspring: bool, iterations: int = 40, lr: float = 0.01,
          gamma: float = 0.99, seed: int = 0, founders=None, log_every: int = 5):
    """Train one shared policy with REINFORCE.

    Every creature alive contributes its own trajectory, and all of them update the same
    network. Sharing one policy across the population is standard in multi agent settings
    and it is also the fairer comparison here, since the evolved populations share a common
    ancestry rather than each creature being independent.
    """
    torch.manual_seed(seed)
    policy = PolicyNet()
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []

    for step in range(iterations):
        log_probs, rewards, stats = run_episode(
            cfg, policy, reward_offspring, seed=seed * 1000 + step, founders=founders
        )
        if not log_probs:
            history.append({"iteration": step, **stats, "loss": 0.0})
            continue

        all_log_probs, all_returns = [], []
        for agent, steps in log_probs.items():
            running = 0.0
            returns = []
            for r in reversed(rewards[agent]):
                running = r + gamma * running
                returns.append(running)
            returns.reverse()
            all_log_probs.extend(steps)
            all_returns.extend(returns)

        returns_t = torch.tensor(all_returns, dtype=torch.float32)
        # Centre and scale the returns. Without this the gradient is dominated by how long
        # a creature happened to live rather than by whether its choices were good ones.
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = -(torch.stack(all_log_probs) * returns_t).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        history.append({"iteration": step, **stats, "loss": float(loss.item())})
        if log_every and step % log_every == 0:
            print(
                f"  iter {step:3d}  population {stats['final_population']:4d}  "
                f"births {stats['births']:4d}  mean return {stats['mean_return']:7.3f}",
                flush=True,
            )

    return policy, history
