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
from evaluate import care_under_conflict, food_seeking_score, pup_seeking_score
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
    """Build the pen. Every phase builds it here, so every phase gets the same pen.

    The order of the three steps matters. Feeders go down first, because food is scattered
    around whatever feeders exist. Nests go down last, with the perimeter setting the config
    asks for.

    This used to be split in two, with the world half built here and finished off separately
    for the long run. That let the two drift apart in ways nobody intended. Training happened
    with food spread evenly over the whole grid, and the pen the trained creatures were then
    released into had it piled around four feeders. Worse, the nests were scattered here
    before the perimeter setting was ever consulted, and since scatter_nests stops once it
    has enough, the later call asking for nests around the walls did nothing at all.
    """
    world = World(
        # No food yet. It is scattered below, once the feeders it should cluster around
        # are in place.
        width=cfg.width, height=cfg.height, n_food=0,
        food_unlimited=cfg.food_unlimited, food_respawn_rate=cfg.food_respawn_rate,
        wrap_edges=cfg.wrap_edges, rng=rng,
    )
    if cfg.n_feeders:
        # Kept clear of the nesting band, and of a feeder's own spread of food, so that no
        # food at all lands among the nest boxes.
        inset = (cfg.perimeter_band + cfg.feeder_spread
                 if cfg.nests_enabled and cfg.nests_on_perimeter else 0)
        world.place_feeders(cfg.n_feeders, inset=inset)
    world.scatter_food(cfg.n_food, spread=cfg.feeder_spread)
    if cfg.nests_enabled:
        world.scatter_nests(cfg.n_nests, on_perimeter=cfg.nests_on_perimeter,
                            band=cfg.perimeter_band)
    return world


def run_episode(cfg, policy: PolicyNet, reward_offspring: bool, seed: int,
                senses_sink: list | None = None):
    """Run one simulation where every creature is driven by the shared policy.

    Returns the log probability and reward of every choice every creature made, along with
    some statistics about how the population did. Dependent pups are skipped: they cannot
    act, so they make no choices to learn from.

    Pass a list as senses_sink to keep a copy of every sense vector the population received.
    The representation analysis needs the situations creatures are actually in, and this is
    where they pass through. Nothing else changes when it is given.
    """
    rng = random.Random(seed)
    world = _build_world(cfg, rng)

    world.agents = []
    for brain in [Brain.random(rng) for _ in range(cfg.n_initial_agents)]:
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
            raw = np.array([a.sense(world, cfg, neighbours=a.neighbours) for a in movers])
            if senses_sink is not None:
                senses_sink.append(raw)
            senses = torch.tensor(raw, dtype=torch.float32)
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
          gamma: float = 0.99, seed: int = 0, log_every: int = 5, probe_every: int = 0):
    """Train one shared policy with REINFORCE.

    Every creature alive contributes its own trajectory, and all of them update the same
    network. Sharing one policy across the population is standard in multi agent settings
    and it is also the fairer comparison here, since the evolved populations share a common
    ancestry rather than each creature being independent.

    With probe_every set, the policy's behaviour is measured every so many iterations using
    the same probes applied to the evolved populations. Measuring only the finished policy
    tells you what it ended up doing but not whether the behaviour appeared early, grew
    steadily, or never appeared at all, and that difference is the interesting part.
    """
    torch.manual_seed(seed)
    policy = PolicyNet()
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []

    for step in range(iterations):
        log_probs, rewards, stats = run_episode(
            cfg, policy, reward_offspring, seed=seed * 1000 + step
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

        record = {"iteration": step, **stats, "loss": float(loss.item())}
        if probe_every and (step % probe_every == 0 or step == iterations - 1):
            brain = policy.to_brain()
            record["pup_seeking"] = pup_seeking_score(brain, trials=300, seed=step)
            record["food_seeking"] = food_seeking_score(brain, trials=300, seed=step)
            record["care_vs_food"] = care_under_conflict(brain, trials=300, seed=step)["preference"]
        history.append(record)
        if log_every and step % log_every == 0:
            print(
                f"  iter {step:3d}  population {stats['final_population']:4d}  "
                f"births {stats['births']:4d}  mean return {stats['mean_return']:7.3f}",
                flush=True,
            )

    return policy, history


def run_enclosure(cfg, policy: PolicyNet, optimiser, update_every: int = 200,
                  probe_every: int = 500, gamma: float = 0.99, seed: int = 0,
                  lr_scale: float = 1.0, verbose: bool = True):
    """Phase two: one colony, founded once, left to run while the policy keeps learning.

    This is deliberately not episodic. Calhoun founded a single colony and watched it for
    years, and the whole phenomenon is what happens to a population over its own history:
    it grows, it fills the space, and only then does anything go wrong. Restarting the world
    every few hundred ticks would destroy exactly that.

    So the world persists from beginning to end, and the policy is updated every so many
    ticks from whatever experience has accumulated in that window. The reward function never
    changes. The environment never changes. The only thing that changes is how many
    creatures are in it, which is what we want to hold responsible for anything we see.
    """
    rng = random.Random(seed)
    world = _build_world(cfg, rng)

    world.agents = []
    for _ in range(cfg.n_initial_agents):
        x, y = world.random_square()
        a = Agent(x, y, cfg.energy_start, Brain.random(rng), lifespan=draw_lifespan(cfg, rng))
        if cfg.ageing_enabled:
            a.age = rng.randrange(0, max(1, cfg.maturity_age * 2))
        world.agents.append(a)

    log_probs: dict = {}
    rewards: dict = {}
    history = []

    def flush():
        """Turn the window's experience into one gradient step, then clear it."""
        if not log_probs:
            return 0.0
        steps, returns = [], []
        for agent, agent_steps in log_probs.items():
            running = 0.0
            agent_returns = []
            for r in reversed(rewards[agent]):
                running = r + gamma * running
                agent_returns.append(running)
            agent_returns.reverse()
            steps.extend(agent_steps)
            returns.extend(agent_returns)
        if not steps:
            log_probs.clear(); rewards.clear()
            return 0.0
        returns_t = torch.tensor(returns, dtype=torch.float32)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        loss = -(torch.stack(steps) * returns_t).mean() * lr_scale
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        log_probs.clear(); rewards.clear()
        return float(loss.item())

    for tick in range(cfg.n_ticks):
        world.rebuild_occupancy(cfg.crowding_radius)
        movers = [a for a in world.agents if a.alive and not a.is_dependent]
        pups_in_care = [(a, a.parent) for a in world.agents
                        if a.alive and a.is_dependent and a.parent is not None]

        chosen: dict = {}
        if movers:
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

        births = 0
        newborns = []
        for a in movers:
            child = a.act(world, cfg, rng, action=chosen.get(a))
            if child is not None:
                if len(world.agents) + len(newborns) < cfg.max_population:
                    newborns.append(child); births += 1
                else:
                    if child.nest is not None:
                        world.release_nest(*child.nest); child.nest = None
                    a.refund_birth(cfg)

        for a in world.agents:
            if a.alive and a.is_dependent:
                a.act(world, cfg, rng)

        for a in movers:
            if a.alive and rewards.get(a):
                rewards[a][-1] += SURVIVAL_REWARD
        for pup, parent in pups_in_care:
            if pup.alive and not pup.is_dependent and rewards.get(parent):
                rewards[parent][-1] += OFFSPRING_REWARD

        survivors = []
        toll = {"starvation": 0, "old age": 0, "neglect": 0}
        for a in world.agents:
            if a.alive:
                survivors.append(a); continue
            if a.nest is not None:
                world.release_nest(*a.nest); a.nest = None
            if a.parent is not None:
                a._leave_the_parent()
            if a.cause_of_death in toll:
                toll[a.cause_of_death] += 1
            # A dead creature's unfinished trajectory is still experience worth learning
            # from. Dropping it would quietly train only on the survivors, which is the
            # one group whose choices are guaranteed not to have been fatal.
        deaths = len(world.agents) - len(survivors)
        world.agents = survivors + newborns
        world.respawn_step()

        record = {
            "tick": tick, "population": world.population, "births": births,
            "deaths": deaths, "died_neglected": toll["neglect"],
            "died_starving": toll["starvation"], "died_old": toll["old age"],
            "nests_free": len(world.nests) - len(world.occupied_nests),
        }
        measured = [a.neighbours for a in world.agents if a.neighbours is not None]
        record["mean_neighbours"] = (sum(measured) / len(measured)) if measured else 0.0
        record["clustering"] = world.clustering(measured, cfg.crowding_radius)

        if (tick + 1) % update_every == 0:
            record["loss"] = flush()

        if probe_every and tick % probe_every == 0:
            brain = policy.to_brain()
            record["pup_seeking"] = pup_seeking_score(brain, trials=300, seed=tick)
            record["food_seeking"] = food_seeking_score(brain, trials=300, seed=tick)
            if verbose:
                print(f"  tick {tick:6d}  pop {world.population:5d}  "
                      f"crowding {record['mean_neighbours']:5.1f}  "
                      f"clustering {record['clustering']:4.1f}  "
                      f"pup-seeking {record['pup_seeking']:.3f}", flush=True)

        history.append(record)
        if world.population == 0:
            break

    return world, history
