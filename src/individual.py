"""Every creature with its own network, and all of them run in one pass.

Until now a single policy drove every creature in the pen. That was convenient but it
quietly ruled out the thing Calhoun actually described. His colony did not fail uniformly;
it came apart into groups. Some males withdrew entirely and did nothing but groom, others
fought constantly, a shrinking minority went on breeding while most stopped. A population
sharing one set of weights cannot differentiate at all, so no amount of added pressure could
ever have produced that.

Here each creature carries its own weights. The architecture is unchanged - the same twelve
inputs, twelve hidden units and six actions - so a population of individuals is directly
comparable to the shared-policy runs, which become the control.

The cost is that two hundred creatures mean two hundred different weight matrices, and
running them one at a time would be far too slow. Instead the whole population's weights are
held in stacked tensors and evaluated with a batched matrix multiply, which costs barely
more than the single shared network did.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from brain import Brain


class Population(nn.Module):
    """The weights of every living creature, stacked so they can be run together.

    Shapes, for one population of n creatures:
        W1  (n, inputs, hidden)      b1  (n, hidden)
        W2  (n, hidden, outputs)     b2  (n, outputs)

    Creatures are addressed by a slot index. Slots are reused as creatures die and are born,
    so the tensors stay a fixed size rather than growing and shrinking every tick.
    """

    def __init__(self, capacity: int, device: str = "cpu"):
        super().__init__()
        self.capacity = capacity
        self.device = device
        i, h, o = Brain.N_INPUTS, Brain.N_HIDDEN, Brain.N_OUTPUTS

        # Held as parameters so gradients can flow. Because each creature's loss only ever
        # touches its own slice of these tensors, one backward pass produces a separate
        # gradient for every creature, and each one is updated only by its own experience.
        # No per creature optimiser is needed, and nothing is shared between them.
        self.W1 = nn.Parameter(torch.zeros(capacity, i, h, device=device))
        self.b1 = nn.Parameter(torch.zeros(capacity, h, device=device))
        self.W2 = nn.Parameter(torch.zeros(capacity, h, o, device=device))
        self.b2 = nn.Parameter(torch.zeros(capacity, o, device=device))

        self._free = list(range(capacity))      # slots not currently in use

    # ------------------------------------------------------------------ slots

    def claim(self) -> int | None:
        """Take a free slot for a new creature. None when the population is full."""
        return self._free.pop() if self._free else None

    def release(self, slot: int) -> None:
        """Give a slot back when its creature dies."""
        self._free.append(slot)

    @property
    def in_use(self) -> int:
        return self.capacity - len(self._free)

    # ------------------------------------------------------------- filling slots

    def set_from_policy(self, slot: int, policy) -> None:
        """Copy a trained shared policy into one slot.

        This is how the founders start: each of them begins as a copy of the same
        pretrained network, exactly as the shared-policy runs do, so the two arms start
        from the same behaviour and any difference between them comes from what happens
        afterwards rather than from where they began.
        """
        with torch.no_grad():
            self.W1.data[slot] = policy.hidden.weight.T.detach().clone()
            self.b1.data[slot] = policy.hidden.bias.detach().clone()
            self.W2.data[slot] = policy.out.weight.T.detach().clone()
            self.b2.data[slot] = policy.out.bias.detach().clone()

    def copy_slot(self, source: int, target: int) -> None:
        """Copy one creature's weights into another slot, for inheritance at birth."""
        with torch.no_grad():
            self.W1.data[target] = self.W1.data[source]
            self.b1.data[target] = self.b1.data[source]
            self.W2.data[target] = self.W2.data[source]
            self.b2.data[target] = self.b2.data[source]

    def inherit(self, parent_slot: int, mutation_std: float = 0.0, rng=None) -> int | None:
        """Give a newborn a slot holding a copy of its parent's network.

        A child starts life already knowing what its parent knew, and then adapts from its
        own experience. That is the arrangement worth testing: creatures that begin alike
        and come apart because their lives differ, rather than because we scattered noise
        over them.

        mutation_std is zero by default, and deliberately so. With no noise, every
        difference that appears between two creatures was produced by something that
        happened to one of them and not the other, which is exactly the claim we want to be
        able to make. Raising it above zero mixes evolution back in and muddies that.

        Returns the new slot, or None when the population is full.
        """
        child = self.claim()
        if child is None:
            return None
        self.copy_slot(parent_slot, child)
        if mutation_std > 0:
            generator = None
            if rng is not None:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(rng.randrange(2**31))
            with torch.no_grad():
                for tensor in (self.W1, self.b1, self.W2, self.b2):
                    noise = torch.randn(
                        tensor.data[child].shape, device=self.device, generator=generator
                    )
                    tensor.data[child] += noise * mutation_std
        return child

    def founder(self, policy) -> int | None:
        """A slot for one of the starting creatures, copied from the pretrained policy.

        Every founder is the same network, which is what the shared-policy runs also start
        from. The two arms therefore begin with identical behaviour and differ only in
        whether the creatures are allowed to drift apart afterwards.
        """
        slot = self.claim()
        if slot is None:
            return None
        self.set_from_policy(slot, policy)
        return slot

    def brain_of(self, slot: int) -> Brain:
        """One creature's weights as a plain Brain, so the usual probes work on it."""
        with torch.no_grad():
            flat = torch.cat([
                self.W1.data[slot].flatten(), self.b1.data[slot],
                self.W2.data[slot].flatten(), self.b2.data[slot],
            ]).cpu().numpy()
        return Brain(flat.astype(np.float64))

    # ---------------------------------------------------------------- thinking

    def action_scores(self, slots: list[int], senses: torch.Tensor) -> torch.Tensor:
        """Score every action for every creature, each through its own network.

        senses is (n, inputs) with one row per creature in `slots`, in the same order. The
        einsum runs a different weight matrix against each row, which is what makes a
        population of distinct networks cost about the same as one shared network.
        """
        index = torch.as_tensor(slots, dtype=torch.long, device=self.device)
        W1, b1 = self.W1[index], self.b1[index]
        W2, b2 = self.W2[index], self.b2[index]
        hidden = torch.tanh(torch.einsum("ni,nih->nh", senses, W1) + b1)
        return torch.einsum("nh,nho->no", hidden, W2) + b2

    # ---------------------------------------------------------------- learning

    def forget_slot(self, optimiser, slot: int) -> None:
        """Wipe the optimiser's memory of a slot before a new creature moves into it.

        Adam keeps a running sense of each weight's recent gradients. Slots are reused when
        creatures die, so without this a newborn would inherit the momentum of whatever
        dead creature held the slot before it, and start its life being pushed in a
        direction that had nothing to do with anything it ever did. It is the same class of
        stale-state bug as reusing a nest without clearing it.
        """
        for tensor in (self.W1, self.b1, self.W2, self.b2):
            state = optimiser.state.get(tensor)
            if not state:
                continue
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"):
                buffer = state.get(key)
                if buffer is not None:
                    buffer[slot] = 0.0

    def learn(self, optimiser, trajectories: dict, gamma: float = 0.99) -> float:
        """One gradient step, giving every creature an update from its own life.

        trajectories maps a slot to the log probabilities and rewards of the choices that
        creature made since the last update.

        Returns are discounted per creature and then standardised across the whole
        population at once, rather than within each creature separately. That is
        deliberate: a creature with only a handful of steps has no meaningful spread of its
        own to normalise by, and standardising against the population makes the baseline
        "how everyone else did", so a creature is pushed towards choices that beat its
        neighbours rather than merely beat its own average. The gradient still only reaches
        its own weights.
        """
        steps, returns = [], []
        for slot, (log_probs, rewards) in trajectories.items():
            running = 0.0
            discounted = []
            for r in reversed(rewards):
                running = r + gamma * running
                discounted.append(running)
            discounted.reverse()
            steps.extend(log_probs)
            returns.extend(discounted)

        if not steps:
            return 0.0

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if returns_t.numel() > 1:
            # unbiased=False, because this is the whole window's experience rather than a
            # sample drawn from something larger, and the unbiased form is undefined for a
            # single step.
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std(unbiased=False) + 1e-8)
        # With a single step there is nothing to standardise against, so the return is used
        # as it stands. Subtracting its own mean would leave exactly zero and quietly
        # cancel the update, which looks like working code that never learns.
        loss = -(torch.stack(steps) * returns_t).mean()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        return float(loss.item())

    # -------------------------------------------------------------- inspection

    def spread(self, slots: list[int]) -> float:
        """How far apart the living creatures' weights are.

        The mean distance from each creature to the population average, which is zero when
        everyone is identical and grows as lineages drift apart. This is the number that
        says whether the population is differentiating, which is the whole reason for
        giving creatures their own networks in the first place.
        """
        if len(slots) < 2:
            return 0.0
        index = torch.as_tensor(slots, dtype=torch.long, device=self.device)
        flat = torch.cat([
            self.W1.data[index].flatten(1), self.b1.data[index],
            self.W2.data[index].flatten(1), self.b2.data[index],
        ], dim=1)
        centre = flat.mean(dim=0, keepdim=True)
        return float((flat - centre).norm(dim=1).mean())
