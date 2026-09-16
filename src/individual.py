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

from brain import Brain


class Population:
    """The weights of every living creature, stacked so they can be run together.

    Shapes, for one population of n creatures:
        W1  (n, inputs, hidden)      b1  (n, hidden)
        W2  (n, hidden, outputs)     b2  (n, outputs)

    Creatures are addressed by a slot index. Slots are reused as creatures die and are born,
    so the tensors stay a fixed size rather than growing and shrinking every tick.
    """

    def __init__(self, capacity: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        i, h, o = Brain.N_INPUTS, Brain.N_HIDDEN, Brain.N_OUTPUTS

        self.W1 = torch.zeros(capacity, i, h, device=device)
        self.b1 = torch.zeros(capacity, h, device=device)
        self.W2 = torch.zeros(capacity, h, o, device=device)
        self.b2 = torch.zeros(capacity, o, device=device)

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
            self.W1[slot] = policy.hidden.weight.T.detach().clone()
            self.b1[slot] = policy.hidden.bias.detach().clone()
            self.W2[slot] = policy.out.weight.T.detach().clone()
            self.b2[slot] = policy.out.bias.detach().clone()

    def copy_slot(self, source: int, target: int) -> None:
        """Copy one creature's weights into another slot, for inheritance at birth."""
        with torch.no_grad():
            self.W1[target] = self.W1[source]
            self.b1[target] = self.b1[source]
            self.W2[target] = self.W2[source]
            self.b2[target] = self.b2[source]

    def brain_of(self, slot: int) -> Brain:
        """One creature's weights as a plain Brain, so the usual probes work on it."""
        with torch.no_grad():
            flat = torch.cat([
                self.W1[slot].flatten(), self.b1[slot],
                self.W2[slot].flatten(), self.b2[slot],
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
            self.W1[index].flatten(1), self.b1[index],
            self.W2[index].flatten(1), self.b2[index],
        ], dim=1)
        centre = flat.mean(dim=0, keepdim=True)
        return float((flat - centre).norm(dim=1).mean())
