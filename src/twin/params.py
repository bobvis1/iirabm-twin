"""The five patient parameters (theta), the prior over them, and space-filling designs.

theta = (inj_number, oxyHeal, infectSpread, numRecurInj, numInfectRepeat); see
docs/parameter_space.md.
The prior is uniform on the box in config/prior.yaml: continuous for oxyHeal, uniform over the
integers (inclusive) for the others.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import yaml
from scipy.stats import qmc

from twin.config import CONFIG_DIR

PARAM_NAMES = ["inj_number", "oxyHeal", "infectSpread", "numRecurInj", "numInfectRepeat"]
INT_PARAMS = {"inj_number", "infectSpread", "numRecurInj", "numInfectRepeat"}
DESCRIPTIONS = {
    "inj_number": "injury size (radius of the initial injury disk)",
    "oxyHeal": "host resilience (oxygen recovery per step)",
    "infectSpread": "toxigenesis (infection added to a neighbour)",
    "numRecurInj": "environmental toxicity (recurrent injuries per cycle)",
    "numInfectRepeat": "microbial invasiveness (spread attempts per step)",
}


@dataclass(frozen=True)
class Theta:
    inj_number: int
    oxyHeal: float
    infectSpread: int
    numRecurInj: int
    numInfectRepeat: int

    def to_engine(self) -> dict:
        """The engine-config fields for this theta."""
        return asdict(self)

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in PARAM_NAMES], dtype=float)

    @classmethod
    def from_array(cls, a) -> Theta:
        return cls(
            inj_number=int(a[0]),
            oxyHeal=float(a[1]),
            infectSpread=int(a[2]),
            numRecurInj=int(a[3]),
            numInfectRepeat=int(a[4]),
        )


@dataclass(frozen=True)
class Prior:
    """Box prior. Integers take every value in [lo, hi]; oxyHeal is uniform on [lo, hi)."""

    bounds: dict[str, tuple[float, float]]

    @classmethod
    def load(cls, path=None) -> Prior:
        path = path or CONFIG_DIR / "prior.yaml"
        raw = yaml.safe_load(open(path))["bounds"]
        return cls({n: (raw[n][0], raw[n][1]) for n in PARAM_NAMES})

    def from_unit(self, u: np.ndarray) -> list[Theta]:
        """Map points in the unit cube (n x 5) to thetas (integers by flooring)."""
        out = []
        for row in np.atleast_2d(u):
            vals = []
            for j, name in enumerate(PARAM_NAMES):
                lo, hi = self.bounds[name]
                if name in INT_PARAMS:
                    k = int(hi) - int(lo) + 1
                    vals.append(int(lo) + min(int(row[j] * k), k - 1))
                else:
                    vals.append(lo + row[j] * (hi - lo))
            out.append(Theta.from_array(vals))
        return out

    def sample(self, rng: np.random.Generator, n: int) -> list[Theta]:
        """n independent draws from the prior (continuous oxyHeal, so never a library point)."""
        return self.from_unit(rng.random((n, len(PARAM_NAMES))))

    def sobol(self, n: int, scramble_seed: int, start: int = 0) -> list[Theta]:
        """Points start..start+n-1 of a scrambled Sobol sequence mapped to the prior.

        Extending a design means calling again with a larger `start`: the earlier points are
        unchanged. n and start need not be powers of two (balance is only guaranteed for those).
        """
        sampler = qmc.Sobol(d=len(PARAM_NAMES), scramble=True, seed=scramble_seed)
        if start:
            sampler.fast_forward(start)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The balance properties")  # documented above
            u = sampler.random(n)
        return self.from_unit(u)

    def contains(self, theta: Theta) -> bool:
        for name in PARAM_NAMES:
            lo, hi = self.bounds[name]
            v = getattr(theta, name)
            if name in INT_PARAMS and not (lo <= v <= hi):
                return False
            if name not in INT_PARAMS and not (lo <= v < hi):
                return False
        return True
