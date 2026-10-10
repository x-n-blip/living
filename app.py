#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

NODES = ["housing", "demand", "capacity", "inventory_pressure", "labor_pressure",
         "labor_cost", "availability", "customer_exp", "price_power", "competition"]
IDX = {n: i for i, n in enumerate(NODES)}
N = len(NODES)

EDGES = [
    ("housing", "demand", 0.30, "housing -> demand"),
    ("customer_exp", "demand", 0.25, "customer_exp -> demand"),
    ("competition", "demand", -0.15, "competition -> demand"),
    ("price_power", "demand", -0.05, "price_power -> demand"),
    ("demand", "capacity", 0.20, "demand -> capacity"),
    ("labor_cost", "capacity", -0.10, "labor_cost -> capacity"),
    ("demand", "inventory_pressure", -0.35, "demand -> inventory_pressure"),
    ("capacity", "inventory_pressure", 0.12, "capacity -> inventory_pressure"),
    ("demand", "labor_pressure", 0.25, "demand -> labor_pressure"),
    ("capacity", "labor_pressure", 0.15, "capacity -> labor_pressure"),
    ("labor_pressure", "labor_cost", 0.50, "labor_pressure -> labor_cost"),
    ("availability", "customer_exp", 0.35, "availability -> customer_exp"),
    ("capacity", "customer_exp", 0.20, "capacity -> customer_exp"),
    ("inventory_pressure", "price_power", -0.25, "inventory_pressure -> price_power"),
    ("competition", "price_power", -0.30, "competition -> price_power"),
    ("inventory_pressure", "availability", 0.10, "inventory_pressure -> availability"),
]
W = np.zeros((N, N))
for s, d, w, _ in EDGES:
    W[IDX[d], IDX[s]] = w

DECAY = 0.75
SIGMA = np.array([0.010, 0.025, 0.012, 0.015, 0.015, 0.012, 0.015, 0.015, 0.020, 0.012])

LABELS = {
    "housing": ("Soft", "Neutral", "Strong"),
    "demand": ("Weak", "Normal", "Strong"),
    "capacity": ("Contracting", "Normal", "Expanding"),
    "inventory_pressure": ("Tight", "Balanced", "Excess"),
    "labor_pressure": ("Easing", "Stable", "Rising"),
    "labor_cost": ("Falling", "Stable", "Rising"),
    "availability": ("Constrained", "Normal", "Abundant"),
    "customer_exp": ("Worsening", "Stable", "Improving"),
    "price_power": ("Weak", "Stable", "Strong"),
    "competition": ("Easing", "Stable", "Intensifying"),
}

def calc_revenue_index(x: np.ndarray) -> np.ndarray:
    d, cap = x[..., IDX["demand"]], x[..., IDX["capacity"]]
    pp, av = x[..., IDX["price_power"]], x[..., IDX["availability"]]
    served = np.minimum(d, cap + 0.15)
    return np.maximum(0.5, (1 + 0.6 * served) * (1 + 0.15 * pp) * (1 + 0.10 * av))

def calc_margin(x: np.ndarray, rev_idx: np.ndarray, base: float = 0.135) -> np.ndarray:
    d, cap = x[..., IDX["demand"]], x[..., IDX["capacity"]]
    m = (base + 0.04 * x[..., IDX["price_power"]] - 0.06 * x[..., IDX["labor_cost"]]
         - 0.02 * np.maximum(x[..., IDX["inventory_pressure"]], 0)
         - 0.03 * np.maximum(cap - d, 0)
         + 0.15 * (rev_idx - 1))
    return np.clip(m, 0.02, 0.25)

@dataclass
class Hypothesis:
    name: str
    prob: float
    effects: dict

@dataclass
class Signal:
    sid: str
    kind: str
    text: str
    source: str
    obs_date: date
    confidence: float
    half_life: float
    hypotheses: list
    strength: float = 1.0
    confirmations: int = 0

    def relevance(self, on: date) -> float:
        age = max((on - self.obs_date).days, 0)
        return 0.5 ** (age / self.half_life)

    def outlook_score(self) -> float:
        sc = 0.0
        for h in self.hypotheses:
            e = h.effects
            sc += h.prob * (e.get("demand", 0) + e.get("customer_exp", 0) + 0.5 * e.get("capacity", 0)
                            + 0.5 * e.get("availability", 0) + 0.5 * e.get("price_power", 0)
                            - 0.5 * e.get("inventory_pressure", 0) - 0.5 * e.get("labor_cost", 0))
        return sc

def make_hyp(*items):
    return [Hypothesis(n, p, e) for n, p, e in items]

TEMPLATES = {
    "distribution_hiring": (45, lambda: make_hyp(
        ("capacity_expansion", .50, dict(capacity=.06, labor_pressure=.03, labor_cost=.015)),
        ("anticipating_demand", .20, dict(demand=.03, capacity=.05, labor_pressure=.03)),
        ("new_segment_launch", .15, dict(capacity=.03, labor_cost=.015)),
        ("turnover_backfill", .15, dict(labor_pressure=.04, labor_cost=.02)))),
    "expansion_guidance": (120, lambda: make_hyp(
        ("footprint_growth", .60, dict(capacity=.05, demand=.015)),
        ("cautious_talk_only", .25, dict()),
        ("capex_pull_forward", .15, dict(capacity=.03, labor_cost=.01)))),
    "delivery_complaints": (30, lambda: make_hyp(
        ("fulfillment_bottleneck", .45, dict(customer_exp=-.06, availability=-.02)),
        ("supplier_delays", .30, dict(availability=-.05, customer_exp=-.04)),
        ("seasonal_noise", .25, dict(customer_exp=-.01)))),
    "positive_reviews": (30, lambda: make_hyp(
        ("better_delivery", .35, dict(customer_exp=.05, capacity=.01)),
        ("better_availability", .25, dict(availability=.04, customer_exp=.04)),
        ("discount_driven", .25, dict(demand=.03, price_power=-.05, customer_exp=.02)),
        ("service_improvement", .15, dict(customer_exp=.05)))),
    "supply_constraint": (60, lambda: make_hyp(
        ("supplier_shortage", .60, dict(availability=-.07, inventory_pressure=-.03, price_power=.02)),
        ("transient_disruption", .40, dict(availability=-.02)))),
    "supply_improving": (60, lambda: make_hyp(
        ("supply_recovery", .70, dict(availability=.06, inventory_pressure=.02)),
        ("temporary_restock", .30, dict(availability=.02)))),
    "pro_demand_strong": (90, lambda: make_hyp(
        ("true_demand_up", .60, dict(demand=.04, housing=.02)),
        ("project_pull_forward", .25, dict(demand=.03)),
        ("anecdotal_noise", .15, dict()))),
    "housing_weak": (120, lambda: make_hyp(
        ("macro_housing_slowdown", .70, dict(housing=-.05, demand=-.02)),
        ("local_only", .30, dict(housing=-.015)))),
    "price_increase": (60, lambda: make_hyp(
        ("real_pricing_power", .35, dict(price_power=.05, demand=-.005)),
        ("cost_passthrough", .45, dict(price_power=.015, demand=-.015)),
        ("mix_shift", .20, dict(price_power=.02)))),
    "inventory_build": (60, lambda: make_hyp(
        ("demand_weakness", .40, dict(inventory_pressure=.06, demand=-.02)),
        ("strategic_prebuild", .40, dict(inventory_pressure=.03, availability=.03)),
        ("supplier_pull_forward", .20, dict(inventory_pressure=.04)))),
    "competitor_promo": (45, lambda: make_hyp(
        ("share_battle", .70, dict(competition=.06, price_power=-.03)),
        ("noise", .30, dict()))),
}

_ANY = r"[^.;。]{0,60}"
RULES = [
    ("distribution_hiring", rf"(distribution|logistics|supply[- ]chain|warehouse|fulfil+ment){_ANY}(hiring|hire|job|opening|posting)|(hiring|hire|job|opening|posting){_ANY}(distribution|logistics|supply[- ]chain|warehouse)"),
    ("expansion_guidance", r"(expand|expansion|new (distribution )?centers?|footprint).{0,40}(distribution|footprint|stores?|capacity)"),
    ("delivery_complaints", rf"(delay|late deliver|complain|worse|worsen|declin|negative){_ANY}(deliver|review|customer|service)|(deliver|review|customer){_ANY}(delay|complain|worse|worsen|declin|negative)"),
    ("positive_reviews", rf"(review|rating|customer){_ANY}(improv|positive|better|faster|great)"),
    ("supply_constraint", r"(shortage|lead times? (lengthen|increas)|supply constraint|stock.?out)"),
    ("supply_improving", r"(supply (improv|recover|eas)|lead times? (shorten|improv))"),
    ("pro_demand_strong", r"(\bpro\b|professional|contractor){_ANY}(demand|strengthen|backlog|strong)".replace("{_ANY}", _ANY)),
    ("housing_weak", r"(housing|mortgage|home sales){_ANY}(soft|weak|slow|elevated|decline)".replace("{_ANY}", _ANY)),
    ("price_increase", r"(price (increase|hike)|raised prices)"),
    ("inventory_build", r"(inventor(y|ies)){_ANY}(build|up |higher|elevated|rose)".replace("{_ANY}", _ANY)),
    ("competitor_promo", r"(competitor|rival){_ANY}(promo|discount|price cut)".replace("{_ANY}", _ANY)),
]
SOURCE_CONF = {"filing": .85, "call": .75, "news": .60, "job_posting": .60, "review": .55}

class SignalExtractor:
    def __init__(self, llm_fn=None):
        self.llm_fn = llm_fn

    def extract(self, text: str, source: str, obs_date: date) -> list[Signal]:
        if self.llm_fn:
            hits = [(h["kind"], h.get("confidence", SOURCE_CONF.get(source, .6))) for h in self.llm_fn(text)]
        else:
            hits = [(k, SOURCE_CONF.get(source, .6)) for k, rx in RULES if re.search(rx, text, re.I)]
        out = []
        for kind, conf in hits:
            hl, mk = TEMPLATES[kind]
            out.append(Signal(sid=kind, kind=kind, text=text, source=source, obs_date=obs_date,
                            confidence=conf, half_life=hl, hypotheses=mk()))
        return out

class SignalStore:
    def __init__(self):
        self.signals = {}

    def add(self, sig: Signal):
        if sig.sid not in self.signals:
            self.signals[sig.sid] = sig
        else:
            old = self.signals[sig.sid]
            old.confirmations += 1
            old.obs_date = max(old.obs_date, sig.obs_date)
            old.confidence = min(.95, old.confidence + .08)
            old.strength = min(1.5, old.strength + .10)
            old.text += f" || {sig.text}"

    def all(self):
        return list(self.signals.values())

def find_conflicts(signals, on, min_rel=.25, min_score=.01):
    live = [s for s in signals if s.relevance(on) >= min_rel and abs(s.outlook_score()) >= min_score]
    pos = [s for s in live if s.outlook_score() > 0]
    neg = [s for s in live if s.outlook_score() < 0]
    pairs = [(p.sid, n.sid) for p in pos for n in neg]
    idx = min(1.0, len(pairs) / 6) if pairs else 0.0
    return pairs, idx

def persistent_shock(node, level, T=12):
    return {"init": {node: level}, "shocks": [(node, level * (1 - DECAY), 0, T)]}

SCENARIOS = {
    "Baseline": {"desc": "Normal run"},
    "Strong demand": {"shocks": [("demand", .03, 0, 9), ("housing", .02, 0, 9)]},
    "Strong demand + cost pressure": {"shocks": [("demand", .03, 0, 9), ("labor_pressure", .04, 0, 9), ("labor_cost", .02, 0, 9)]},
    "Weak demand": {"shocks": [("demand", -.04, 0, 9), ("housing", -.02, 0, 9)]},
    "Supply constraint": {"shocks": [("demand", .02, 0, 9), ("availability", -.06, 0, 9)]},
}

COUNTERFACTUALS = {
    "Demand -10pct": persistent_shock("demand", -.10),
    "Labor cost up fast": {"shocks": [("labor_cost", .02, 0, 9), ("labor_pressure", .02, 0, 9)]},
    "Supply fix": persistent_shock("availability", .08),
    "Pricing power down": persistent_shock("price_power", -.10),
    "Capacity delay": {"shocks": [("capacity", -.04, 0, 6)]},
}

@dataclass
class Result:
    name: str
    X: np.ndarray
    rev: np.ndarray
    margin: np.ndarray
    wmult: np.ndarray
    conflict: float

    @property
    def rev12(self): return self.rev.sum(1)
    @property
    def op12(self): return (self.rev * self.margin).sum(1)
    @property
    def end_margin(self): return self.margin[:, -3:].mean(1)

class Simulator:
    def __init__(self, company, signals, as_of, base_rev=13.0, horizon=12, init_state=None):
        self.company, self.signals, self.as_of = company, signals, as_of
        self.base, self.T = base_rev, horizon
        self.init_state = init_state or {}

    def get_draws(self, n, seed=7):
        r = np.random.default_rng(seed)
        S = max(len(self.signals), 1)
        return dict(n=n, U=r.random((n, S)), mag=np.exp(.25 * r.standard_normal((n, S))),
                    wmult=np.exp(.25 * r.standard_normal((n, N, N))),
                    eps=r.standard_normal((n, self.T, N)))

    @staticmethod
    def _calc_probs(sig, override):
        p = np.array([h.prob for h in sig.hypotheses], float)
        p /= p.sum()
        if override:
            fixed = {i: override[h.name] for i, h in enumerate(sig.hypotheses) if h.name in override}
            rest = [i for i in range(len(p)) if i not in fixed]
            left = max(0.0, 1 - sum(fixed.values()))
            q = np.zeros_like(p)
            for i, v in fixed.items(): q[i] = v
            if rest:
                base = p[rest] / p[rest].sum() if p[rest].sum() > 0 else np.ones(len(rest)) / len(rest)
                q[rest] = left * base
            p = q / q.sum()
        return p

    def run(self, draws, scenario=None, name="run", overrides=None, disabled=()):
        n, T = draws["n"], self.T
        scn, overrides = scenario or {}, overrides or {}
        init = np.zeros(N)
        for k, v in {**self.init_state}.items(): init[IDX[k]] += v
        for k, v in scn.get("init", {}).items(): init[IDX[k]] += v

        shock = np.zeros((T, N))
        for node, f, s, e in scn.get("shocks", []):
            shock[s:min(e, T), IDX[node]] += f

        pairs, cidx = find_conflicts([s for s in self.signals if s.sid not in disabled], self.as_of)
        noise_mult = 1 + 0.5 * cidx

        forcing = []
        for si, sg in enumerate(self.signals):
            if sg.sid in disabled: continue
            p = self._calc_probs(sg, overrides.get(sg.sid))
            hi = np.minimum(np.searchsorted(np.cumsum(p), draws["U"][:, si]), len(p) - 1)
            E = np.array([[h.effects.get(nd, 0.0) for nd in NODES] for h in sg.hypotheses])
            imp = E[hi] * sg.strength * sg.confidence * draws["mag"][:, si, None]
            rel = np.array([sg.relevance(self.as_of + timedelta(days=30 * m)) for m in range(T)])
            forcing.append((imp, rel))

        Wp = W[None] * draws["wmult"]
        x = np.tile(init, (n, 1))
        X = np.zeros((n, T + 1, N))
        X[:, 0] = x
        for m in range(T):
            f = np.zeros((n, N))
            for imp, rel in forcing: f += imp * rel[m]
            x = (DECAY * x + np.einsum("nij,nj->ni", Wp, x) + f + shock[m] + noise_mult * SIGMA * draws["eps"][:, m])
            x = np.clip(x, -1.5, 1.5)
            X[:, m + 1] = x

        ri = calc_revenue_index(X[:, 1:])
        return Result(name, X, self.base * ri, calc_margin(X[:, 1:], ri), draws["wmult"], cidx)

def summarize_result(res, runrate):
    q = lambda a, p: float(np.percentile(a, p))
    return {
        "Rev Mean ($B)": res.rev12.mean(), "Rev P5": q(res.rev12, 5), "Rev P50": q(res.rev12, 50),
        "Rev P95": q(res.rev12, 95), "P(Rev > RunRate)": float((res.rev12 > runrate).mean()),
        "Margin Mean": res.end_margin.mean(), "Margin P5": q(res.end_margin, 5),
        "Margin P95": q(res.end_margin, 95), "OP Mean ($B)": res.op12.mean(),
    }

def make_table(results, runrate):
    return pd.DataFrame({r.name: summarize_result(r, runrate) for r in results}).T

def spearman_corr(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    if ra.std() == 0 or rb.std() == 0: return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])

def get_sensitivity(res):
    avg = res.X[:, 1:].mean(1)
    nodes = pd.DataFrame({"Driver": NODES,
                          "Rev Corr": [spearman_corr(avg[:, i], res.rev12) for i in range(N)],
                          "Margin Corr": [spearman_corr(avg[:, i], res.end_margin) for i in range(N)]})
    nodes["Importance"] = nodes[["Rev Corr", "Margin Corr"]].abs().max(axis=1)
    return nodes.sort_values("Importance", ascending=False)

def explain_sig(sg, on):
    lines = [f"[{sg.sid}] Text: \"{sg.text[:80]}\" (Rel: {sg.relevance(on):.2f})"]
    for h in sg.hypotheses:
        eff_str = ", ".join(f"{k}{v:+.3f}" for k, v in h.effects.items()) or "none"
        lines.append(f"   -> {h.prob:.0%} {h.name}: {eff_str}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=1000)
    a = ap.parse_args()

    as_of = date(2026, 10, 9)
    store = SignalStore()
    ex = SignalExtractor()
    
    docs = [
        ("Posted 140 new distribution center and logistics job openings across the Southeast", "job_posting", 5),
        ("Hiring is increasingly focused on distribution operations (warehouse job postings up)", "job_posting", 40),
        ("Management: we expect to expand our distribution footprint next year", "call", 20),
        ("Customers still report delivery delays and complain about service", "review", 3),
    ]
    for text, src, ago in docs:
        for sg in ex.extract(text, src, as_of - timedelta(days=ago)):
            store.add(sg)

    sigs = store.all()
    sim = Simulator("Retailer A", sigs, as_of, base_rev=13.0, init_state={"demand": .03})
    runrate = sim.base * 12
    draws = sim.get_draws(a.paths)

    print("Running Living Company simulation demo...")
    base = sim.run(draws, SCENARIOS["Baseline"], "Baseline")
    
    print("\n--- Signals Parsed ---")
    for s in sigs:
        print(explain_sig(s, as_of))

    print("\n--- Scenarios ---")
    res = [sim.run(draws, s, nm) for nm, s in SCENARIOS.items()]
    print(make_table(res, runrate).round(3).to_string())

    print("\n--- Counterfactuals (delta vs baseline) ---")
    cf = [sim.run(draws, s, nm) for nm, s in COUNTERFACTUALS.items()]
    for r in cf:
        print(f"{r.name:<22} dRev={r.rev12.mean()-base.rev12.mean():+.2f}B  "
              f"dMargin={(r.end_margin.mean()-base.end_margin.mean())*100:+.2f}pp")

    print("\n--- Audit trail (signal ablation) ---")
    for sg in sigs:
        r = sim.run(draws, SCENARIOS["Baseline"], "x", disabled=(sg.sid,))
        print(f"{sg.sid:<22} contributes {base.rev12.mean()-r.rev12.mean():+.2f}B revenue")

    print("\n--- Investor override: hiring = turnover backfill ---")
    ov = {"distribution_hiring": {"turnover_backfill": .70, "capacity_expansion": .10}}
    mine = sim.run(draws, SCENARIOS["Baseline"], "mine", overrides=ov)
    print(f"Rev {base.rev12.mean():.1f} -> {mine.rev12.mean():.1f}")

    print("\n--- Top drivers ---")
    print(get_sensitivity(base).head(5).round(3).to_string(index=False))

if __name__ == "__main__":
    main()
