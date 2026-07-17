# Claim 2 - Exact sufficient optimum


---
<!-- trackio-cell
{"type": "code", "id": "cell_eefdea7b6628", "created_at": "2026-07-17T04:15:17+00:00", "title": "Official geometry plus direct optimizer", "command": ["python", "repro/src/run_fair_decisions.py", "--output", "outputs/summary.json"], "exit_code": 0, "duration_s": 23.658}
-->
````bash
$ python repro/src/run_fair_decisions.py --output outputs/summary.json
````

exit 0 · 23.7s


````python title=run_fair_decisions.py
#!/usr/bin/env python3
"""Independent audit of exact sufficient classification on finite scores."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "official" / "experiments"))
from boundary_trace import GroupScoreDistribution, trace_intersection


def stats(rule, dist):
    s, w, t = dist.s_input, dist.w_input, np.asarray(rule)
    tp = np.sum(w * s * t); fp = np.sum(w * (1-s) * t)
    fn = np.sum(w * s * (1-t)); tn = np.sum(w * (1-s) * (1-t))
    pos, neg = tp + fp, fn + tn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "ppv": tp/pos, "for": fn/neg, "acc": tp+tn,
            "tpr": tp/(tp+fn), "fpr": fp/(fp+tn)}


def aggregate(rules, dists, prob_a1=.5):
    st = [stats(r, d) for r, d in zip(rules, dists)]
    wa = np.array([1-prob_a1, prob_a1])
    acc = sum(wa[a]*st[a]["acc"] for a in range(2))
    # Direct definition: E_{A,Y} TV(P(R|A,Y),P(R|Y)).
    pis = np.array([d.pi for d in dists]); pi = wa @ pis
    tpr = sum(wa[a]*pis[a]*st[a]["tpr"] for a in range(2))/pi
    fpr = sum(wa[a]*(1-pis[a])*st[a]["fpr"] for a in range(2))/(1-pi)
    dsep = sum(wa[a]*(pis[a]*abs(st[a]["tpr"]-tpr) +
                       (1-pis[a])*abs(st[a]["fpr"]-fpr)) for a in range(2))
    return st, float(acc), float(dsep)


def equality_constraints(x, dists):
    a = len(dists[0].s_input)
    s0, s1 = stats(x[:a], dists[0]), stats(x[a:], dists[1])
    # Cross-products avoid ratio singularities.
    c_ppv = s0["tp"]*(s1["tp"]+s1["fp"]) - s1["tp"]*(s0["tp"]+s0["fp"])
    c_for = s0["fn"]*(s1["fn"]+s1["tn"]) - s1["fn"]*(s0["fn"]+s0["tn"])
    return np.array([c_ppv, c_for])


def direct_optimize(dists, prob_a1, objective, official_rules, seed):
    """Multi-start SLSQP over every randomized decision probability."""
    n = sum(len(d.s_input) for d in dists)
    rng = np.random.default_rng(seed)
    starts = [np.concatenate(official_rules)]
    starts += [rng.uniform(.02, .98, n) for _ in range(15)]
    cons = {"type": "eq", "fun": lambda x: equality_constraints(x, dists)}
    best = None
    feasible = 0
    for x0 in starts:
        def f(x):
            cut = len(dists[0].s_input)
            _, acc, dsep = aggregate([x[:cut], x[cut:]], dists, prob_a1)
            return -acc if objective == "accuracy" else dsep
        res = minimize(f, x0, method="SLSQP", bounds=[(1e-9, 1-1e-9)]*n,
                       constraints=cons, options={"ftol": 1e-12, "maxiter": 1000})
        violation = np.max(np.abs(equality_constraints(res.x, dists)))
        if res.success and violation < 1e-8:
            feasible += 1
            if best is None or res.fun < best.fun:
                best = res
    if best is None:
        raise RuntimeError("independent optimizer found no feasible point")
    cut = len(dists[0].s_input)
    st, acc, dsep = aggregate([best.x[:cut], best.x[cut:]], dists, prob_a1)
    return {"feasible_starts": feasible, "accuracy": acc, "dsep": dsep,
            "max_crossproduct_violation": float(np.max(np.abs(equality_constraints(best.x, dists)))),
            "ppv_gap": float(abs(st[0]["ppv"]-st[1]["ppv"])),
            "for_gap": float(abs(st[0]["for"]-st[1]["for"]))}


def paper_instances():
    d0 = GroupScoreDistribution([.1,.2,.5,.7,.9], [.1,.3,.3,.15,.15], name="Synth 0")
    d1a = GroupScoreDistribution([.12,.3,.85], [.15,.45,.4], name="Synth 1a")
    d1b = GroupScoreDistribution([.2,.3,.75], [.3,.5,.2], name="Synth 1b")
    return [("A", [d0,d1a]), ("B", [d0,d1b])]


def main(output):
    rows = []
    for case, dists in paper_instances():
        result = trace_intersection(*dists, .5)
        case_row = {"case": case}
        for name, optimum in [("accuracy", result.max_acc), ("separation", result.min_dsep)]:
            rules = [d.selection_rule(optimum.p, optimum.q) for d in dists]
            st, acc, dsep = aggregate(rules, dists, .5)
            direct = direct_optimize(dists, .5, name, rules, 100 + len(rows)*2 + (name=="separation"))
            official_value = acc if name == "accuracy" else dsep
            direct_value = direct["accuracy"] if name == "accuracy" else direct["dsep"]
            case_row[name] = {
                "p": optimum.p, "q": optimum.q, "accuracy": acc, "dsep": dsep,
                "ppv_gap": abs(st[0]["ppv"]-st[1]["ppv"]),
                "for_gap": abs(st[0]["for"]-st[1]["for"]),
                "official_reported_value": optimum.value,
                "direct_optimizer": direct,
                "objective_gap_vs_direct": abs(official_value-direct_value),
                "rules": [r.tolist() for r in rules],
            }
        rows.append(case_row)

    # Claim 1: these are true conditional class probabilities by construction.
    dists = paper_instances()[0][1]
    threshold_rules = [(d.s_input >= .5).astype(float) for d in dists]
    th_stats, th_acc, _ = aggregate(threshold_rules, dists)

    # Exhaust every pair of deterministic rules: randomization is required for
    # the nontrivial exact optimum in case A.
    gaps = []
    for bits0 in itertools.product((0.,1.), repeat=len(dists[0].s_input)):
        for bits1 in itertools.product((0.,1.), repeat=len(dists[1].s_input)):
            if sum(bits0) in (0,len(bits0)) or sum(bits1) in (0,len(bits1)):
                continue
            st, _, _ = aggregate([bits0,bits1], dists)
            gaps.append(max(abs(st[0]["ppv"]-st[1]["ppv"]), abs(st[0]["for"]-st[1]["for"])))

    summary = {
        "paper": "XiD5RhcEDK",
        "claim_1": {"scores_are_exact_class_probabilities": True,
                    "threshold": .5, "accuracy": th_acc,
                    "ppv_gap": abs(th_stats[0]["ppv"]-th_stats[1]["ppv"]),
                    "for_gap": abs(th_stats[0]["for"]-th_stats[1]["for"])},
        "claim_2_and_3": rows,
        "negative_controls": {
            "minimum_predictive_parity_gap_over_nonconstant_deterministic_pairs": min(gaps),
            "deterministic_pairs_checked": len(gaps),
            "thresholding_rejected": max(abs(th_stats[0]["ppv"]-th_stats[1]["ppv"]),
                                           abs(th_stats[0]["for"]-th_stats[1]["for"])) > .01,
        },
    }
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    json_default = lambda x: x.item() if isinstance(x, np.generic) else str(x)
    output.write_text(json.dumps(summary, indent=2, default=json_default)+"\n")
    print(json.dumps({
        "threshold_gaps": [summary["claim_1"]["ppv_gap"], summary["claim_1"]["for_gap"]],
        "cases": [{"case": r["case"],
                   "max_acc": r["accuracy"]["accuracy"],
                   "acc_gap_vs_direct": r["accuracy"]["objective_gap_vs_direct"],
                   "min_dsep": r["separation"]["dsep"],
                   "dsep_gap_vs_direct": r["separation"]["objective_gap_vs_direct"]} for r in rows],
        "deterministic_pairs": len(gaps), "min_deterministic_gap": min(gaps),
    }, indent=2, default=json_default))
    return summary


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--output", default="outputs/summary.json")
    main(p.parse_args().output)

````


````json title=summary.json
{
  "paper": "XiD5RhcEDK",
  "claim_1": {
    "scores_are_exact_class_probabilities": true,
    "threshold": 0.5,
    "accuracy": 0.7535000000000001,
    "ppv_gap": 0.19999999999999996,
    "for_gap": 0.07999999999999999
  },
  "claim_2_and_3": [
    {
      "case": "A",
      "accuracy": {
        "p": 0.8,
        "q": 0.3142857142857143,
        "accuracy": 0.7238823529411765,
        "dsep": 0.026824195825048115,
        "ppv_gap": 0.0,
        "for_gap": 5.551115123125783e-17,
        "official_reported_value": 0.7238823529411764,
        "direct_optimizer": {
          "feasible_starts": 13,
          "accuracy": 0.7238823524804844,
          "dsep": 0.026824195933336688,
          "max_crossproduct_violation": 8.68749516769185e-15,
          "ppv_gap": 4.39648317751562e-14,
          "for_gap": 1.965094753586527e-14
        },
        "objective_gap_vs_direct": 4.60692151094122e-10,
        "rules": [
          [
            0.0,
            0.0,
            3.3306690738754696e-16,
            1.0,
            1.0
          ],
          [
            0.051532377656945086,
            0.051532377656945086,
            0.8425543746910531
          ]
        ]
      },
      "separation": {
        "p": 0.8,
        "q": 0.3142857142857143,
        "accuracy": 0.7238823529411765,
        "dsep": 0.026824195825048115,
        "ppv_gap": 0.0,
        "for_gap": 5.551115123125783e-17,
        "official_reported_value": 0.026824195825048087,
        "direct_optimizer": {
          "feasible_starts": 16,
          "accuracy": 0.7238823524804715,
          "dsep": 0.026824195933331206,
          "max_crossproduct_violation": 2.040034807748725e-15,
          "ppv_gap": 1.84297022087776e-14,
          "for_gap": 4.274358644806853e-15
        },
        "objective_gap_vs_direct": 1.0828309082722143e-10,
        "rules": [
          [
            0.0,
            0.0,
            3.3306690738754696e-16,
            1.0,
            1.0
          ],
          [
            0.051532377656945086,
            0.051532377656945086,
            0.8425543746910531
          ]
        ]
      }
    },
    {
      "case": "B",
      "accuracy": {
        "p": 0.6975576820700883,
        "q": 0.26122115896495574,
        "accuracy": 0.7247235512938053,
        "dsep": 0.09432395514407416,
        "ppv_gap": 1.1102230246251565e-16,
        "for_gap": 1.6653345369377348e-16,
        "official_reported_value": 0.7247235512938052,
        "direct_optimizer": {
          "feasible_starts": 15,
          "accuracy": 0.7247235509306595,
          "dsep": 0.09432395540827249,
          "max_crossproduct_violation": 2.1048440768112187e-13,
          "ppv_gap": 2.014166611274959e-12,
          "for_gap": 4.997668945350142e-13
        },
        "objective_gap_vs_direct": 3.631457357045065e-10,
        "rules": [
          [
            0.0,
            0.0,
            0.5185438341677227,
            1.0,
            1.0
          ],
          [
            0.0,
            0.05276448706194681,
            1.0
          ]
        ]
      },
      "separation": {
        "p": 0.75,
        "q": 0.29687499999999994,
        "accuracy": 0.7148275862068966,
        "dsep": 0.09279981753645708,
        "ppv_gap": 2.220446049250313e-16,
        "for_gap": 5.551115123125783e-17,
        "official_reported_value": 0.09279981753645695,
        "direct_optimizer": {
          "feasible_starts": 10,
          "accuracy": 0.7148275861237827,
          "dsep": 0.09279981801800892,
          "max_crossproduct_violation": 6.494804694057166e-15,
          "ppv_gap": 8.215650382226158e-15,
          "for_gap": 1.176836406102666e-14
        },
        "objective_gap_vs_direct": 4.815518400702601e-10,
        "rules": [
          [
            0.0,
            0.0,
            0.20000000000000073,
            1.0,
            1.0
          ],
          [
            0.0,
            0.0,
            0.696551724137932
          ]
        ]
      }
    }
  ],
  "negative_controls": {
    "minimum_predictive_parity_gap_over_nonconstant_deterministic_pairs": 0.018181818181818132,
    "deterministic_pairs_checked": 180,
    "thresholding_rejected": true
  }
}

````


````output
{
  "threshold_gaps": [
    0.19999999999999996,
    0.07999999999999999
  ],
  "cases": [
    {
      "case": "A",
      "max_acc": 0.7238823529411765,
      "acc_gap_vs_direct": 4.60692151094122e-10,
      "min_dsep": 0.026824195825048115,
      "dsep_gap_vs_direct": 1.0828309082722143e-10
    },
    {
      "case": "B",
      "max_acc": 0.7247235512938053,
      "acc_gap_vs_direct": 3.631457357045065e-10,
      "min_dsep": 0.09279981753645708,
      "dsep_gap_vs_direct": 4.815518400702601e-10
    }
  ],
  "deterministic_pairs": 180,
  "min_deterministic_gap": 0.018181818181818132
}

````


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_091a22de04ff", "created_at": "2026-07-17T04:15:18+00:00", "title": "For both paper synthetic instances, Algorithm 1 produces PPV and FOR gaps below…"}
-->
For both paper synthetic instances, Algorithm 1 produces PPV and FOR gaps below 1e-12. Accuracy optima are 0.723882 and 0.724724; independent 16-start constrained optimization agrees within 4.7e-10.
