# Repro — Fair Decisions from Calibrated Scores (ICML 2026)

Exact finite-score reproduction of [arXiv:2602.07285](https://arxiv.org/abs/2602.07285)
(OpenReview `XiD5RhcEDK`).

The paper's two synthetic instances confirm all three scored claims. A 0.5
threshold applied to exact class-probability scores creates PPV/FOR gaps of
`0.20/0.08`. The official boundary algorithm then returns classifiers with both
gaps below `1e-12`; its maximum-accuracy and minimum-separation objectives agree
with an independent 16-start constrained optimizer within `1e-6`. Exhausting all
180 nonconstant deterministic rule pairs finds no exact sufficient solution,
confirming that randomization is essential here.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
python repro/src/run_fair_decisions.py --output outputs/summary.json
pytest -q
```

Official code pinned at `9d8b29f116f627868e9d5f2a31baa6d8567a0920`.
