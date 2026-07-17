# STATUS — Fair Decisions (`XiD5RhcEDK`)

**Session:** autoloop. **Last updated:** 2026-07-17. **State:** locally complete;
publication queued.

## Evidence

- Exact calibrated scores become predictively unfair after thresholding:
  PPV gap `0.20`, FOR gap `0.08`.
- Both paper synthetic cases attain PPV/FOR equality below `1e-12`.
- Accuracy optima: `0.723882` and `0.724724`; separation minima: `0.026824`
  and `0.092800`.
- Independent direct optimization over every randomized score-bin decision
  probability agrees with all four optima within `1e-6`.
- 180/180 nonconstant deterministic rule pairs exhausted; none is sufficient
  (minimum max parity gap `0.01818`).

## Next

Complete Trackio, publish public GitHub, and queue HF publication until the daily
Space-creation quota resets.
