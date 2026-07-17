# Source audit

The official repository provides the exact boundary-tracing implementation and
two finite synthetic score-distribution pairs. These are the natural full-scale
instances for the three mathematical challenge claims; no learned model or
dataset subsampling is involved.

We execute the official algorithm, reconstruct randomized decision rules, and
evaluate PPV, FOR, accuracy, TPR, FPR, and total-variation separation directly
from the joint distribution. For independent validation, SciPy SLSQP optimizes
all 8 randomized bin decisions subject only to cross-multiplied PPV/FOR equality,
from 16 starts per objective. This path does not use the paper's boundary
parameterization or closed-form objective updates.
