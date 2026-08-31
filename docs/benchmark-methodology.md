# Benchmark methodology

The private campaign used six scout tasks and three hard tasks on fixed source
repositories. Both the candidate and control ran on the same host with isolated
CPU and GPU allocation. Temperature, output cap, reasoning policy, task set,
verification commands, and hidden grading thresholds were held fixed.

The primary metric was:

```text
seconds_per_clean = mean_worker_wall_seconds / clean_rate
```

This penalizes fast failures instead of rewarding them. “Clean” required all
of the following: successful worker completion, public verification, maximum
hidden score, a real scoped commit, source isolation, a valid finish reason,
reasoning preservation, and no output-cap or token-loop failure.

The strongest timing comparison uses active worker timestamps for warm tasks
2–9: 22.34 seconds for Flash-Next versus 28.70 for the control. The aggregate
32.111/40.784 seconds-per-clean figures came from controllers with different
polling intervals (15 seconds versus 5 seconds), so they are supporting trend
evidence and not treated as perfectly matched stopwatch measurements.

The public `scripts/accept.py` is deliberately different. It contains no
private tasks or graders. It verifies the properties another operator can
reproduce: API health, model identity, reasoning visibility, a tool-call
round-trip, exact 65K/120K prefills, and absence of fatal container-log markers.
