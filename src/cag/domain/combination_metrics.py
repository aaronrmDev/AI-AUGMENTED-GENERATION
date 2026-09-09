def compounded_reduction(stage_reductions: list[float]) -> float:
    # What a genuine pipeline would achieve: each stage acting on what
    # the previous stage left, so the reductions multiply. This is the
    # prediction CAG.md's "each step reduces what the next step has to
    # handle" makes, stated as a number so it can be checked.
    if not stage_reductions:
        raise ValueError("stage_reductions must be non-empty")
    if any(reduction <= 0 for reduction in stage_reductions):
        raise ValueError("every stage reduction must be positive")
    product = 1.0
    for reduction in stage_reductions:
        product *= reduction
    return product


def best_single_reduction(stage_reductions: list[float]) -> float:
    # What "merely compatible" would look like: the techniques coexist
    # without reinforcing, so the combination is no better than whichever
    # one was strongest on its own. This is the null hypothesis the
    # measured result has to beat before a combination has earned the
    # word "synergy".
    if not stage_reductions:
        raise ValueError("stage_reductions must be non-empty")
    return max(stage_reductions)


def synergy_score(measured_reduction: float, stage_reductions: list[float]) -> float:
    # Where the measured combination actually falls between the two
    # hypotheses: 0.0 means it did no better than its strongest single
    # technique (merely compatible), 1.0 means it achieved the full
    # product (genuinely compounding). Values above 1.0 would mean the
    # stages helped each other beyond simple multiplication; below 0.0,
    # that combining actively hurt.
    #
    # Reported rather than a bare pass/fail because CAG.md itself grades
    # these combinations A+ and A rather than yes/no, and a score is the
    # only shape that can distinguish those.
    if measured_reduction <= 0:
        raise ValueError("measured_reduction must be positive")
    floor = best_single_reduction(stage_reductions)
    ceiling = compounded_reduction(stage_reductions)
    if ceiling == floor:
        return 1.0 if measured_reduction >= ceiling else 0.0
    return (measured_reduction - floor) / (ceiling - floor)


def interference_drift(standalone_reduction: float, in_pipeline_reduction: float) -> float:
    # The question a synergy score alone cannot answer, and the one that
    # actually tests CAG.md's "each step reduces what the next step has
    # to handle": does a technique perform as well on what the previous
    # stage left as it does on the raw cache?
    #
    # Necessary because a pipeline reducing toward a FIXED target makes
    # the naive product telescope -- measure eviction, compression and
    # offloading as ratios into a fixed GPU capacity and their product is
    # original/capacity no matter how the stages divide the work, so a
    # synergy score of 1.0 there is close to tautological. Drift is
    # measured per stage against that stage run alone, so it cannot
    # telescope: 0.0 means the stages compose cleanly, and a negative
    # value means the earlier stage left the later one something harder
    # to work with than it started with, which the product would
    # overstate.
    if standalone_reduction <= 0:
        raise ValueError("standalone_reduction must be positive")
    if in_pipeline_reduction <= 0:
        raise ValueError("in_pipeline_reduction must be positive")
    return (in_pipeline_reduction - standalone_reduction) / standalone_reduction
