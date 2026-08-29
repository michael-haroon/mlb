# Routing algorithm
"""
Generate importance data

For each test:
    if CI upper bound < 0: reject (since we have 2026 fold, it includes data from this season already, so if not a single fold is positive, get rid of it)
    if strongly monotonically increasing:
        if latest fold positive: accept
        if latest fold negative AND theil sen slope > 0 such that next projected fold is likely>0: reject, but flag it for next season as a potentional signal
    if strongly monotonically decreasing:
        if recent fold is positive AND theil sen slope indicates that next projected fold is likely>0: keep
        if recent years are negative: reject
     if not monotonic:
        if oscilating between >null and <null 
            if median > null and mean > null:
                flag test as unstable and only feed it to trees. (should any other model get it? or does that need to be tested)
            if median > null and mean < null:
                if negative outlier exists and is in the first 4 folds: accept
                else: left tailed due to some large number or outlier. do we reject or give it to trees? Likely reject
            if median < null and mean > null:
                right tailed. Just luck. It's noise. Reject
            if median < null and mean < null:
                reject
        if oscilating but entire CI is > null:
            keep, but flag it as unstable
        if U shaped:
            we need a way to see if we can measure U shapes AND see if recent folds are > null (since U shape implies it is growing in importance)
        if upside down U shaped:
            reject
        if uniform > null:
            if it contains an outlier in the last fold that kills the mean:
                reject. can be noise, but assume it is rule change
            if it contains an outlier early on but is overall a positive importance: keep since it is just left skewed
        if uniform < null:
            reject
"""

"""Note that these tests emphasise the analysis of time series of size n=8, a very small dataset that makes any test quite weak."""