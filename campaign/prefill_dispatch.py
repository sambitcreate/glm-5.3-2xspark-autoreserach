"""Conservative experimental dispatch frozen before the holdout sweep.

Fitted on the first synthetic sweep, not qualified for service deployment.
"""

def use_minblocks2(m: int, k: int, n: int) -> bool:
    return (k == 4096 and n == 2048 and m >= 1024) or (
        k == 1024 and n == 4096 and m >= 2048
    )
