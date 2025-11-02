import argparse
import math
import random
from collections import Counter

# Local alignment with composition-adjusted significance for arbitrary alphabets.
# Two positional arguments are read: query and subject. They are treated as raw strings.

MATCH_SCORE = 2
MISMATCH_PENALTY = 1
GAP_PENALTY = 2
PSEUDOCOUNT = 0.5  # symmetric Dirichlet smoothing


def run_mask(seq):
    # Masks homopolymer runs so that long single-symbol stretches don't dominate
    n = len(seq)
    if n == 0:
        return [False] * 0
    # threshold scales gently with length; minimum of eight
    thr = max(8, int(0.04 * n))
    mask = [False] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and seq[j] == seq[i]:
            j += 1
        run_len = j - i
        if run_len >= thr:
            for k in range(i, j):
                mask[k] = True
        i = j
    return mask


def symbol_probs(seq, other_seq, mask):
    # Probabilities are computed on the union alphabet with symmetric smoothing
    unmasked = [c for i, c in enumerate(seq) if not mask[i]]
    counts = Counter(unmasked)
    alpha = set(seq) | set(other_seq)
    k = len(alpha) if len(alpha) > 0 else 1
    total = sum(counts[c] for c in alpha) + PSEUDOCOUNT * k
    return {c: (counts.get(c, 0) + PSEUDOCOUNT) / total for c in alpha}


def smith_waterman_linear(q, s, qmask, smask):
    # Returns best score, coordinates, and traceback-derived stats
    n, m = len(q), len(s)
    if n == 0 or m == 0:
        return 0, (0, 0, 0, 0), {"matches": 0, "mismatches": 0, "gaps": 0, "L": 0}

    # DP matrix H with zeros; store as list of lists of ints
    H = [[0] * (m + 1) for _ in range(n + 1)]
    best = 0
    best_i, best_j = 0, 0

    for i in range(1, n + 1):
        qi = q[i - 1]
        qi_masked = qmask[i - 1]
        row = H[i]
        prev = H[i - 1]
        for j in range(1, m + 1):
            sj = s[j - 1]
            sj_masked = smask[j - 1]
            if not qi_masked and not sj_masked and qi == sj:
                sub = MATCH_SCORE
            else:
                sub = -MISMATCH_PENALTY
            diag = prev[j - 1] + sub
            up = prev[j] - GAP_PENALTY
            left = row[j - 1] - GAP_PENALTY
            val = diag
            if up > val:
                val = up
            if left > val:
                val = left
            if val < 0:
                val = 0
            row[j] = val
            if val > best:
                best = val
                best_i, best_j = i, j

    # Traceback to get alignment region and counts
    i, j = best_i, best_j
    matches = 0
    mismatches = 0
    gaps = 0
    L = 0
    while i > 0 and j > 0 and H[i][j] > 0:
        qi = q[i - 1]
        sj = s[j - 1]
        qi_masked = qmask[i - 1]
        sj_masked = smask[j - 1]
        if not qi_masked and not sj_masked and qi == sj:
            sub = MATCH_SCORE
        else:
            sub = -MISMATCH_PENALTY
        if H[i][j] == H[i - 1][j - 1] + sub:
            L += 1
            if sub > 0:
                matches += 1
            else:
                mismatches += 1
            i -= 1
            j -= 1
        elif H[i][j] == H[i - 1][j] - GAP_PENALTY:
            gaps += 1
            i -= 1
        elif H[i][j] == H[i][j - 1] - GAP_PENALTY:
            gaps += 1
            j -= 1
        else:
            break

    q_start = i
    q_end = best_i
    s_start = j
    s_end = best_j
    stats = {"matches": matches, "mismatches": mismatches, "gaps": gaps, "L": L}
    return best, (q_start, q_end, s_start, s_end), stats


def expected_match_share(pq, ps):
    # Sum over union alphabet of pq[a]*ps[a]
    # If alphabets differ, missing symbols have zero mass because of smoothing union
    alpha = set(pq.keys()) | set(ps.keys())
    return sum(pq.get(a, 0.0) * ps.get(a, 0.0) for a in alpha)


def adjusted_score(raw_score, L, pq, ps):
    if L <= 0:
        return 0.0
    em = L * expected_match_share(pq, ps)
    # Only centers the substitution portion; gaps remain as-is
    expected_sub_score = MATCH_SCORE * em - MISMATCH_PENALTY * (L - em)
    return raw_score - expected_sub_score


def shuffle_preserving_counts(seq):
    arr = list(seq)
    random.shuffle(arr)
    return "".join(arr)


def auto_null_samples(n, m):
    # Adaptive budget to keep runtime sane
    area = n * m
    # target work proportional to ~5e7 DP cell updates
    b = int(max(50, min(2000, (5e7 // max(1, area)))))
    return b


def main():
    parser = argparse.ArgumentParser(description="Composition-aware local alignment significance for arbitrary strings.")
    parser.add_argument("query", type=str, help="Query sequence")
    parser.add_argument("subject", type=str, help="Subject sequence")
    args = parser.parse_args()

    q = args.query
    s = args.subject

    # Preprocess
    qmask = run_mask(q)
    smask = run_mask(s)

    # Frequencies on union alphabet with smoothing
    pq = symbol_probs(q, s, qmask)
    ps = symbol_probs(s, q, smask)

    # Real alignment
    best, coords, stats = smith_waterman_linear(q, s, qmask, smask)
    L = stats["L"]
    raw_score = stats["matches"] * MATCH_SCORE - stats["mismatches"] * MISMATCH_PENALTY - stats["gaps"] * GAP_PENALTY
    adj = adjusted_score(raw_score, L, pq, ps)

    # Null calibration with permutation of each sequence
    n, m = len(q), len(s)
    B = auto_null_samples(n, m)
    null_ge = 0
    for _ in range(B):
        q_shuf = shuffle_preserving_counts(q)
        s_shuf = shuffle_preserving_counts(s)
        qmask_b = run_mask(q_shuf)
        smask_b = run_mask(s_shuf)
        pq_b = symbol_probs(q_shuf, s_shuf, qmask_b)
        ps_b = symbol_probs(s_shuf, q_shuf, smask_b)
        best_b, _, stats_b = smith_waterman_linear(q_shuf, s_shuf, qmask_b, smask_b)
        L_b = stats_b["L"]
        raw_b = stats_b["matches"] * MATCH_SCORE - stats_b["mismatches"] * MISMATCH_PENALTY - stats_b["gaps"] * GAP_PENALTY
        adj_b = adjusted_score(raw_b, L_b, pq_b, ps_b)
        if adj_b >= adj:
            null_ge += 1

    p_value = (null_ge + 1) / (B + 1)

    # Identity computed on aligned pairs only
    identity = (stats["matches"] / L) if L > 0 else 0.0

    # Output as a compact, readable report
    print("{")
    print(f'  "p_value": {p_value:.6g},')
    print(f'  "adjusted_score": {adj:.6g},')
    print(f'  "raw_score": {raw_score},')
    print('  "alignment": {')
    print(f'    "query_start": {coords[0]}, "query_end": {coords[1]}, "subject_start": {coords[2]}, "subject_end": {coords[3]},')
    print(f'    "aligned_pairs": {L}, "matches": {stats["matches"]}, "mismatches": {stats["mismatches"]}, "gaps": {stats["gaps"]}, "identity": {identity:.6g}')
    print("  },")
    print(f'  "alphabet_size": {len(set(q) | set(s))},')
    print(f'  "null_samples": {B}')
    print("}")


if __name__ == "__main__":
    main()
