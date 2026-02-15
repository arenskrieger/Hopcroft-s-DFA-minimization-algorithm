# Hopcroft's algorithm for DFA minimization.
# DFA is given in inverse form: inverse_transition_func[(symbol, state)] -> predecessor states


def _get_all_states(accepting_states, inverse_transition_func):
    """Pull together all states we can find from the inputs."""
    states = set(accepting_states)
    for (sym, tgt), preds in inverse_transition_func.items():
        states.add(tgt)
        if preds:
            states.update(preds)
    return states


def hopcroft_minimize(accepting_states, input_alphabet, inverse_transition_func):
    """
    Partition DFA states into equivalence classes using Hopcroft's algorithm.
    Returns a set of frozensets.
    """
    Q = _get_all_states(accepting_states, inverse_transition_func)
    if not Q:
        return set()

    F = frozenset(s for s in accepting_states if s in Q)
    non_F = frozenset(Q - F)

    P = set()
    if F:
        P.add(F)
    if non_F:
        P.add(non_F)

    if len(P) <= 1:
        return P

    # keep track of which block each state is in right now
    block_of = {}
    for block in P:
        for s in block:
            block_of[s] = block

    # seed worklist with the smaller group (that's the Hopcroft trick)
    W = set()
    W.add(F if len(F) <= len(non_F) else non_F)

    while W:
        splitter = W.pop()

        for sym in input_alphabet:
            # group predecessors of splitter directly by their block
            affected = {}
            for tgt in splitter:
                preds = inverse_transition_func.get((sym, tgt))
                if not preds:
                    continue
                for q in preds:
                    blk = block_of.get(q)
                    if blk is not None:
                        affected.setdefault(blk, set()).add(q)

            if not affected:
                continue

            # try to split each affected block
            for Y, overlap in affected.items():
                if len(overlap) == len(Y):
                    continue  # whole block is in pre, nothing to split

                part1 = frozenset(overlap)
                part2 = Y - part1

                P.remove(Y)
                P.add(part1)
                P.add(part2)

                for s in part1:
                    block_of[s] = part1
                for s in part2:
                    block_of[s] = part2

                if Y in W:
                    W.remove(Y)
                    W.add(part1)
                    W.add(part2)
                else:
                    # only add the smaller half — keeps it O(n log n)
                    W.add(part1 if len(part1) <= len(part2) else part2)

    return P


if __name__ == "__main__":
    # quick test: DFA that accepts strings ending in 'b'
    acc = {1}
    alph = {"a", "b"}
    inv = {
        ("a", 0): {0, 1},
        ("b", 1): {0, 1},
    }
    print(hopcroft_minimize(acc, alph, inv))
