"""Tests for hopcroft_minimize."""

import pytest
from main import hopcroft_minimize


# --- little helpers so I don't repeat myself ---

def check_partition(result, expected_states):
    """Make sure result is actually a valid partition of the expected states."""
    seen = set()
    for block in result:
        assert isinstance(block, frozenset)
        assert len(block) > 0
        assert not (block & seen), "overlapping blocks!"
        seen.update(block)
    assert seen == expected_states


def same_block(result, *states):
    """Check that all given states ended up in the same block."""
    blk = [b for b in result if states[0] in b]
    assert len(blk) == 1
    for s in states[1:]:
        assert s in blk[0], f"{states[0]} and {s} should be together"


def diff_blocks(result, a, b):
    """Check that a and b are NOT in the same block."""
    blk_a = next(bl for bl in result if a in bl)
    blk_b = next(bl for bl in result if b in bl)
    assert blk_a != blk_b, f"{a} and {b} shouldn't be together"


# === basics / edge cases ===

class TestBasics:

    def test_empty(self):
        assert hopcroft_minimize(set(), set(), {}) == set()

    def test_single_accept(self):
        result = hopcroft_minimize({0}, {"a"}, {("a", 0): {0}})
        assert result == {frozenset({0})}

    def test_single_reject(self):
        result = hopcroft_minimize(set(), {"a"}, {("a", 0): {0}})
        assert result == {frozenset({0})}

    def test_no_alphabet(self):
        result = hopcroft_minimize({0}, set(), {})
        assert result == {frozenset({0})}

    def test_missing_predecessors(self):
        """some symbols just don't have entries — shouldn't crash"""
        inv = {("a", 1): {0}, ("a", 2): {1}}
        result = hopcroft_minimize({2}, {"a", "b"}, inv)
        check_partition(result, {0, 1, 2})

    def test_empty_pred_sets(self):
        inv = {
            ("a", 0): set(),
            ("a", 1): {0},
            ("b", 0): {1},
            ("b", 1): set(),
        }
        result = hopcroft_minimize({1}, {"a", "b"}, inv)
        check_partition(result, {0, 1})


# === already minimal DFAs ===

class TestMinimal:

    def test_ends_with_b(self):
        inv = {("a", 0): {0, 1}, ("b", 1): {0, 1}}
        result = hopcroft_minimize({1}, {"a", "b"}, inv)
        assert result == {frozenset({0}), frozenset({1})}

    def test_mod3_binary(self):
        """divisible by 3 in binary — classic 3-state DFA"""
        inv = {
            ("0", 0): {0}, ("1", 1): {0},
            ("0", 2): {1}, ("1", 0): {1},
            ("0", 1): {2}, ("1", 2): {2},
        }
        result = hopcroft_minimize({0}, {"0", "1"}, inv)
        assert len(result) == 3

    def test_all_distinct(self):
        # 0(rej) -> 1(acc) -> 2(rej) -> 2
        inv = {("a", 1): {0}, ("a", 2): {1, 2}}
        result = hopcroft_minimize({1}, {"a"}, inv)
        assert len(result) == 3


# === states that should merge ===

class TestMerging:

    def test_two_equivalent_accept_states(self):
        """1 and 2 both accept and have the same transitions"""
        inv = {
            ("a", 1): {0, 1}, ("b", 2): {0, 2},
            ("a", 2): {2},    ("b", 1): {1},
        }
        result = hopcroft_minimize({1, 2}, {"a", "b"}, inv)
        same_block(result, 1, 2)
        diff_blocks(result, 0, 1)

    def test_all_accept_cycle(self):
        inv = {("a", 1): {0}, ("a", 2): {1}, ("a", 0): {2}}
        result = hopcroft_minimize({0, 1, 2}, {"a"}, inv)
        assert result == {frozenset({0, 1, 2})}

    def test_parallel_reject_states(self):
        # 1 and 2 are both reject, both go to 0(accept)
        inv = {("a", 0): {0, 1, 2}}
        result = hopcroft_minimize({0}, {"a"}, inv)
        same_block(result, 1, 2)
        diff_blocks(result, 0, 1)

    def test_duplicate_in_mod3(self):
        """state 3 is a clone of state 2 — should merge with it"""
        inv = {
            ("0", 0): {0},    ("1", 1): {0},
            ("0", 2): {1},    ("1", 0): {1},
            ("0", 1): {2, 3}, ("1", 2): {2, 3},
        }
        result = hopcroft_minimize({0}, {"0", "1"}, inv)
        same_block(result, 2, 3)
        assert len(result) == 3

    def test_self_loops_all_accept(self):
        inv = {("a", 0): {0}, ("a", 1): {1}, ("a", 2): {2}}
        result = hopcroft_minimize({0, 1, 2}, {"a"}, inv)
        assert result == {frozenset({0, 1, 2})}


# === states that must stay separate ===

class TestSplitting:

    def test_accept_vs_reject(self):
        inv = {("a", 0): {0, 1}}
        result = hopcroft_minimize({1}, {"a"}, inv)
        diff_blocks(result, 0, 1)

    def test_one_step_distinguishable(self):
        """1 -> 0(rej), 2 -> 2(acc) — they diverge after one step"""
        inv = {("a", 1): {0}, ("a", 0): {1}, ("a", 2): {2}}
        result = hopcroft_minimize({1, 2}, {"a"}, inv)
        diff_blocks(result, 1, 2)

    def test_two_step_distinguishable(self):
        # 0(r)->1(a)->2(a)->0  and  3(r)->4(a)->4
        # 1 and 4 look the same at first, but 1 eventually reaches reject
        inv = {
            ("a", 1): {0}, ("a", 2): {1}, ("a", 0): {2},
            ("a", 4): {3, 4},
        }
        result = hopcroft_minimize({1, 2, 4}, {"a"}, inv)
        diff_blocks(result, 1, 4)

    def test_split_by_second_symbol(self):
        """1 and 2 agree on 'a' but differ on 'b'"""
        inv = {
            ("a", 0): {1, 2},
            ("b", 3): {1},   # 1 -b-> 3(accept)
            ("b", 0): {2},   # 2 -b-> 0(reject)
        }
        result = hopcroft_minimize({3}, {"a", "b"}, inv)
        diff_blocks(result, 1, 2)

    def test_self_loop_accept_vs_reject(self):
        inv = {("a", 0): {0}, ("a", 1): {1}}
        result = hopcroft_minimize({0}, {"a"}, inv)
        assert result == {frozenset({0}), frozenset({1})}


# === non-integer state names ===

class TestWeirdStates:

    def test_string_states(self):
        inv = {("x", "accept"): {"start"}, ("x", "start"): {"start"}}
        result = hopcroft_minimize({"accept"}, {"x"}, inv)
        diff_blocks(result, "start", "accept")

    def test_tuple_states(self):
        inv = {("a", (1, 0)): {(0, 0)}, ("a", (0, 0)): {(1, 0)}}
        result = hopcroft_minimize({(1, 0)}, {"a"}, inv)
        diff_blocks(result, (0, 0), (1, 0))

    def test_state_only_in_predecessors(self):
        """state 5 only shows up as a predecessor, never as a target key"""
        inv = {("a", 0): {5}, ("a", 5): set()}
        result = hopcroft_minimize({0}, {"a"}, inv)
        check_partition(result, {0, 5})


# === partition sanity checks ===

class TestPartitionStructure:

    def test_covers_everything(self):
        inv = {
            ("a", 1): {0}, ("b", 2): {0},
            ("a", 3): {2}, ("b", 3): {1},
            ("a", 0): {3}, ("b", 1): {3},
        }
        result = hopcroft_minimize({1, 3}, {"a", "b"}, inv)
        check_partition(result, {0, 1, 2, 3})

    def test_blocks_are_disjoint(self):
        inv = {("a", 0): {0, 1}, ("a", 1): {2}}
        result = hopcroft_minimize({0}, {"a"}, inv)
        check_partition(result, {0, 1, 2})

    def test_no_empty_blocks(self):
        inv = {("a", 0): {0, 1}, ("a", 1): set()}
        result = hopcroft_minimize({0}, {"a"}, inv)
        for block in result:
            assert len(block) > 0


# === bigger DFAs ===

class TestScale:

    def test_ring_1000_one_accept(self):
        """all states distinguishable because each is a different distance from accept"""
        n = 1000
        inv = {("a", (i + 1) % n): {i} for i in range(n)}
        result = hopcroft_minimize({0}, {"a"}, inv)
        assert len(result) == n

    def test_ring_1000_all_accept(self):
        n = 1000
        inv = {("a", (i + 1) % n): {i} for i in range(n)}
        result = hopcroft_minimize(set(range(n)), {"a"}, inv)
        assert len(result) == 1

    def test_star_500_leaves(self):
        """500 leaves all point to a hub — leaves should all merge"""
        leaves = set(range(500))
        inv = {("a", "hub"): leaves | {"hub"}}
        result = hopcroft_minimize({"hub"}, {"a"}, inv)
        same_block(result, 0, 1, 42, 499)  # spot check
        diff_blocks(result, "hub", 0)

    def test_binary_counter_10bit(self):
        """mod 1024 DFA — many states collapse due to shared bit structure"""
        n = 1024
        inv = {}
        for s in range(n):
            inv.setdefault(("0", (2 * s) % n), set()).add(s)
            inv.setdefault(("1", (2 * s + 1) % n), set()).add(s)
        result = hopcroft_minimize({0}, {"0", "1"}, inv)
        # not all 1024 are distinguishable — states with same
        # power-of-2 structure collapse. just check it's a valid
        # partition and reasonably reduced
        check_partition(result, set(range(n)))
        assert len(result) < n


# === tricky regression-ish cases ===

class TestRegression:

    def test_worklist_heuristic(self):
        """
        two separate acc/rej cycles — structurally identical,
        so 1&2 should merge and 0&3 should merge.
        """
        inv = {
            ("a", 1): {0}, ("a", 0): {1},
            ("a", 3): {2}, ("a", 2): {3},
        }
        result = hopcroft_minimize({1, 2}, {"a"}, inv)
        same_block(result, 1, 2)
        same_block(result, 0, 3)
        assert len(result) == 2

    def test_accepting_state_not_in_transitions(self):
        """accept state only known from accepting_states set"""
        inv = {("a", 1): {1}}
        result = hopcroft_minimize({0, 1}, {"a"}, inv)
        check_partition(result, {0, 1})
