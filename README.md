# Hopcroft DFA Minimization

Implementation of Hopcroft's algorithm to minimize a deterministic finite automaton (DFA).

## Structure

```
solution/
├── main.py    # Implementation of hopcroft_minimize()
└── test.py    # Test suite (pytest)
```

## Setup

Requires Python 3.11+ and pytest.

```bash
pip install pytest
```

## Run

Run the solution:

```bash
python solution/main.py
```

Run tests:

```bash
python -m pytest -q solution/test.py -v
```

## API

```python
hopcroft_minimize(accepting_states, input_alphabet, inverse_transition_func)
```

- `accepting_states`: set of accepting states
- `input_alphabet`: set of input symbols
- `inverse_transition_func`: dict mapping `(symbol, state)` → `set` of predecessor states

Returns a `set` of `frozenset`s representing the minimal DFA partition.
