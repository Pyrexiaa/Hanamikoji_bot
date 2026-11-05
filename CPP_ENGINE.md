# Using the C++ engine

The game engine exists twice: in Python (`engine/`, the default) and in C++ (`engine_cpp/`, used
from Python through the module `hana_engine_cpp`). Both play exactly the same game and offer the
same Python API, so the code you write for the tasks runs on either. This page explains when the
C++ engine is worth using, how to build and switch to it, and what to check when you move code
written on the Python engine.

## When it helps

The C++ engine makes **walking the game tree** faster: `execute_action`, `undo_move`,
`get_legal_actions`, `get_info_state` and the chance probabilities. It does not speed up NumPy
arithmetic. Timings for one round of tiny_hana, from the task notebooks:

| | Python engine | C++ engine |
|---|---|---|
| Task 1: posteriors for 40 information sets | 12.6 s | 0.9 s |
| Task 2: building the treeplex (`TreeplexGame()`) | 0.7 s | 0.1 s |
| Task 3: the CFR run in the notebook | 67 s | 66 s |

CFR itself runs on the treeplex arrays, so it takes the same time on both engines. The C++ engine
pays off when your code spends its time in the game tree: posteriors, building treeplexes for
larger games (medium_hana, or one treeplex per Markov state in a multi-round solver), sampling
or simulating many games, and Monte Carlo methods. That makes it most useful for Task 4.

## 1. Build it (once)

The C++ engine is a compiled Python extension. You need a C++17 compiler:

- macOS: the Xcode command-line tools (`xcode-select --install`)
- Linux: `build-essential` (or your distribution's `g++`)

The rest of the build tools (CMake, Ninja, pybind11, scikit-build-core) are in `requirements.txt`.
From the repo root:

```bash
pip install -r requirements.txt
pip install -e . --no-build-isolation      # compiles engine_cpp/; this can take a few minutes
python -c "import hana_engine_cpp"          # no output means it worked
```

After changing anything in `engine_cpp/`, run the `pip install` line again to rebuild.

## 2. Switch to it

The engine is chosen by the environment variable `HANA_ENGINE` **when `engine` is first
imported**. `HANA_ENGINE=cpp` selects the C++ engine; anything else, or leaving it unset, selects
the Python engine.

- **Scripts:** `HANA_ENGINE=cpp python my_script.py`
- **Notebooks:** start Jupyter with `HANA_ENGINE=cpp jupyter notebook`. Or make this the first line
  of the notebook's setup cell, then **restart the kernel** (a kernel that has already imported
  `engine` keeps the engine it started with):

  ```python
  import os; os.environ["HANA_ENGINE"] = "cpp"
  ```

To check which engine is running:

```python
import tiny_hana as env
print(env.RoundInfoState.__module__)   # 'hana_engine_cpp' for C++, 'engine.info_state' for Python
```

Import the game as usual in both cases (`import tiny_hana as env`, or `env_config` in
`student_attempts/`); the same names refer to the C++ classes when the C++ engine is active.

## 3. Porting code from the Python engine

Code that uses the documented API (`GameState`, `RoundInfoState`, `Action`, agents and the helpers
in `student_attempts/`) needs no changes. These work the same on both engines:

- `execute_action` / `undo_move`, `get_legal_actions`, `get_chance_actions_with_probs`,
  `current_player`, `is_round_complete`, `terminal_payoff`, `to_key`, `pretty_print`
- information states: `==`, use as dictionary keys and in sets, `str(info_state)`, and all the
  `get_...` query methods
- `copy.copy(state)` and `copy.deepcopy(state)`
- setting starting favors by assignment, and `make_round_start_state`
- actions and their fields (`action_type`, `cards`, `card.suit`, `choice`), `Game`, agents,
  policy tables, `TreeplexGame`

The engines differ in a few places. Check your own code for these:

| If your code… | On the C++ engine | Write this instead (works on both) |
|---|---|---|
| samples chance with `state.sample_chance_action(random.Random(...))` | `TypeError`: it needs the engine's own generator | `actions, probs = zip(*state.get_chance_actions_with_probs())` then `rng.choices(actions, weights=probs)[0]`; or pass `engine.Rng(seed)` when the C++ engine is active |
| sets favors with `state.favors[suit] = v` or `state.favors.update(...)` | **silently does nothing**: reading `state.favors` gives a copy | assign the whole dictionary: `state.favors = {**state.favors, suit: v}` |
| compares `action.action_type.value` with a string such as `"reserve"` | `.value` is an integer | compare with the enum, `action.action_type == env.ActionType.RESERVE`, or use `.name` |
| reads `info_state.events` | not available | use the `get_...` query methods |
| relies on the used actions from `get_current_hand_and_actions()` being a `set` | it is a `list` | only test membership (`in`) or iterate |
| calls `next(...)` on `get_legal_actions()` | it returns a `list` | iterate over it, or use `list(...)` |

The favors case is the one to watch: it raises no error. If you set up rounds with non-neutral
favors, print `state.favors` once after setting them.

For full_hana, `get_chance_actions_with_probs()` at the initial deal lists 125,953 deals; sample
it with `sample_chance_action` instead (and `engine.Rng` on the C++ engine).

A good way to port: run your code once on each engine and compare the numbers. Both engines play
the same game, so values such as expected payoffs, best-response values and saddle-point
residuals agree exactly.

## Using the engine from C++

The C++ engine is header-only, so a C++ program can use it directly, without Python.
`cpp_samples.cpp` shows how: the same examples as `engine_samples.py`, written in C++. From the repo root:

```bash
g++ -std=c++17 -I engine_cpp/include -o cpp_samples cpp_samples.cpp
./cpp_samples
```

## Troubleshooting

- **`ModuleNotFoundError: No module named 'hana_engine_cpp'`**: the C++ engine is not built, or was
  built for another Python. Run the build in step 1 with the same `python`/`pip` you run your code with.
- **Still running on the Python engine in a notebook**: the kernel imported `engine` before
  `HANA_ENGINE` was set. Restart the kernel and run the cells from the top.
- **The build fails with a compiler error**: check that a C++17 compiler is installed (step 1).
