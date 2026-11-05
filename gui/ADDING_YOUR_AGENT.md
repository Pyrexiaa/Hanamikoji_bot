# Adding Your Agent to the GUI

This guide shows you how to make your custom agent appear in the GUI's opponent list so you can
play against it interactively.

---

## Quick overview

Two steps:

1. **Import** your agent class at the top of `gui/server.py`.
2. **Add a registry entry** to `_AGENT_REGISTRY` in the same file.

The server reads `_AGENT_REGISTRY` to build the opponent list. Each entry tells the GUI which
class to instantiate for each game variant (tiny_hana / medium_hana / full_hana), or `None` if your
agent does not support that variant.

---

## Prerequisites

Your agent must:

- Be a subclass of `Agent` (from `engine.agents`)
- Implement `get_action_distribution(info_state: RoundInfoState)`

Your implementation lives in `student_attempts/` — for example, you might load a saved
policy-table agent or wrap your CFR solver output in an agent class there.

---

## Step 1 — Import your agent in `gui/server.py`

Open `gui/server.py` and find the import block at the top of the file. Add your import
**after** the existing imports:

```python
# --- add after the existing import block ---
from student_attempts.your_module import YourAgent
```

The server already adds the repo root to `sys.path`, so any module under `student_attempts/`
is importable using dot notation. The starter modules there (`solver.py`,
`treeplex_representation.py`, ...) import each other by bare name (`from env_config import ...`);
if your module imports them, put `student_attempts/` on the path first:

```python
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "student_attempts"))
from student_attempts.your_module import YourAgent
```

---

## Step 2 — Add a registry entry to `_AGENT_REGISTRY`

Find `_AGENT_REGISTRY` (after the imports) and add a new dict to the list:

```python
_AGENT_REGISTRY = [
    # ... existing entries ...

    # --- add your entry here ---
    {"id": "your_agent", "label": "Your Agent", "tiny": YourAgent, "medium": YourAgent, "full": YourAgent},
]
```

### Field reference

| Field | Type | Purpose |
|-------|------|---------|
| `"id"` | unique lowercase string | Used internally to look up the agent; never shown to users |
| `"label"` | string | Display name shown on the agent's tile in the GUI |
| `"tiny"` | class or `None` | Class to instantiate for tiny_hana; `None` hides it for that variant |
| `"medium"` | class or `None` | Class to instantiate for medium_hana; `None` hides it for that variant |
| `"full"` | class or `None` | Class to instantiate for full_hana; `None` hides it for that variant |

### Three common patterns

**Works with every variant (same class for all)**

```python
{"id": "your_agent", "label": "Your Agent", "tiny": YourAgent, "medium": YourAgent, "full": YourAgent},
```

**Works with tiny_hana only**

```python
{"id": "your_agent", "label": "Your Agent", "tiny": YourAgent, "medium": None, "full": None},
```

**Works with medium_hana only**

```python
{"id": "your_agent", "label": "Your Agent", "tiny": None, "medium": YourAgent, "full": None},
```

> **Note:** If you want your agent to work across variants, make sure its logic only
> relies on `RoundInfoState` helpers (like `get_available_action_types()`) rather than
> hardcoding variant-specific actions. The same class can often run on every variant unchanged.

---

### Agents from a policy table

A strategy saved as a policy table (see *Playing your agents* in `tasks/task4.ipynb`) needs no
class of its own: register a function that loads the file. The server calls it with no arguments
when a game starts, like a class. Give it a fallback for the states the table does not cover
(a single-round table only covers round 1 from neutral favors):

```python
from tiny_hana import load_policy_table_agent   # TinySuit and RandomAgent are already imported

def MyNashAgent():
    return load_policy_table_agent("tasks/my_nash_tiny.json", TinySuit,
                                   expected_variant="tiny_hana", fallback=RandomAgent())
```

```python
{"id": "my_nash", "label": "My Nash Agent", "tiny": MyNashAgent, "medium": None, "full": None},
```

The path is relative to the repo root, where you start the server.

### Agents that need a generated file

An agent that loads an artifact, such as a solver's policy table, can define a
static `unavailable_reason()` method. It returns `None` when the agent is ready,
or a message saying what is missing and how to create it. The GUI then shows the
agent greyed out with that message, and refuses to start a game with it. For the
`MyNashAgent` function above (a function can carry the method too):

```python
def _my_nash_unavailable():
    if os.path.exists("tasks/my_nash_tiny.json"):
        return None
    return "Policy table not generated: run the last code cell of tasks/task4.ipynb."

MyNashAgent.unavailable_reason = _my_nash_unavailable
```

## Step 3 — Verify

Run the server from the **repo root**:

```bash
python gui/server.py
```

Then open `http://localhost:8080` in your browser:

1. Select a game variant (tiny, medium or full).
2. Look under **Opponent**.
3. Confirm a tile with your agent's label appears.
4. Start a game and play a round.

---

## Troubleshooting

### Import error on startup

```
ModuleNotFoundError: No module named 'student_attempts.your_module'
```

- Make sure you are running from the **repo root**: `python gui/server.py`, not
  `python server.py` from inside `gui/`.
- Double-check that the filename and class name match exactly (Python is case-sensitive).

### Agent missing from the opponent list

- Check that the `"id"` value is **unique** — duplicate ids cause the second entry to be
  silently shadowed.
- Check that the correct variant's class is **not `None`**. If you selected
  *medium_hana* in the GUI but your entry has `"medium": None`, the agent will not appear.
