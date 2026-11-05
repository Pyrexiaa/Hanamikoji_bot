"""
HTTP server for Hanamikoji single-player web GUI (tiny_hana, medium_hana and
full_hana variants).

Run from repo root:  python gui/server.py
Then open:          http://localhost:8080
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import random

from tiny_hana.tiny_hana import (GameState as TinyGameState, Suit as TinySuit, GAME_CONFIG as TINY_CONFIG)
from tiny_hana.example_agents import (
    CardCountingAgent as TinyCardCounting,
    GreedyAgent as TinyGreedy,
    BalancedAgent as TinyBalanced,
    AdaptiveAgent as TinyAdaptive,
)
from medium_hana.medium_hana import (GameState as MediumGameState, Suit as MediumSuit, GAME_CONFIG as MEDIUM_CONFIG)
from medium_hana.example_agents import (
    CardCountingAgent as MediumCardCounting,
    GreedyAgent as MediumGreedy,
    BalancedAgent as MediumBalanced,
)
from full_hana.full_hana import (GameState as FullGameState, Suit as FullSuit, GAME_CONFIG as FULL_CONFIG)
from full_hana.example_agents import (
    CardCountingAgent as FullCardCounting,
    GreedyAgent as FullGreedy,
)
from shared_example_agents import (
    LeadProtectorAgent,
    ComebackContesterAgent,
    RankWeightedAgent,
    MajorityThresholdAgent,
    DenialFirstAgent,
    ActionOrderTemplateAgent,
    PairSynergyAgent,
    RiskAdaptiveAgent,
    BeliefBucketAgent,
    CoverageFirstAgent,
)
from engine.constants import CHANCE_PLAYER
from engine.models import Action, Card
from engine.enums import ActionType
from engine.agents import RandomAgent, Agent
from engine.game import Game
from engine.events import (
    RoundStartEvent, DrawCardEvent, OwnReserveEvent, OpponentReserveEvent,
    OwnSplitEvent, OpponentSplitEvent, SplitChoiceEvent, RoundEndEvent,
    OwnDiscardEvent, OpponentDiscardEvent,
    OwnTradeOffEvent, OpponentTradeOffEvent, OwnGiftEvent, OpponentGiftEvent,
    GiftChoiceEvent, OwnCompetitionEvent, OpponentCompetitionEvent, CompetitionChoiceEvent,
)

# ---------------------------------------------------------------------------
# Global state (single local user, no sessions needed)
# ---------------------------------------------------------------------------

_game = None
_human_player: int = 0
_ai_agent: Agent = None
_rng: random.Random = random.Random()
_awaiting_next_round: bool = False
_single_round_mode: bool = False
_prev_favors: dict = {}          # favor snapshot captured before update_favors()

_variant = {
    "game_state_cls": TinyGameState,
    "suit":           TinySuit,
    "config":         TINY_CONFIG,
}

_VARIANTS = {
    "tiny":   {"game_state_cls": TinyGameState,   "suit": TinySuit,   "config": TINY_CONFIG},
    "medium": {"game_state_cls": MediumGameState, "suit": MediumSuit, "config": MEDIUM_CONFIG},
    "full":   {"game_state_cls": FullGameState,   "suit": FullSuit,   "config": FULL_CONFIG},
}

# GUI variant id (as sent by the page) -> _VARIANTS / _AGENT_REGISTRY key
_VARIANT_IDS = {"tiny_hana": "tiny", "medium_hana": "medium", "full_hana": "full"}


def _variant_key(variant: str) -> str:
    return _VARIANT_IDS.get(variant, "tiny")


_AGENT_REGISTRY = [
    # ── Universal ──────────────────────────────────────────────────────────
    {"id": "random",           "label": "Random Agent",            "tiny": RandomAgent,             "medium": RandomAgent, "full": RandomAgent},
    {"id": "card_counting",    "label": "Card Counting Agent",     "tiny": TinyCardCounting,        "medium": MediumCardCounting, "full": FullCardCounting},

    # ── Variant-specific ───────────────────────────────────────────────────
    {"id": "greedy",           "label": "Greedy Agent",            "tiny": TinyGreedy,              "medium": MediumGreedy, "full": FullGreedy},
    {"id": "balanced",         "label": "Balanced Agent",          "tiny": TinyBalanced,            "medium": MediumBalanced, "full": None},
    {"id": "adaptive",         "label": "Adaptive Agent",          "tiny": TinyAdaptive,            "medium": None, "full": None},

    # ── Shared heuristic agents (all variants) ─────────────────────────────
    {"id": "rank_weighted",       "label": "Rank Weighted Agent",       "tiny": RankWeightedAgent,       "medium": RankWeightedAgent, "full": RankWeightedAgent},
    {"id": "lead_protector",      "label": "Lead Protector Agent",      "tiny": LeadProtectorAgent,      "medium": LeadProtectorAgent, "full": LeadProtectorAgent},
    {"id": "comeback_contester",  "label": "Comeback Contester Agent",  "tiny": ComebackContesterAgent,  "medium": ComebackContesterAgent, "full": ComebackContesterAgent},
    {"id": "majority_threshold",  "label": "Majority Threshold Agent",  "tiny": MajorityThresholdAgent,  "medium": MajorityThresholdAgent, "full": MajorityThresholdAgent},
    {"id": "denial_first",        "label": "Denial First Agent",        "tiny": DenialFirstAgent,        "medium": DenialFirstAgent, "full": DenialFirstAgent},
    {"id": "action_order",        "label": "Action Order Agent",        "tiny": ActionOrderTemplateAgent, "medium": ActionOrderTemplateAgent, "full": ActionOrderTemplateAgent},
    {"id": "pair_synergy",        "label": "Pair Synergy Agent",        "tiny": PairSynergyAgent,        "medium": PairSynergyAgent, "full": PairSynergyAgent},
    {"id": "risk_adaptive",       "label": "Risk Adaptive Agent",       "tiny": RiskAdaptiveAgent,       "medium": RiskAdaptiveAgent, "full": RiskAdaptiveAgent},
    {"id": "belief_bucket",       "label": "Belief Bucket Agent",       "tiny": BeliefBucketAgent,       "medium": BeliefBucketAgent, "full": BeliefBucketAgent},
    {"id": "coverage_first",      "label": "Coverage First Agent",      "tiny": CoverageFirstAgent,      "medium": CoverageFirstAgent, "full": CoverageFirstAgent},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _suit_display(name: str) -> str:
    """Convert suit enum name to display string: GEISHA_1 → Geisha 1."""
    return name.replace("_", " ").title()


def _build_win_note(config) -> str:
    parts = [f"{config.win_geisha_count}+ geishas"]
    if config.win_favor_points is not None:
        parts.append(f"{config.win_favor_points}+ favor pts")
    return "Win by controlling " + " or ".join(parts)



def _favor_label(fav: int, hp: int) -> str:
    """Convert a raw favor value (-1/0/1) to 'You'/'AI'/'—' from the human's POV."""
    if fav == 0:
        return "—"
    human_has = (fav == -1 and hp == 0) or (fav == 1 and hp == 1)
    return "You" if human_has else "AI"


def _build_round_end_summary(g, hp: int, config) -> dict:
    """Compute round-end summary after update_favors() but before _begin_round()."""
    def suit_totals(ps):
        counts = {}
        for c in ps.collected_cards:
            counts[c.suit.name] = counts.get(c.suit.name, 0) + 1
        return counts

    h_totals = suit_totals(g.players[hp])
    a_totals = suit_totals(g.players[1 - hp])

    # The AI's reserved cards, revealed at the end of the round.
    ai_reserved = []
    for event in (g.info_state_history[hp][-1] if g.info_state_history[hp] else []):
        if isinstance(event, RoundEndEvent):
            ai_reserved = [suit.name for suit in getattr(event, f"p{1 - hp}_reserved_revealed")]

    human_geishas = 0; ai_geishas = 0
    human_pts = 0;     ai_pts = 0
    per_geisha = []
    for suit in config.suits:
        fav = g.favors[suit]
        human_has = (fav == -1 and hp == 0) or (fav == 1 and hp == 1)
        ai_has    = fav != 0 and not human_has
        if human_has: human_geishas += 1; human_pts += suit.rank
        if ai_has:    ai_geishas    += 1; ai_pts    += suit.rank
        prev_fav = _prev_favors.get(suit, 0)
        per_geisha.append({
            "name":          suit.name,
            "rank":          suit.rank,
            "human_total":   h_totals.get(suit.name, 0),
            "ai_total":      a_totals.get(suit.name, 0),
            "prev_favor":    _favor_label(prev_fav, hp),
            "updated_favor": _favor_label(fav, hp),
        })

    return {
        "per_geisha":         per_geisha,
        "human_geisha_count": human_geishas,
        "ai_geisha_count":    ai_geishas,
        "human_favor_pts":    human_pts,
        "ai_favor_pts":       ai_pts,
        "win_geisha_count":   config.win_geisha_count,
        "win_favor_pts":      config.win_favor_points,
        "ai_reserved":        ai_reserved,
    }


def _event_to_log(event, player_id: int) -> str:
    """Convert an ObservationEvent to a human-readable log string."""
    if isinstance(event, DrawCardEvent):
        return f"You drew {_suit_display(event.card.name)}."
    if isinstance(event, OwnReserveEvent):
        return f"You reserved {_suit_display(event.card.name)} (hidden)."
    if isinstance(event, OpponentReserveEvent):
        return "Opponent reserved a card (hidden)."
    if isinstance(event, OwnDiscardEvent):
        reveal = _suit_display(event.revealed_card.name)
        hide   = _suit_display(event.hidden_card.name)
        return f"You discarded: revealed {reveal}, hid {hide}."
    if isinstance(event, OpponentDiscardEvent):
        reveal = _suit_display(event.revealed_card.name)
        return f"Opponent discarded: revealed {reveal} (other hidden)."
    if isinstance(event, OwnSplitEvent):
        cards = " + ".join(_suit_display(c.name) for c in event.cards)
        return f"You offered split: {cards}."
    if isinstance(event, OpponentSplitEvent):
        cards = " + ".join(_suit_display(c.name) for c in event.cards)
        return f"Opponent offered split: {cards}."
    if isinstance(event, SplitChoiceEvent):
        gets = _suit_display(event.chooser_gets.name)
        gives = _suit_display(event.offerer_gets.name)
        if event.chooser == player_id:
            return f"You took {gets} from split; opponent gets {gives}."
        else:
            return f"Opponent took {gets}; you get {gives}."
    if isinstance(event, OwnTradeOffEvent):
        cards = " + ".join(_suit_display(c.name) for c in event.cards)
        return f"You traded off {cards} (face down)."
    if isinstance(event, OpponentTradeOffEvent):
        return "Opponent traded off 2 cards (face down)."
    if isinstance(event, OwnGiftEvent):
        cards = ", ".join(_suit_display(c.name) for c in event.cards)
        return f"You offered a gift: {cards}."
    if isinstance(event, OpponentGiftEvent):
        cards = ", ".join(_suit_display(c.name) for c in event.cards)
        return f"Opponent offered a gift: {cards}."
    if isinstance(event, GiftChoiceEvent):
        gets = _suit_display(event.chooser_gets.name)
        rest = " + ".join(_suit_display(c.name) for c in event.offerer_gets)
        if event.chooser == player_id:
            return f"You took {gets} from the gift; opponent keeps {rest}."
        return f"Opponent took {gets}; you keep {rest}."
    if isinstance(event, (OwnCompetitionEvent, OpponentCompetitionEvent)):
        pairs = " vs ".join(" + ".join(_suit_display(c.name) for c in pair) for pair in event.pairs)
        who = "You" if isinstance(event, OwnCompetitionEvent) else "Opponent"
        return f"{who} offered a competition: {pairs}."
    if isinstance(event, CompetitionChoiceEvent):
        gets = " + ".join(_suit_display(c.name) for c in event.chooser_gets)
        rest = " + ".join(_suit_display(c.name) for c in event.offerer_gets)
        if event.chooser == player_id:
            return f"You took the pair {gets}; opponent gets {rest}."
        return f"Opponent took the pair {gets}; you get {rest}."
    if isinstance(event, RoundEndEvent):
        return "Round complete — favors updated."
    return str(event)


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------

def advance_to_human_turn() -> None:
    """Resolve chance nodes and AI turns until it's the human's turn (or game over)."""
    global _game, _awaiting_next_round, _single_round_mode, _prev_favors
    while True:
        if _game.winner is not None:
            return

        if _game.phase == "round_complete":
            _prev_favors = dict(_game.favors)
            _game.update_favors()
            w = _game.check_winner()
            if w is not None:
                _game.winner = w
                return
            if _single_round_mode:
                _game.winner = -1  # tie
                return
            _awaiting_next_round = True
            return

        cp = _game.current_player
        if cp is None:
            return

        if cp == CHANCE_PLAYER:
            _game.execute_action(_game.sample_chance_action(_rng))
            continue

        if cp != _human_player:
            info = _game.get_info_state(cp)
            dist = _ai_agent.get_action_distribution(info)
            _game.execute_action(Game.sample_action_from_distribution(dist, _rng))
            continue

        break  # human's turn


# ---------------------------------------------------------------------------
# State serialization
# ---------------------------------------------------------------------------

def state_to_json() -> dict:
    g = _game
    hp = _human_player

    # Geisha favor state
    geishas = [
        {"name": suit.name, "rank": suit.rank, "favor": g.favors[suit]}
        for suit in _variant["config"].suits
    ]

    # Pending offer awaiting a response: SPLIT (2 cards), GIFT (3) or
    # COMPETITION (4; pairs are cards[0:2] and cards[2:4]).
    pending_offer = None
    if g.pending_offer is not None:
        offerer_id, offer_type, cards = g.pending_offer
        pending_offer = {
            "offerer": offerer_id,
            "type": offer_type.value,
            "cards": [c.suit.name for c in cards],
        }
    pending_split = pending_offer if pending_offer and pending_offer["type"] == "split" else None

    # Human player state
    hp_state = g.players[hp]
    human = {
        "hand": [c.suit.name for c in hp_state.hand],
        "collected": [c.suit.name for c in hp_state.collected_cards],
        "discarded_revealed": [c.suit.name for c in hp_state.discarded_revealed],
        "discarded_hidden": [c.suit.name for c in hp_state.discarded_hidden],
        "reserved": [c.suit.name for c in hp_state.reserved_cards],
        "used_actions": [at.value for at in hp_state.used_actions],
    }

    # AI player state
    ai_id = 1 - hp
    ai_state = g.players[ai_id]
    ai = {
        "hand_count": len(ai_state.hand),
        "collected": [c.suit.name for c in ai_state.collected_cards],
        "discarded_revealed": [c.suit.name for c in ai_state.discarded_revealed],
        "discarded_hidden_count": len(ai_state.discarded_hidden),
        "reserved_count": len(ai_state.reserved_cards),
        "used_actions": [at.value for at in ai_state.used_actions],
    }

    # Legal actions (only populated when it's the human's turn)
    legal_actions = []
    if g.winner is None and g.current_player == hp:
        for action in g.get_legal_actions():
            legal_actions.append({
                "action_type": action.action_type.value,
                "cards": [c.suit.name for c in action.cards],
                "choice": action.choice,
            })

    # Human-readable event log for current round
    # Each entry: {"bold": bool, "text": str}
    log = []
    if g.info_state_history[hp]:
        tpr = _variant["config"].turns_per_round
        last_turn = -1
        for event in g.info_state_history[hp][-1]:
            if isinstance(event, RoundStartEvent):
                first = "You" if event.starting_player == hp else "AI"
                log.append({"bold": False, "text": f"Round {g.round_count} starts — first to act: {first}."})
                continue
            t = event.turn
            if t != last_turn:
                last_turn = t
                if t >= tpr:
                    label = "End of Round"
                else:
                    actor = (g.round_starting_player + t) % 2
                    whose = "You" if actor == hp else "AI"
                    label = f"Turn {t + 1}/{tpr} \u2014 {whose}"
                log.append({"bold": True, "text": label})
            log.append({"bold": False, "text": _event_to_log(event, hp)})

    return {
        "round": g.round_count,
        "turn": g.turn_count,
        "turns_per_round": _variant["config"].turns_per_round,
        "deck_count": g.deck.total(),
        "variant_display": _variant["config"].name,
        "win_note": _build_win_note(_variant["config"]),
        "win_geisha_count": _variant["config"].win_geisha_count,
        "win_favor_points": _variant["config"].win_favor_points,
        "game_mode": "single_round" if _single_round_mode else "full_game",
        "round_end_summary": (
            _build_round_end_summary(g, hp, _variant["config"])
            if _awaiting_next_round else None
        ),
        "game_end_summary": (
            _build_round_end_summary(g, hp, _variant["config"])
            if g.winner is not None else None
        ),
        "winner": g.winner,
        "human_player": hp,
        "current_player": g.current_player if g.winner is None else None,
        "pending_split": pending_split,
        "pending_offer": pending_offer,
        "geishas": geishas,
        "human": human,
        "ai": ai,
        "legal_actions": legal_actions,
        "log": log,
    }


def _unavailable_reason(agent_cls):
    """Why `agent_cls` can't be played right now, or None.  Agents that depend
    on an artifact (e.g. a solver's policy table) define unavailable_reason()."""
    check = getattr(agent_cls, "unavailable_reason", None)
    return check() if check is not None else None


def _prepare_new_game(data: dict) -> dict:
    """Validate a /new_game request and build the new game without touching
    the current one.  Raises ValueError with a message for the player."""
    key = _variant_key(data.get("variant"))
    variant = _VARIANTS[key]
    human_player = int(data.get("human_player", 0))
    if human_player not in (0, 1):
        raise ValueError(f"human_player must be 0 or 1, got {human_player}")

    agent_name = data.get("agent", "random")
    agent_cls = next((e[key] for e in _AGENT_REGISTRY if e["id"] == agent_name), None)
    if agent_cls is None:
        agent_cls = RandomAgent
    reason = _unavailable_reason(agent_cls)
    if reason is not None:
        raise ValueError(reason)

    game = variant["game_state_cls"]()
    initial_favors = data.get("initial_favors", {})
    favors = dict(game.favors)
    for suit in favors:
        if suit.name in initial_favors:
            value = int(initial_favors[suit.name])
            if value not in (-1, 0, 1):
                raise ValueError(f"favor for {suit.name} must be -1, 0 or 1, got {value}")
            favors[suit] = value
    game.favors = favors   # assigned, not edited in place: the C++ engine returns a copy
    # A round can only start from favors where nobody has won yet: the game
    # ends as soon as someone wins, so such states never occur in play.
    winner = game.check_winner()
    if winner is not None:
        who = "You" if winner == human_player else "The AI"
        raise ValueError(f"{who} would already have won with these favors. "
                         "Choose favors where nobody has won yet.")

    return {"human_player": human_player, "agent": agent_cls(), "variant": variant,
            "single_round": data.get("game_mode") == "single_round", "game": game}


def _deserialize_action(data: dict) -> Action:
    at = ActionType(data["action_type"])
    cards = [Card(_variant["suit"][name]) for name in data.get("cards", [])]
    return Action(at, cards, choice=data.get("choice"))


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress per-request logs

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
        with open(html_path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_html()
        elif self.path == '/agent_list':
            result = []
            for entry in _AGENT_REGISTRY:
                result.append({
                    "id":     entry["id"],
                    "label":  entry["label"],
                    "tiny":   entry["tiny"]   is not None,
                    "medium": entry["medium"] is not None,
                    "full":   entry["full"]   is not None,
                    # Why the agent can't be played right now (e.g. a missing
                    # solver artifact), or null.
                    "unavailable": _unavailable_reason(entry["tiny"] or entry["medium"] or entry["full"]),
                })
            self._send_json(result)
        else:
            self.send_error(404)

    def do_POST(self):
        global _game, _human_player, _ai_agent, _rng, _awaiting_next_round, _single_round_mode, _prev_favors

        if self.path == "/new_game":
            data = self._read_body()
            try:
                setup = _prepare_new_game(data)
            except ValueError as e:
                self._send_json({"error": str(e)}, 400)
                return
            _human_player = setup["human_player"]
            _ai_agent = setup["agent"]
            _rng = random.Random()
            _awaiting_next_round = False
            _single_round_mode = setup["single_round"]
            _variant.update(setup["variant"])
            _prev_favors = {}
            _game = setup["game"]
            advance_to_human_turn()
            self._send_json(state_to_json())

        elif self.path == "/action":
            if _game is None:
                self._send_json({"error": "No game in progress"}, 400)
                return
            try:
                action = _deserialize_action(self._read_body())
                ok = _game.execute_action(action)
                if not ok:
                    self._send_json({"error": "Illegal action"}, 400)
                    return
                advance_to_human_turn()
                self._send_json(state_to_json())
            except Exception as e:
                self._send_json({"error": str(e)}, 400)

        elif self.path == "/favor_setup_info":
            data = self._read_body()
            cfg = _VARIANTS[_variant_key(data.get("variant"))]["config"]
            self._send_json({
                "geishas": [{"name": s.name, "rank": s.rank} for s in cfg.suits],
                # Lets the page avoid favor states where someone has already won.
                "win_geisha_count": cfg.win_geisha_count,
                "win_favor_points": cfg.win_favor_points,
            })

        elif self.path == "/next_round":
            if not _awaiting_next_round:
                self._send_json({"error": "Not at round end"}, 400)
                return
            _awaiting_next_round = False
            _game._begin_round()
            advance_to_human_turn()
            self._send_json(state_to_json())

        else:
            self.send_error(404)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = 8080
    server = HTTPServer(("", port), Handler)
    print(f"Hanamikoji GUI → http://localhost:{port}")
    server.serve_forever()
