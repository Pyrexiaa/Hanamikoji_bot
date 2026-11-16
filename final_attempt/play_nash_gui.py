"""Play the submitted tiny_hana policy in the local GUI.

Run: python final_attempt/play_nash_gui.py
The policy covers one round with neutral starting favors, for either seat.
"""
import argparse
from pathlib import Path
import sys


def configure_server(project_root=None):
    """Register the saved policy and reject setups outside its solved game."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    project_root = Path(project_root).resolve()
    sys.path.insert(0, str(project_root))

    from gui import server
    from tiny_hana import Suit, load_policy_table_agent

    policy = load_policy_table_agent(
        str(project_root / "final_attempt" / "tasks" / "my_nash_tiny.json"),
        Suit,
        expected_variant="tiny_hana",
    )
    agent_id = "final_nash"
    server._AGENT_REGISTRY[:] = [
        entry for entry in server._AGENT_REGISTRY if entry["id"] != agent_id
    ]
    server._AGENT_REGISTRY.append({
        "id": agent_id,
        "label": "Final Nash bot (neutral single round)",
        "tiny": lambda: policy,
        "medium": None,
        "full": None,
    })
    prepare_new_game = server._prepare_new_game

    def prepare_supported_game(data):
        if data.get("agent") != agent_id:
            return prepare_new_game(data)
        if data.get("variant") != "tiny_hana":
            raise ValueError("Final Nash bot supports Tiny Hanamikoji only.")
        if data.get("game_mode") != "single_round":
            raise ValueError(
                "Final Nash bot solves one round only. Select Single round."
            )
        setup = prepare_new_game(data)
        if any(favor != 0 for favor in setup["game"].favors.values()):
            raise ValueError(
                "Final Nash bot requires neutral starting favors. "
                "Set all three geishas to Neutral, then click Begin round."
            )
        return setup

    server._prepare_new_game = prepare_supported_game
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = configure_server()
    with server.HTTPServer(("127.0.0.1", args.port), server.Handler) as httpd:
        print(f"Open http://localhost:{args.port}", flush=True)
        print("Choose Tiny Hanamikoji, Final Nash bot, Single round.", flush=True)
        print("On Starting favors, leave all three Neutral, then click Begin round.", flush=True)
        print("Press Ctrl+C to stop.", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
