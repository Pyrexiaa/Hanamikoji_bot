"""
Simple demo runner for the standard (medium) Hanamikoji variant.
"""

from medium_hana import Game, RandomAgent


def main():
    """Run a short example game."""
    print("Hanamikoji Demo - Example Game\n")

    agent0 = RandomAgent()
    agent1 = RandomAgent()

    game = Game(agent0, agent1, verbose=True)
    winner = game.play_game(max_rounds=5)

    print(f"\n{'='*50}")
    print(f"Final Result: Player {winner} wins!" if winner >= 0 else "Draw!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

