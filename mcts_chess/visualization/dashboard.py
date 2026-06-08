import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path("data")
MODELS_DIR = DATA_DIR / "models"
GAMES_DIR = DATA_DIR / "games"

GAME_TYPES = {
    "gomoku": {"name": "Gomoku", "board_size": 15, "module": "mcts_chess.games.gomoku"},
    "connect4": {"name": "Connect4", "rows": 6, "cols": 7, "module": "mcts_chess.games.connect4"},
    "othello": {"name": "Othello", "board_size": 8, "module": "mcts_chess.games.othello"},
    "go": {"name": "Go (9x9)", "board_size": 9, "module": "mcts_chess.games.go"},
}


@st.cache_data
def load_training_log() -> list[dict]:
    path = DATA_DIR / "training_log.json"
    if not path.exists():
        return []
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data
def load_elo_ratings() -> dict:
    path = DATA_DIR / "elo_ratings.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data
def load_game_records() -> list[dict]:
    path = DATA_DIR / "game_records.json"
    if not path.exists():
        return []
    with open(path, "r") as f:
        return json.load(f)


def board_to_display(board: np.ndarray, game_type: str) -> go.Figure:
    rows, cols = board.shape
    display_board = np.zeros((rows, cols), dtype=np.float64)

    text_array = np.full((rows, cols), "", dtype=object)
    for r in range(rows):
        for c in range(cols):
            if board[r, c] == 1:
                text_array[r, c] = "X"
            elif board[r, c] == -1:
                text_array[r, c] = "O"

    color_array = np.full((rows, cols), "empty", dtype=object)
    for r in range(rows):
        for c in range(cols):
            if board[r, c] == 1:
                color_array[r, c] = "black"
            elif board[r, c] == -1:
                color_array[r, c] = "white"

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=display_board,
            colorscale=[[0, "#DEB887"], [1, "#DEB887"]],
            showscale=False,
            x=list(range(cols)),
            y=list(range(rows)),
        )
    )

    for r in range(rows):
        for c in range(cols):
            if board[r, c] == 1:
                fig.add_trace(
                    go.Scatter(
                        x=[c],
                        y=[r],
                        mode="markers+text",
                        marker=dict(size=20, color="black", symbol="circle"),
                        text=["X"],
                        textposition="middle center",
                        textfont=dict(color="white", size=12),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            elif board[r, c] == -1:
                fig.add_trace(
                    go.Scatter(
                        x=[c],
                        y=[r],
                        mode="markers+text",
                        marker=dict(size=20, color="white", symbol="circle", line=dict(color="black", width=1)),
                        text=["O"],
                        textposition="middle center",
                        textfont=dict(color="black", size=12),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    fig.update_layout(
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="black",
            dtick=1,
            range=[-0.5, cols - 0.5],
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="black",
            dtick=1,
            range=[rows - 0.5, -0.5],
            scaleanchor="x",
            scaleratio=1 if game_type != "connect4" else rows / cols,
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        height=500,
        plot_bgcolor="#DEB887",
    )

    return fig


def policy_heatmap(policy: np.ndarray, board_shape: tuple[int, ...], legal_moves: list[int]) -> go.Figure:
    rows, cols = board_shape
    heatmap_data = np.zeros((rows, cols), dtype=np.float64)
    for move in legal_moves:
        if move < len(policy):
            r = move // cols
            c = move % cols
            if r < rows and c < cols:
                heatmap_data[r, c] = policy[move]

    fig = go.Figure(
        go.Heatmap(
            z=heatmap_data,
            colorscale="YlOrRd",
            showscale=True,
            colorbar=dict(title="Probability"),
            x=list(range(cols)),
            y=list(range(rows)),
        )
    )

    fig.update_layout(
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="gray", dtick=1, range=[-0.5, cols - 0.5]),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="gray",
            dtick=1,
            range=[rows - 0.5, -0.5],
            scaleanchor="x",
            scaleratio=1,
        ),
        margin=dict(l=20, r=20, t=30, b=20),
        height=500,
        title="Move Probabilities",
    )

    return fig


def page_training_monitor():
    st.header("Training Monitor")

    training_log = load_training_log()
    if not training_log:
        st.info("No training data found. Start training to see metrics here.")
        return

    df = pd.DataFrame(training_log)

    st.subheader("Loss Curves")
    loss_cols = [c for c in ["policy_loss", "value_loss", "total_loss"] if c in df.columns]
    if loss_cols:
        step_col = "step" if "step" in df.columns else df.index
        loss_df = df[loss_cols].copy()
        if "step" in df.columns:
            loss_df["step"] = df["step"]
            st.line_chart(loss_df, x="step", y=loss_cols)
        else:
            st.line_chart(loss_df[loss_cols])

    st.subheader("Self-Play Games")
    if "games_played" in df.columns:
        games_df = pd.DataFrame({"games_played": df["games_played"].cumsum()})
        if "step" in df.columns:
            games_df["step"] = df["step"]
            st.line_chart(games_df, x="step", y="games_played")
        else:
            st.line_chart(games_df)

    st.subheader("Elo Rating Over Time")
    elo_data = load_elo_ratings()
    if elo_data:
        elo_rows = []
        for version, info in elo_data.items():
            if isinstance(info, dict) and "rating" in info:
                elo_rows.append({"version": version, "rating": info["rating"]})
            elif isinstance(info, (int, float)):
                elo_rows.append({"version": version, "rating": info})
        if elo_rows:
            elo_df = pd.DataFrame(elo_rows)
            fig = px.line(elo_df, x="version", y="rating", markers=True, title="Elo Rating by Model Version")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Elo rating data available.")

    st.subheader("Win Rate Matrix")
    win_matrix_path = DATA_DIR / "win_matrix.json"
    if win_matrix_path.exists():
        with open(win_matrix_path, "r") as f:
            win_matrix = json.load(f)
        if win_matrix:
            versions = sorted(set(k.split("_")[0] for k in win_matrix.keys()) | set(k.split("_")[1] for k in win_matrix.keys()))
            mat = np.full((len(versions), len(versions)), np.nan)
            for i, v1 in enumerate(versions):
                for j, v2 in enumerate(versions):
                    key = f"{v1}_{v2}"
                    if key in win_matrix:
                        mat[i, j] = win_matrix[key]
            fig = go.Figure(
                go.Heatmap(
                    z=mat,
                    x=versions,
                    y=versions,
                    colorscale="RdYlGn",
                    zmin=0,
                    zmax=1,
                    colorbar=dict(title="Win Rate"),
                )
            )
            fig.update_layout(title="Win Rate Matrix (Row vs Column)", xaxis_title="Opponent", yaxis_title="Player")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No win rate matrix data available.")


def _render_search_tree(tree: dict, depth: int = 0):
    if not tree:
        return
    action = tree.get("action")
    q = tree.get("Q", 0)
    n = tree.get("N", 0)
    p = tree.get("P", 0)
    label = f"Action={action} | Q={q:.3f} | N={n} | P={p:.4f}" if action is not None else "Root"
    children = tree.get("children", [])
    if children:
        with st.expander(label, expanded=(depth < 1)):
            for child in children:
                _render_search_tree(child, depth + 1)
    else:
        st.text(label)


def page_search_visualization():
    st.header("Search Visualization")

    game_type = st.selectbox("Game Type", list(GAME_TYPES.keys()), format_func=lambda k: GAME_TYPES[k]["name"])

    game_info = GAME_TYPES[game_type]
    game = _create_game(game_type)

    if game is None:
        st.error(f"Could not create game of type: {game_type}")
        return

    if "search_game" not in st.session_state or st.session_state.get("search_game_type") != game_type:
        st.session_state.search_game = game
        st.session_state.search_game_type = game_type

    current_game = st.session_state.search_game

    st.subheader("Current Board")
    fig = board_to_display(current_game.board, game_type)
    st.plotly_chart(fig, use_container_width=True)

    legal_moves = current_game.get_legal_moves()
    st.write(f"Current Player: {'X (1)' if current_game.get_current_player() == 1 else 'O (-1)'}")
    st.write(f"Legal Moves: {len(legal_moves)}")

    st.subheader("MCTS Search Controls")
    num_iterations = st.slider("Number of Iterations", 50, 2000, 400, step=50)
    use_puct = st.checkbox("Use PUCT (Neural Network)", value=False)

    model_version = None
    if use_puct:
        available_models = _get_available_models()
        if available_models:
            model_version = st.selectbox("Model Version", available_models)
        else:
            st.warning("No trained models found. PUCT search requires a neural network model.")

    if st.button("Run MCTS Search"):
        from mcts_chess.mcts.engine import MCTSEngine

        neural_net = None
        if use_puct and model_version:
            neural_net = _load_model(model_version, game_type)

        engine = MCTSEngine(
            num_iterations=num_iterations,
            use_puct=use_puct and neural_net is not None,
            neural_net=neural_net,
        )

        with st.spinner(f"Running MCTS search ({num_iterations} iterations)..."):
            best_action, policy = engine.search(current_game)

        st.success(f"Best action: {best_action}")

        st.subheader("Move Visit Heatmap")
        board_shape = current_game.board.shape
        visit_fig = policy_heatmap(policy, board_shape, legal_moves)
        st.plotly_chart(visit_fig, use_container_width=True)

        st.subheader("Search Tree (Top Nodes)")
        search_tree = engine.get_search_tree(current_game, max_depth=3, max_children=5)
        if search_tree:
            _render_search_tree(search_tree)

        new_game = current_game.make_move(best_action)
        st.session_state.search_game = new_game
        st.rerun()


def page_battle_records():
    st.header("Battle Records")

    records = load_game_records()
    if not records:
        st.info("No game records found. Play some games to see records here.")
        return

    display_rows = []
    for i, rec in enumerate(records):
        display_rows.append(
            {
                "index": i,
                "Date": rec.get("date", "N/A"),
                "Player1": rec.get("player1", "N/A"),
                "Player2": rec.get("player2", "N/A"),
                "Result": rec.get("result", "N/A"),
                "Moves": rec.get("num_moves", len(rec.get("moves", []))),
            }
        )

    df = pd.DataFrame(display_rows)
    st.dataframe(df[["Date", "Player1", "Player2", "Result", "Moves"]], use_container_width=True)

    selected_index = st.selectbox("Select game to replay", df["index"].tolist(), format_func=lambda i: f"Game {i}: {df.iloc[i]['Player1']} vs {df.iloc[i]['Player2']} ({df.iloc[i]['Result']})")

    if selected_index is not None and selected_index < len(records):
        rec = records[selected_index]
        moves = rec.get("moves", [])
        game_type = rec.get("game_type", "gomoku")

        if moves:
            if "replay_step" not in st.session_state:
                st.session_state.replay_step = 0

            step = st.session_state.replay_step
            step = max(0, min(step, len(moves)))

            game = _create_game(game_type)
            if game is not None:
                for m in moves[:step]:
                    game = game.make_move(m)

                fig = board_to_display(game.board, game_type)
                st.plotly_chart(fig, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("⬅ Prev") and st.session_state.replay_step > 0:
                        st.session_state.replay_step -= 1
                        st.rerun()
                with col2:
                    st.write(f"Move {step} / {len(moves)}")
                with col3:
                    if st.button("Next ➡") and st.session_state.replay_step < len(moves):
                        st.session_state.replay_step += 1
                        st.rerun()

                move_info = rec.get("move_stats", [])
                if move_info and step > 0 and step - 1 < len(move_info):
                    stats = move_info[step - 1]
                    st.subheader(f"MCTS Stats for Move {step}")
                    if "policy" in stats:
                        policy_arr = np.array(stats["policy"])
                        legal = game.get_legal_moves()
                        policy_fig = policy_heatmap(policy_arr, game.board.shape, legal)
                        st.plotly_chart(policy_fig, use_container_width=True)
                    if "value" in stats:
                        st.write(f"Value estimate: {stats['value']:.4f}")


def page_human_vs_ai():
    st.header("Human vs AI")

    game_type = st.selectbox("Game Type", list(GAME_TYPES.keys()), format_func=lambda k: GAME_TYPES[k]["name"], key="hvai_game_type")

    available_models = _get_available_models()
    model_version = st.selectbox("AI Model Version", available_models) if available_models else None

    difficulty = st.selectbox("Difficulty (MCTS Iterations)", [100, 400, 800, 1600], index=1)

    game_key = f"human_game_{game_type}"
    if game_key not in st.session_state:
        st.session_state[game_key] = _create_game(game_type)
        st.session_state[f"{game_key}_over"] = False
        st.session_state[f"{game_key}_result"] = None

    game = st.session_state[game_key]
    game_over = st.session_state[f"{game_key}_over"]
    game_result = st.session_state[f"{game_key}_result"]

    fig = board_to_display(game.board, game_type)
    st.plotly_chart(fig, use_container_width=True)

    if game_over:
        if game_result == 1.0:
            st.success("Game Over - You Win! 🎉")
        elif game_result == -1.0:
            st.error("Game Over - AI Wins!")
        else:
            st.info("Game Over - Draw!")
    else:
        st.write(f"Your turn: {'X (1)' if game.get_current_player() == 1 else 'O (-1)'}")

    if not game_over:
        legal_moves = game.get_legal_moves()
        rows, cols = game.board.shape

        if game_type == "connect4":
            clicked_col = st.selectbox("Select Column", list(range(cols)))
            if st.button("Place Stone"):
                if clicked_col in legal_moves:
                    game = game.make_move(clicked_col)
                    st.session_state[game_key] = game

                    if game.is_game_over():
                        st.session_state[f"{game_key}_over"] = True
                        st.session_state[f"{game_key}_result"] = game.get_result()
                        st.rerun()
                    else:
                        ai_action = _run_ai_move(game, game_type, difficulty, model_version)
                        if ai_action is not None:
                            game = game.make_move(ai_action)
                            st.session_state[game_key] = game
                            if game.is_game_over():
                                st.session_state[f"{game_key}_over"] = True
                                st.session_state[f"{game_key}_result"] = game.get_result()
                        st.rerun()
        else:
            board_cols = cols
            col_labels = [str(c) for c in range(board_cols)]
            row_labels = [str(r) for r in range(rows)]

            selected_row = st.selectbox("Row", list(range(rows)))
            selected_col = st.selectbox("Column", list(range(cols)))
            action = selected_row * cols + selected_col

            if st.button("Place Stone"):
                if action in legal_moves:
                    game = game.make_move(action)
                    st.session_state[game_key] = game

                    if game.is_game_over():
                        st.session_state[f"{game_key}_over"] = True
                        st.session_state[f"{game_key}_result"] = game.get_result()
                        st.rerun()
                    else:
                        ai_action = _run_ai_move(game, game_type, difficulty, model_version)
                        if ai_action is not None:
                            game = game.make_move(ai_action)
                            st.session_state[game_key] = game
                            if game.is_game_over():
                                st.session_state[f"{game_key}_over"] = True
                                st.session_state[f"{game_key}_result"] = game.get_result()
                        st.rerun()
                else:
                    st.warning("Invalid move! That position is not legal.")

    if st.button("New Game"):
        st.session_state[game_key] = _create_game(game_type)
        st.session_state[f"{game_key}_over"] = False
        st.session_state[f"{game_key}_result"] = None
        st.rerun()


def _create_game(game_type: str):
    try:
        if game_type == "gomoku":
            from mcts_chess.games.gomoku import Gomoku
            return Gomoku()
        elif game_type == "connect4":
            from mcts_chess.games.connect4 import Connect4
            return Connect4()
        elif game_type == "othello":
            from mcts_chess.games.othello import Othello
            return Othello()
        elif game_type == "go":
            from mcts_chess.games.go import GoGame
            return GoGame()
    except Exception:
        return None
    return None


def _get_available_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    models = []
    for f in sorted(MODELS_DIR.iterdir()):
        if f.is_file() and f.suffix in (".pt", ".pth"):
            models.append(f.stem)
    return models


def _load_model(model_name: str, game_type: str):
    model_path = MODELS_DIR / f"{model_name}.pt"
    if not model_path.exists():
        model_path = MODELS_DIR / f"{model_name}.pth"
    if not model_path.exists():
        return None

    try:
        import torch
        from mcts_chess.network.model import AlphaZeroNet

        info = GAME_TYPES.get(game_type, {})
        if game_type == "connect4":
            board_size = 7
            action_size = 7
            input_channels = 3
        elif game_type == "gomoku":
            board_size = 15
            action_size = 225
            input_channels = 3
        elif game_type == "othello":
            board_size = 8
            action_size = 64
            input_channels = 3
        elif game_type == "go":
            board_size = 9
            action_size = 82
            input_channels = 19
        else:
            return None

        net = AlphaZeroNet(
            input_channels=input_channels,
            board_size=board_size,
            action_size=action_size,
        )
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        net.load_state_dict(state_dict)
        net.eval()
        return net
    except Exception:
        return None


def _run_ai_move(game, game_type: str, difficulty: int, model_version: str | None) -> int | None:
    try:
        from mcts_chess.mcts.engine import MCTSEngine

        neural_net = None
        use_puct = False
        if model_version:
            neural_net = _load_model(model_version, game_type)
            if neural_net is not None:
                use_puct = True

        engine = MCTSEngine(
            num_iterations=difficulty,
            use_puct=use_puct,
            neural_net=neural_net,
        )
        best_action, _ = engine.search(game)
        return best_action
    except Exception:
        legal = game.get_legal_moves()
        return legal[0] if legal else None


def run_dashboard():
    st.set_page_config(page_title="MCTS Chess Dashboard", page_icon="♟", layout="wide")

    page = st.sidebar.selectbox(
        "Navigation",
        ["Training Monitor", "Search Visualization", "Battle Records", "Human vs AI"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Info")
    st.sidebar.write(f"Data dir: `{DATA_DIR}`")
    st.sidebar.write(f"Models dir: `{MODELS_DIR}`")

    if page == "Training Monitor":
        page_training_monitor()
    elif page == "Search Visualization":
        page_search_visualization()
    elif page == "Battle Records":
        page_battle_records()
    elif page == "Human vs AI":
        page_human_vs_ai()


if __name__ == "__main__":
    run_dashboard()
