import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path("data")
MODELS_DIR = DATA_DIR / "models"
GAMES_DIR = DATA_DIR / "game_records"
MODEL_TAGS_FILE = DATA_DIR / "model_tags.json"

GAME_TYPES = {
    "gomoku": {"name": "Gomoku", "board_size": 15, "module": "mcts_chess.games.gomoku"},
    "connect4": {"name": "Connect4", "rows": 6, "cols": 7, "module": "mcts_chess.games.connect4"},
    "othello": {"name": "Othello", "board_size": 8, "module": "mcts_chess.games.othello"},
    "go": {"name": "Go (9x9)", "board_size": 9, "module": "mcts_chess.games.go"},
}

API_BASE_URL = os.environ.get("MCTS_CHESS_API_URL", "http://localhost:8000")


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
def load_game_records_list() -> list[dict]:
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for json_file in sorted(GAMES_DIR.glob("*.json"), reverse=True):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                record = json.load(f)
            records.append({
                "game_id": record.get("game_id"),
                "game_type": record.get("game_type"),
                "player1": record.get("player1"),
                "player2": record.get("player2"),
                "winner": record.get("winner"),
                "result": record.get("result"),
                "num_moves": record.get("num_moves"),
                "duration_seconds": record.get("duration_seconds"),
                "start_time": record.get("start_time"),
                "end_time": record.get("end_time"),
            })
        except Exception:
            continue
    return records


@st.cache_data
def load_game_record_detail(game_id: str) -> dict | None:
    filepath = GAMES_DIR / f"{game_id}.json"
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_model_tags() -> dict[str, list[str]]:
    if MODEL_TAGS_FILE.exists():
        with open(MODEL_TAGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


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

    if "ws_training_events" not in st.session_state:
        st.session_state.ws_training_events = []

    if "ws_connected" not in st.session_state:
        st.session_state.ws_connected = False

    ws_status_placeholder = st.empty()

    try:
        import websocket as ws_module
        ws_available = True
    except ImportError:
        try:
            from websockets.sync import client as ws_module
            ws_available = True
        except ImportError:
            ws_available = False

    col_ws1, col_ws2 = st.columns([3, 1])
    with col_ws1:
        ws_url = st.text_input("WebSocket URL", f"ws://localhost:8000/ws/training", key="ws_url")
    with col_ws2:
        st.write("")
        st.write("")
        if st.button("Connect" if not st.session_state.ws_connected else "Reconnect"):
            st.session_state.ws_connected = True
            st.session_state.ws_training_events = []

    if st.session_state.ws_connected and ws_available:
        ws_status_placeholder.info("Attempting WebSocket connection... (listening for events)")
        try:
            _listen_ws_events(ws_url)
        except Exception as e:
            ws_status_placeholder.warning(f"WebSocket connection issue: {e}. Showing cached data below.")

    training_log = load_training_log()
    ws_events = st.session_state.get("ws_training_events", [])

    if ws_events:
        st.subheader("Live Training Events")
        live_rows = []
        for ev in ws_events[-20:]:
            ev_type = ev.get("event_type", "unknown")
            ts = ev.get("timestamp", "")
            if ev_type == "selfplay_done":
                live_rows.append({
                    "Time": ts,
                    "Type": "Self-Play Done",
                    "Detail": f"Game {ev.get('game_count', '?')} | Buffer: {ev.get('buffer_size', '?')}",
                })
            elif ev_type == "train_step":
                live_rows.append({
                    "Time": ts,
                    "Type": "Train Step",
                    "Detail": f"Loss: {ev.get('total_loss', 0):.4f} | P: {ev.get('policy_loss', 0):.4f} V: {ev.get('value_loss', 0):.4f}",
                })
            elif ev_type == "eval_done":
                live_rows.append({
                    "Time": ts,
                    "Type": "Eval Done",
                    "Detail": f"WR: {ev.get('win_rate', 0):.2%} | Elo: {ev.get('challenger_elo', '?')}",
                })
            else:
                live_rows.append({"Time": ts, "Type": ev_type, "Detail": json.dumps(ev)[:80]})
        if live_rows:
            st.dataframe(pd.DataFrame(live_rows[::-1]), use_container_width=True, height=300)

    if not training_log and not ws_events:
        st.info("No training data found. Start training to see metrics here.")
        return

    all_log_entries = list(training_log)
    for ev in ws_events:
        if ev.get("event_type") == "train_step" and "step" in ev:
            all_log_entries.append({
                "step": ev.get("step"),
                "policy_loss": ev.get("policy_loss"),
                "value_loss": ev.get("value_loss"),
                "total_loss": ev.get("total_loss"),
            })

    if all_log_entries:
        df = pd.DataFrame(all_log_entries)

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


def _listen_ws_events(ws_url: str):
    try:
        import websocket as ws_client
    except ImportError:
        return

    if "_ws_thread_running" not in st.session_state:
        st.session_state._ws_thread_running = False

    if st.session_state._ws_thread_running:
        return

    st.session_state._ws_thread_running = True

    def on_message(ws, message):
        try:
            event = json.loads(message)
            event["timestamp"] = time.strftime("%H:%M:%S")
            st.session_state.ws_training_events.append(event)
            if len(st.session_state.ws_training_events) > 500:
                st.session_state.ws_training_events = st.session_state.ws_training_events[-500:]
        except Exception:
            pass

    def on_error(ws, error):
        pass

    def on_close(ws, close_status_code, close_msg):
        st.session_state._ws_thread_running = False

    def on_open(ws):
        pass

    def run_ws():
        try:
            ws = ws_client.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception:
            st.session_state._ws_thread_running = False

    thread = threading.Thread(target=run_ws, daemon=True)
    thread.start()


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

    records = load_game_records_list()
    if not records:
        st.info("No game records found. Play some games to see records here.")
        return

    display_rows = []
    for i, rec in enumerate(records):
        winner_str = rec.get("winner", "N/A")
        result_val = rec.get("result")
        if result_val is not None:
            if result_val > 0:
                winner_str = f"{rec.get('player1', 'P1')} wins"
            elif result_val < 0:
                winner_str = f"{rec.get('player2', 'P2')} wins"
            else:
                winner_str = "Draw"

        display_rows.append({
            "index": i,
            "Game ID": rec.get("game_id", "N/A")[:8],
            "Game Type": rec.get("game_type", "N/A"),
            "Player 1": rec.get("player1", "N/A"),
            "Player 2": rec.get("player2", "N/A"),
            "Result": winner_str,
            "Moves": rec.get("num_moves", 0),
            "Duration": f"{rec.get('duration_seconds', 0):.1f}s",
            "Start": rec.get("start_time", "N/A"),
        })

    df = pd.DataFrame(display_rows)
    st.dataframe(
        df[["Game ID", "Game Type", "Player 1", "Player 2", "Result", "Moves", "Duration", "Start"]],
        use_container_width=True,
    )

    selected_index = st.selectbox(
        "Select game to replay",
        df["index"].tolist(),
        format_func=lambda i: f"Game {i}: {df.iloc[i]['Player 1']} vs {df.iloc[i]['Player 2']} ({df.iloc[i]['Result']})",
    )

    if selected_index is not None and selected_index < len(records):
        rec_summary = records[selected_index]
        game_id = rec_summary.get("game_id")

        if game_id:
            record = load_game_record_detail(game_id)
        else:
            record = None

        if record is None:
            st.warning("Could not load game details.")
            return

        moves = record.get("moves", [])
        game_type = record.get("game_type", "gomoku")

        st.subheader("Game Info")
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.write(f"**Player 1:** {record.get('player1', 'N/A')}")
            st.write(f"**Player 2:** {record.get('player2', 'N/A')}")
        with info_col2:
            st.write(f"**Duration:** {record.get('duration_seconds', 0):.1f}s")
            st.write(f"**Total Moves:** {record.get('num_moves', 0)}")
        with info_col3:
            result_val = record.get("result")
            if result_val is not None:
                if result_val > 0:
                    st.write(f"**Winner:** {record.get('player1', 'P1')}")
                elif result_val < 0:
                    st.write(f"**Winner:** {record.get('player2', 'P2')}")
                else:
                    st.write("**Result:** Draw")
            else:
                st.write("**Result:** N/A")

        if moves:
            if "replay_step" not in st.session_state:
                st.session_state.replay_step = 0
            if st.session_state.get("replay_game_id") != game_id:
                st.session_state.replay_step = 0
                st.session_state.replay_game_id = game_id

            step = st.session_state.replay_step
            step = max(0, min(step, len(moves)))

            game = _create_game(game_type)
            if game is None:
                st.error(f"Could not create game of type: {game_type}")
                return

            for m in moves[:step]:
                game = game.make_move(m["action"])

            fig = board_to_display(game.board, game_type)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
            with col1:
                if st.button("⏮ First") and st.session_state.replay_step > 0:
                    st.session_state.replay_step = 0
                    st.rerun()
            with col2:
                if st.button("⬅ Prev") and st.session_state.replay_step > 0:
                    st.session_state.replay_step -= 1
                    st.rerun()
            with col3:
                step_slider = st.slider("Step", 0, len(moves), step, key="replay_slider")
                if step_slider != st.session_state.replay_step:
                    st.session_state.replay_step = step_slider
                    st.rerun()
            with col4:
                if st.button("Next ➡") and st.session_state.replay_step < len(moves):
                    st.session_state.replay_step += 1
                    st.rerun()
            with col5:
                if st.button("Last ⏭"):
                    st.session_state.replay_step = len(moves)
                    st.rerun()

            if step > 0 and step - 1 < len(moves):
                current_move = moves[step - 1]
                mcts_stats = current_move.get("mcts_stats", [])

                st.subheader(f"Move {step} Details")
                detail_col1, detail_col2 = st.columns(2)
                with detail_col1:
                    player_str = "X (1)" if current_move.get("player") == 1 else "O (-1)"
                    st.write(f"**Player:** {player_str}")
                    st.write(f"**Action:** {current_move.get('action')}")
                with detail_col2:
                    if mcts_stats:
                        st.write("**MCTS Top-5 Visits:**")
                        stats_df = pd.DataFrame(mcts_stats)
                        stats_df["Q"] = stats_df["Q"].round(4)
                        stats_df["P"] = stats_df["P"].round(6)
                        stats_df.columns = ["Action", "Visits", "Q", "P"]
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No MCTS stats recorded for this move.")

                if mcts_stats:
                    st.subheader("MCTS Visit Distribution")
                    visit_fig_data = {
                        "action": [s["action"] for s in mcts_stats],
                        "visits": [s["visits"] for s in mcts_stats],
                    }
                    bar_fig = go.Figure(
                        go.Bar(
                            x=[str(a) for a in visit_fig_data["action"]],
                            y=visit_fig_data["visits"],
                            marker_color="steelblue",
                        )
                    )
                    bar_fig.update_layout(
                        xaxis_title="Action",
                        yaxis_title="Visit Count",
                        height=300,
                        margin=dict(l=20, r=20, t=30, b=20),
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)


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
            st.success("Game Over - You Win!")
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


def page_model_management():
    st.header("Model Management")

    models = _get_models_with_metadata()

    if not models:
        st.info("No model checkpoints found. Train a model first.")
        return

    rows = []
    for m in models:
        size_mb = m["file_size"] / (1024 * 1024) if m["file_size"] else 0
        elo_str = f"{m['elo_rating']:.0f}" if m["elo_rating"] is not None else "N/A"
        tags_str = ", ".join(m.get("tags", [])) if m.get("tags") else "-"
        rows.append({
            "Version": m["version"],
            "Created": m.get("created_time", "N/A")[:19] if m.get("created_time") else "N/A",
            "Size (MB)": f"{size_mb:.2f}",
            "Elo": elo_str,
            "Tags": tags_str,
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    tab_tag, tab_compare, tab_delete = st.tabs(["Tag Model", "Compare Models", "Delete Model"])

    with tab_tag:
        st.subheader("Tag a Model Version")
        tag_versions = [m["version"] for m in models]
        tag_version = st.selectbox("Select Version", tag_versions, key="tag_version_select")
        tag_options = ["production", "baseline", "candidate", "experimental"]
        existing_tags = load_model_tags()
        current_tags = existing_tags.get(tag_version, [])

        new_tag = st.selectbox("Select Tag to Add", tag_options + ["Custom..."], key="tag_select")
        if new_tag == "Custom...":
            new_tag = st.text_input("Enter custom tag", key="custom_tag_input")

        if st.button("Add Tag"):
            if tag_version and new_tag:
                _add_model_tag(tag_version, new_tag)
                st.success(f"Added tag '{new_tag}' to version {tag_version}")
                st.rerun()

        if current_tags:
            st.write(f"**Current tags for {tag_version}:** {', '.join(current_tags)}")
            remove_tag = st.selectbox("Select Tag to Remove", current_tags, key="remove_tag_select")
            if st.button("Remove Tag"):
                _remove_model_tag(tag_version, remove_tag)
                st.success(f"Removed tag '{remove_tag}' from version {tag_version}")
                st.rerun()

    with tab_compare:
        st.subheader("Compare Two Model Versions")
        compare_versions = [m["version"] for m in models]
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            version1 = st.selectbox("Model 1 (Challenger)", compare_versions, key="compare_v1")
        with col_v2:
            version2_options = [v for v in compare_versions if v != version1]
            version2 = st.selectbox("Model 2 (Champion)", version2_options, key="compare_v2") if version2_options else None

        game_type = st.selectbox("Game Type", list(GAME_TYPES.keys()), format_func=lambda k: GAME_TYPES[k]["name"], key="compare_game_type")
        num_games = st.slider("Number of Games", 4, 100, 20, key="compare_num_games")

        if version1 and version2:
            if st.button("Run Arena Comparison"):
                with st.spinner(f"Running {num_games} games between {version1} and {version2}..."):
                    result = _run_arena_comparison(version1, version2, num_games, game_type)
                if result and "error" not in result:
                    st.success("Comparison complete!")
                    res_col1, res_col2, res_col3 = st.columns(3)
                    with res_col1:
                        st.metric("Wins", result.get("wins", 0))
                    with res_col2:
                        st.metric("Losses", result.get("losses", 0))
                    with res_col3:
                        st.metric("Draws", result.get("draws", 0))
                    st.write(f"**Win Rate:** {result.get('win_rate', 0):.2%}")
                    st.write(f"**{version1} Elo:** {result.get('challenger_elo', 'N/A')}")
                    st.write(f"**{version2} Elo:** {result.get('champion_elo', 'N/A')}")
                elif result:
                    st.error(result.get("error", "Unknown error"))

    with tab_delete:
        st.subheader("Delete a Model Checkpoint")
        delete_versions = [m["version"] for m in models]
        delete_version = st.selectbox("Select Version to Delete", delete_versions, key="delete_version_select")

        if delete_version:
            st.warning(f"You are about to delete model version **{delete_version}**. This action cannot be undone.")
            confirm = st.checkbox(f"I confirm I want to delete version {delete_version}", key="delete_confirm")
            if st.button("Delete Model", disabled=not confirm):
                result = _delete_model(delete_version)
                if result and result.get("status") == "deleted":
                    st.success(f"Model version {delete_version} has been deleted.")
                    st.rerun()
                else:
                    st.error(result.get("error", "Failed to delete model"))


def _get_models_with_metadata() -> list[dict]:
    try:
        import requests
        resp = requests.get(f"{API_BASE_URL}/models/metadata", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("models", [])
    except Exception:
        pass

    if not MODELS_DIR.exists():
        return []

    tags_data = load_model_tags()
    elo_path = DATA_DIR / "elo_ratings.json"
    elo_data: dict = {}
    if elo_path.exists():
        with open(elo_path, "r") as f:
            elo_data = json.load(f)

    result = []
    for f in sorted(MODELS_DIR.iterdir()):
        if f.is_file() and f.suffix in (".pt", ".pth"):
            version = f.stem
            stat = f.stat()
            elo_rating = None
            if version in elo_data:
                val = elo_data[version]
                if isinstance(val, dict) and "rating" in val:
                    elo_rating = val["rating"]
                elif isinstance(val, (int, float)):
                    elo_rating = val
            result.append({
                "version": version,
                "created_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_ctime)),
                "file_size": stat.st_size,
                "elo_rating": elo_rating,
                "tags": tags_data.get(version, []),
            })
    return result


def _add_model_tag(version: str, tag: str) -> dict | None:
    try:
        import requests
        resp = requests.post(f"{API_BASE_URL}/models/{version}/tags", json={"tag": tag}, timeout=5)
        return resp.json()
    except Exception:
        tags_data = load_model_tags()
        if version not in tags_data:
            tags_data[version] = []
        if tag not in tags_data[version]:
            tags_data[version].append(tag)
        with open(MODEL_TAGS_FILE, "w", encoding="utf-8") as f:
            json.dump(tags_data, f, indent=2)
        return {"version": version, "tags": tags_data[version]}


def _remove_model_tag(version: str, tag: str) -> dict | None:
    try:
        import requests
        resp = requests.delete(f"{API_BASE_URL}/models/{version}/tags/{tag}", timeout=5)
        return resp.json()
    except Exception:
        tags_data = load_model_tags()
        if version in tags_data and tag in tags_data[version]:
            tags_data[version].remove(tag)
        with open(MODEL_TAGS_FILE, "w", encoding="utf-8") as f:
            json.dump(tags_data, f, indent=2)
        return {"version": version, "tags": tags_data.get(version, [])}


def _run_arena_comparison(version1: str, version2: str, num_games: int, game_type: str) -> dict | None:
    try:
        import requests
        resp = requests.post(
            f"{API_BASE_URL}/models/compare",
            json={
                "version1": version1,
                "version2": version2,
                "num_games": num_games,
                "game_type": game_type,
            },
            timeout=300,
        )
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def _delete_model(version: str) -> dict | None:
    try:
        import requests
        resp = requests.delete(f"{API_BASE_URL}/models/{version}", timeout=5)
        return resp.json()
    except Exception:
        pt_file = MODELS_DIR / f"{version}.pt"
        if pt_file.exists():
            pt_file.unlink()
            return {"status": "deleted", "version": version}
        return {"error": f"Model version {version} not found"}


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
        ["Training Monitor", "Search Visualization", "Battle Records", "Human vs AI", "Model Management"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Info")
    st.sidebar.write(f"Data dir: `{DATA_DIR}`")
    st.sidebar.write(f"Models dir: `{MODELS_DIR}`")
    st.sidebar.write(f"API URL: `{API_BASE_URL}`")

    if page == "Training Monitor":
        page_training_monitor()
    elif page == "Search Visualization":
        page_search_visualization()
    elif page == "Battle Records":
        page_battle_records()
    elif page == "Human vs AI":
        page_human_vs_ai()
    elif page == "Model Management":
        page_model_management()


if __name__ == "__main__":
    run_dashboard()
