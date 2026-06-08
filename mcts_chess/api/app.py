from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from mcts_chess import AlphaZeroNet, Arena, Connect4, GoGame, Gomoku, MCTSEngine, Othello

GAME_MAP: dict[str, type] = {
    "gomoku": Gomoku,
    "othello": Othello,
    "connect4": Connect4,
    "go": GoGame,
}

GAME_CONFIG: dict[str, dict] = {
    "gomoku": {
        "input_channels": 3,
        "board_size": 15,
        "action_size": 225,
        "dirichlet_alpha": 0.03,
    },
    "othello": {
        "input_channels": 3,
        "board_size": 8,
        "action_size": 64,
        "dirichlet_alpha": None,
    },
    "connect4": {
        "input_channels": 3,
        "board_size": 7,
        "action_size": 7,
        "dirichlet_alpha": None,
        "board_height": 6,
        "board_width": 7,
    },
    "go": {
        "input_channels": 19,
        "board_size": 9,
        "action_size": 82,
        "dirichlet_alpha": None,
    },
}

DATA_DIR: Path = Path(os.environ.get("MCTS_CHESS_DATA_DIR", "data"))
MODELS_DIR: Path = DATA_DIR / "models"
GAMES_DIR: Path = DATA_DIR / "game_records"
MODEL_TAGS_FILE: Path = DATA_DIR / "model_tags.json"


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict) -> None:
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)


training_manager = ConnectionManager()


def _extract_mcts_stats(engine: MCTSEngine, top_k: int = 5) -> list[dict]:
    root = engine._last_root
    if root is None:
        return []
    sorted_children = sorted(root.children.items(), key=lambda x: x[1].N, reverse=True)
    top5 = []
    for action, child in sorted_children[:top_k]:
        top5.append({
            "action": action,
            "visits": child.N,
            "Q": round(child.Q / child.N, 4) if child.N > 0 else 0.0,
            "P": round(float(child.P), 6),
        })
    return top5


def _save_game_record(record: dict) -> None:
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{record['game_id']}.json"
    filepath = GAMES_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, default=str)


def _load_model_tags() -> dict[str, list[str]]:
    if MODEL_TAGS_FILE.exists():
        with open(MODEL_TAGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_model_tags(tags: dict[str, list[str]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_TAGS_FILE, "w", encoding="utf-8") as f:
        json.dump(tags, f, indent=2, ensure_ascii=False)


class ModelManager:
    def __init__(self, models_dir: str = "data/models") -> None:
        self.models_dir: Path = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.models: dict[str, AlphaZeroNet] = {}
        self._load_all_models()

    def _load_all_models(self) -> None:
        if not self.models_dir.exists():
            return
        import torch

        for pt_file in self.models_dir.glob("*.pt"):
            version = pt_file.stem
            if version in self.models:
                continue
            for game_type, config in GAME_CONFIG.items():
                try:
                    kwargs = {
                        "input_channels": config["input_channels"],
                        "board_size": config["board_size"],
                        "action_size": config["action_size"],
                    }
                    if "board_height" in config:
                        kwargs["board_height"] = config["board_height"]
                        kwargs["board_width"] = config["board_width"]
                    model = AlphaZeroNet(**kwargs)
                    state_dict = torch.load(pt_file, map_location="cpu", weights_only=True)
                    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                        model.load_state_dict(state_dict["model_state_dict"])
                    else:
                        model.load_state_dict(state_dict)
                    model.eval()
                    self.models[version] = model
                    break
                except Exception:
                    continue

    def get_model(self, version: str) -> AlphaZeroNet | None:
        return self.models.get(version)

    def save_model(self, version: str, model: AlphaZeroNet) -> None:
        import torch

        path = self.models_dir / f"{version}.pt"
        torch.save(model.state_dict(), path)
        self.models[version] = model

    def list_models(self) -> list[str]:
        available: set[str] = set(self.models.keys())
        if self.models_dir.exists():
            for pt_file in self.models_dir.glob("*.pt"):
                available.add(pt_file.stem)
        return sorted(available)

    def list_models_with_metadata(self) -> list[dict]:
        import torch

        result = []
        tags_data = _load_model_tags()
        elo_path = DATA_DIR / "elo_ratings.json"
        elo_data: dict = {}
        if elo_path.exists():
            with open(elo_path, "r", encoding="utf-8") as f:
                elo_data = json.load(f)

        for version in self.list_models():
            pt_file = self.models_dir / f"{version}.pt"
            file_size = 0
            created_time = None
            if pt_file.exists():
                stat = pt_file.stat()
                file_size = stat.st_size
                created_time = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat()

            elo_rating = None
            if version in elo_data:
                val = elo_data[version]
                if isinstance(val, dict) and "rating" in val:
                    elo_rating = val["rating"]
                elif isinstance(val, (int, float)):
                    elo_rating = val

            result.append({
                "version": version,
                "created_time": created_time,
                "file_size": file_size,
                "elo_rating": elo_rating,
                "tags": tags_data.get(version, []),
            })
        return result

    def delete_model(self, version: str) -> bool:
        pt_file = self.models_dir / f"{version}.pt"
        if not pt_file.exists():
            return False
        pt_file.unlink()
        self.models.pop(version, None)
        tags_data = _load_model_tags()
        tags_data.pop(version, None)
        _save_model_tags(tags_data)
        return True

    def get_latest_model(self) -> AlphaZeroNet | None:
        versions = self.list_models()
        if not versions:
            return None
        numeric_versions: list[tuple[float, str]] = []
        for v in versions:
            try:
                numeric_versions.append((float(v), v))
            except ValueError:
                continue
        if not numeric_versions:
            return self.models.get(versions[-1])
        numeric_versions.sort(key=lambda x: x[0])
        return self.models.get(numeric_versions[-1][1])


class GameSession:
    def __init__(
        self,
        session_id: str,
        game_type: str,
        player_mode: str,
        model_version: str | None,
        num_iterations: int = 400,
    ) -> None:
        self.session_id: str = session_id
        self.game_type: str = game_type
        self.player_mode: str = player_mode
        self.model_version: str | None = model_version
        self.game = GAME_MAP[game_type]()
        self.engine = MCTSEngine(num_iterations=num_iterations, use_puct=False)
        self.move_history: list[dict] = []
        self.is_finished: bool = False
        self.start_time: float = time.time()
        self.player1_info: str = "human" if player_mode == "human_vs_ai" else (model_version or "random")
        self.player2_info: str = model_version if player_mode == "human_vs_ai" else (model_version or "random")
        self.initial_board: list = self.game.board.tolist()
        self._last_engine: MCTSEngine | None = None

    def _build_move_record(
        self, action: int, player: int, board_before: np.ndarray, mcts_stats: list[dict] | None = None
    ) -> dict:
        return {
            "action": action,
            "player": player,
            "board_before": board_before.tolist(),
            "mcts_stats": mcts_stats or [],
        }

    def make_human_move(self, action: int) -> dict:
        legal_moves = self.game.get_legal_moves()
        if action not in legal_moves:
            return {"error": "Illegal move", "legal_moves": legal_moves}
        board_before = self.game.board.copy()
        player = self.game.get_current_player()
        self.game = self.game.make_move(action)
        move_record = self._build_move_record(action, player, board_before)
        self.move_history.append(move_record)
        if self.game.is_game_over():
            self.is_finished = True
            self._save_game_record()
        return self.get_state()

    def make_ai_move(self, model: AlphaZeroNet | None = None) -> dict:
        board_before = self.game.board.copy()
        player = self.game.get_current_player()
        if model is not None:
            engine = MCTSEngine(
                num_iterations=self.engine.num_iterations,
                use_puct=True,
                neural_net=model,
            )
            action = engine.get_move(self.game, move_count=len(self.move_history))
            mcts_stats = _extract_mcts_stats(engine)
            self._last_engine = engine
        else:
            action = self.engine.get_move(self.game, move_count=len(self.move_history))
            mcts_stats = _extract_mcts_stats(self.engine)
            self._last_engine = self.engine
        self.game = self.game.make_move(action)
        move_record = self._build_move_record(action, player, board_before, mcts_stats)
        self.move_history.append(move_record)
        if self.game.is_game_over():
            self.is_finished = True
            self._save_game_record()
        return self.get_state()

    def _save_game_record(self) -> None:
        end_time = time.time()
        result = self.game.get_result() if self.game.is_game_over() else None
        winner = "draw"
        if result is not None:
            if result > 0:
                winner = "player1"
            elif result < 0:
                winner = "player2"

        record = {
            "game_id": self.session_id,
            "game_type": self.game_type,
            "player1": self.player1_info,
            "player2": self.player2_info,
            "start_time": datetime.fromtimestamp(self.start_time, tz=timezone.utc).isoformat(),
            "end_time": datetime.fromtimestamp(end_time, tz=timezone.utc).isoformat(),
            "duration_seconds": round(end_time - self.start_time, 2),
            "result": result,
            "winner": winner,
            "num_moves": len(self.move_history),
            "initial_board": self.initial_board,
            "moves": self.move_history,
        }
        _save_game_record(record)

    def get_state(self) -> dict:
        result = None
        if self.game.is_game_over():
            result = self.game.get_result()
        return {
            "board": self.game.board.tolist(),
            "current_player": self.game.get_current_player(),
            "legal_moves": self.game.get_legal_moves(),
            "is_game_over": self.game.is_game_over(),
            "result": result,
            "move_history": self.move_history,
        }


app = FastAPI(title="MCTS Chess API")

model_manager = ModelManager(models_dir=str(MODELS_DIR))

sessions: dict[str, GameSession] = {}


class CreateSessionRequest(BaseModel):
    game_type: str
    player_mode: str
    model_version: str | None = None
    num_iterations: int = 400


class MoveRequest(BaseModel):
    action: int


class TagRequest(BaseModel):
    tag: str


class CompareRequest(BaseModel):
    version1: str
    version2: str
    num_games: int = 20
    game_type: str = "gomoku"


@app.get("/games")
def list_games() -> dict:
    return {
        "games": [
            {
                "type": game_type,
                "config": GAME_CONFIG[game_type],
            }
            for game_type in GAME_MAP
        ]
    }


@app.get("/models")
def list_models() -> dict:
    return {"models": model_manager.list_models()}


@app.get("/models/metadata")
def list_models_metadata() -> dict:
    return {"models": model_manager.list_models_with_metadata()}


@app.delete("/models/{version}")
def delete_model(version: str) -> dict:
    success = model_manager.delete_model(version)
    if success:
        return {"status": "deleted", "version": version}
    return {"error": f"Model version {version} not found"}


@app.post("/models/{version}/tags")
def add_model_tag(version: str, request: TagRequest) -> dict:
    tags_data = _load_model_tags()
    if version not in tags_data:
        tags_data[version] = []
    if request.tag not in tags_data[version]:
        tags_data[version].append(request.tag)
    _save_model_tags(tags_data)
    return {"version": version, "tags": tags_data[version]}


@app.delete("/models/{version}/tags/{tag}")
def remove_model_tag(version: str, tag: str) -> dict:
    tags_data = _load_model_tags()
    if version in tags_data and tag in tags_data[version]:
        tags_data[version].remove(tag)
        _save_model_tags(tags_data)
    return {"version": version, "tags": tags_data.get(version, [])}


@app.post("/models/compare")
def compare_models(request: CompareRequest) -> dict:
    model1 = model_manager.get_model(request.version1)
    model2 = model_manager.get_model(request.version2)
    if model1 is None:
        return {"error": f"Model {request.version1} not found or not loaded"}
    if model2 is None:
        return {"error": f"Model {request.version2} not found or not loaded"}

    game_class = GAME_MAP.get(request.game_type)
    if game_class is None:
        return {"error": f"Unknown game type: {request.game_type}"}

    arena = Arena(
        game_class=game_class,
        num_games=request.num_games,
        num_iterations=400,
    )
    result = arena.evaluate_models(
        model_challenger=model1,
        model_champion=model2,
        model_challenger_id=request.version1,
        model_champion_id=request.version2,
    )
    return result


@app.get("/game-records")
def list_game_records() -> dict:
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
    return {"records": records}


@app.get("/game-records/{game_id}")
def get_game_record(game_id: str) -> dict:
    filepath = GAMES_DIR / f"{game_id}.json"
    if not filepath.exists():
        return {"error": "Game record not found"}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/sessions")
def create_session(request: CreateSessionRequest) -> dict:
    if request.game_type not in GAME_MAP:
        return {"error": f"Unknown game type: {request.game_type}"}

    session_id = str(uuid.uuid4())
    session = GameSession(
        session_id=session_id,
        game_type=request.game_type,
        player_mode=request.player_mode,
        model_version=request.model_version,
        num_iterations=request.num_iterations,
    )
    if request.model_version is not None:
        model = model_manager.get_model(request.model_version)
        if model is not None:
            game_config = GAME_CONFIG[request.game_type]
            engine = MCTSEngine(
                num_iterations=request.num_iterations,
                use_puct=True,
                neural_net=model,
                dirichlet_alpha=game_config.get("dirichlet_alpha"),
            )
            session.engine = engine
    sessions[session_id] = session
    state = session.get_state()
    state["session_id"] = session_id
    return state


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    session = sessions.get(session_id)
    if session is None:
        return {"error": "Session not found"}
    state = session.get_state()
    state["session_id"] = session_id
    return state


@app.post("/sessions/{session_id}/move")
def make_move(session_id: str, request: MoveRequest) -> dict:
    session = sessions.get(session_id)
    if session is None:
        return {"error": "Session not found"}
    if session.is_finished:
        return {"error": "Game is already finished"}
    result = session.make_human_move(request.action)
    result["session_id"] = session_id
    return result


@app.post("/sessions/{session_id}/ai-move")
def make_ai_move(session_id: str) -> dict:
    session = sessions.get(session_id)
    if session is None:
        return {"error": "Session not found"}
    if session.is_finished:
        return {"error": "Game is already finished"}
    model = None
    if session.model_version is not None:
        model = model_manager.get_model(session.model_version)
    result = session.make_ai_move(model=model)
    result["session_id"] = session_id
    return result


@app.websocket("/ws/game/{session_id}")
async def websocket_game(websocket: WebSocket, session_id: str) -> None:
    session = sessions.get(session_id)
    if session is None:
        await websocket.close(code=4004, reason="Session not found")
        return
    await websocket.accept()
    try:
        state = session.get_state()
        await websocket.send_text(json.dumps({"type": "state", **state}))
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue
            msg_type = message.get("type")
            if msg_type == "move":
                action = message.get("action")
                if action is None or not isinstance(action, int):
                    await websocket.send_text(json.dumps({"type": "error", "message": "Invalid action"}))
                    continue
                if session.is_finished:
                    await websocket.send_text(json.dumps({"type": "error", "message": "Game is already finished"}))
                    continue
                result = session.make_human_move(action)
                if "error" in result:
                    await websocket.send_text(json.dumps({"type": "error", "message": result["error"]}))
                    continue
                if session.is_finished:
                    await websocket.send_text(json.dumps({"type": "game_over", "result": result.get("result")}))
                else:
                    await websocket.send_text(json.dumps({"type": "state", **result}))
            elif msg_type == "ai_move":
                if session.is_finished:
                    await websocket.send_text(json.dumps({"type": "error", "message": "Game is already finished"}))
                    continue
                model = None
                if session.model_version is not None:
                    model = model_manager.get_model(session.model_version)
                result = session.make_ai_move(model=model)
                if session.is_finished:
                    await websocket.send_text(json.dumps({"type": "game_over", "result": result.get("result")}))
                else:
                    await websocket.send_text(json.dumps({"type": "state", **result}))
            else:
                await websocket.send_text(json.dumps({"type": "error", "message": f"Unknown message type: {msg_type}"}))
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket) -> None:
    await training_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        training_manager.disconnect(websocket)


async def emit_training_event(event: dict) -> None:
    await training_manager.broadcast(event)


def emit_training_event_sync(event: dict) -> None:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(emit_training_event(event))
        else:
            loop.run_until_complete(emit_training_event(event))
    except RuntimeError:
        try:
            asyncio.run(emit_training_event(event))
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
