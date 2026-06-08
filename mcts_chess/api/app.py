from __future__ import annotations

import asyncio
import json
import os
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

    def make_human_move(self, action: int) -> dict:
        legal_moves = self.game.get_legal_moves()
        if action not in legal_moves:
            return {"error": "Illegal move", "legal_moves": legal_moves}
        self.game = self.game.make_move(action)
        self.move_history.append({"action": action, "player": self.game.get_current_player()})
        if self.game.is_game_over():
            self.is_finished = True
        return self.get_state()

    def make_ai_move(self, model: AlphaZeroNet | None = None) -> dict:
        if model is not None:
            engine = MCTSEngine(
                num_iterations=self.engine.num_iterations,
                use_puct=True,
                neural_net=model,
            )
            action = engine.get_move(self.game, move_count=len(self.move_history))
        else:
            action = self.engine.get_move(self.game, move_count=len(self.move_history))
        self.game = self.game.make_move(action)
        self.move_history.append({"action": action, "player": self.game.get_current_player()})
        if self.game.is_game_over():
            self.is_finished = True
        return self.get_state()

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


@app.post("/sessions")
def create_session(request: CreateSessionRequest) -> dict:
    if request.game_type not in GAME_MAP:
        return {"error": f"Unknown game type: {request.game_type}"}
    import uuid

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
