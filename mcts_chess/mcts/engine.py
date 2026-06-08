from __future__ import annotations

import time

import numpy as np

from mcts_chess.games.base import BaseGame


class MCTSNode:
    def __init__(
        self,
        game: BaseGame,
        parent: MCTSNode | None = None,
        action_that_led_here: int | None = None,
    ) -> None:
        self.game = game
        self.parent = parent
        self.action_that_led_here = action_that_led_here
        self.children: dict[int, MCTSNode] = {}
        self.N: int = 0
        self.Q: float = 0.0
        self.P: float = 0.0
        self.is_expanded: bool = False

    def is_leaf(self) -> bool:
        return not self.is_expanded or len(self.children) == 0

    def get_ucb1(self, c: float = 1.414) -> float:
        if self.N == 0:
            return float("inf")
        return self.Q / self.N + c * np.sqrt(np.log(self.parent.N) / self.N)

    def get_puct(self, c_puct: float = 1.5) -> float:
        if self.N == 0:
            return float("inf")
        return self.Q / self.N + c_puct * self.P * np.sqrt(self.parent.N) / (1 + self.N)

    def select_child(self, c_puct: float = 1.5, use_puct: bool = False) -> tuple[int, MCTSNode]:
        best_action = None
        best_child = None
        best_score = float("-inf")
        for action, child in self.children.items():
            if use_puct:
                score = child.get_puct(c_puct)
            else:
                score = child.get_ucb1(c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def expand(self, prior: np.ndarray | None = None) -> None:
        legal_moves = self.game.get_legal_moves()
        num_legal = len(legal_moves)
        self.is_expanded = True
        if num_legal == 0:
            return
        if prior is not None:
            masked_prior = np.zeros_like(prior)
            for move in legal_moves:
                masked_prior[move] = prior[move]
            prior_sum = masked_prior.sum()
            if prior_sum > 0:
                masked_prior /= prior_sum
            else:
                masked_prior = np.zeros_like(prior)
                for move in legal_moves:
                    masked_prior[move] = 1.0 / num_legal
        else:
            masked_prior = np.zeros(self.game.action_size)
            for move in legal_moves:
                masked_prior[move] = 1.0 / num_legal
        for move in legal_moves:
            child_game = self.game.make_move(move)
            child = MCTSNode(game=child_game, parent=self, action_that_led_here=move)
            child.P = masked_prior[move]
            self.children[move] = child
        self.is_expanded = True

    def backpropagate(self, value: float) -> None:
        node = self
        while node is not None:
            node.N += 1
            node.Q += value
            value = -value
            node = node.parent


class MCTSEngine:
    def __init__(
        self,
        num_iterations: int = 800,
        time_limit: float = 5.0,
        c_puct: float = 1.5,
        use_puct: bool = False,
        epsilon: float = 0.25,
        dirichlet_alpha: float | None = None,
        temperature: float = 1.0,
        temperature_threshold: int = 30,
        neural_net=None,
    ) -> None:
        self.num_iterations = num_iterations
        self.time_limit = time_limit
        self.c_puct = c_puct
        self.use_puct = use_puct
        self.epsilon = epsilon
        self.dirichlet_alpha = dirichlet_alpha
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        self.neural_net = neural_net
        self._last_root: MCTSNode | None = None

    def search(self, game: BaseGame) -> tuple[int, np.ndarray]:
        root = MCTSNode(game=game.copy())
        if self.use_puct and self.neural_net is not None:
            features = root.game.get_feature_planes()
            policy, value = self.neural_net.predict(features)
            legal_moves = root.game.get_legal_moves()
            masked_policy = np.zeros(game.action_size, dtype=np.float32)
            for move in legal_moves:
                if move < len(policy):
                    masked_policy[move] = policy[move]
            policy_sum = masked_policy.sum()
            if policy_sum > 0:
                masked_policy /= policy_sum
            else:
                for move in legal_moves:
                    masked_policy[move] = 1.0 / len(legal_moves)
            root.expand(prior=masked_policy)
            root.backpropagate(value)
        else:
            root.expand()

        if self.dirichlet_alpha is not None and len(root.children) > 0:
            self._add_dirichlet_noise(root)

        start_time = time.time()
        for _ in range(self.num_iterations):
            if time.time() - start_time >= self.time_limit:
                break

            node = root
            while not node.is_leaf():
                _, next_node = node.select_child(c_puct=self.c_puct, use_puct=self.use_puct)
                if next_node is None:
                    break
                node = next_node

            if node.game.is_game_over():
                value = node.game.get_result()
            else:
                if self.use_puct and self.neural_net is not None:
                    features = node.game.get_feature_planes()
                    policy, value = self.neural_net.predict(features)
                    legal_moves = node.game.get_legal_moves()
                    masked_policy = np.zeros(game.action_size, dtype=np.float32)
                    for move in legal_moves:
                        if move < len(policy):
                            masked_policy[move] = policy[move]
                    policy_sum = masked_policy.sum()
                    if policy_sum > 0:
                        masked_policy /= policy_sum
                    else:
                        for move in legal_moves:
                            masked_policy[move] = 1.0 / len(legal_moves)
                    node.expand(prior=masked_policy)
                else:
                    node.expand()
                    value = self._simulate(node.game)

            node.backpropagate(value)

        policy = np.zeros(game.action_size, dtype=np.float32)
        total_visits = 0
        for action, child in root.children.items():
            policy[action] = child.N
            total_visits += child.N
        if total_visits > 0:
            policy /= total_visits

        best_action = max(root.children.items(), key=lambda x: x[1].N)[0]
        self._last_root = root
        return best_action, policy

    def _simulate(self, game: BaseGame) -> float:
        sim_game = game.copy()
        while not sim_game.is_game_over():
            legal_moves = sim_game.get_legal_moves()
            if len(legal_moves) == 0:
                break
            move = legal_moves[np.random.randint(len(legal_moves))]
            sim_game = sim_game.make_move(move)
        result = sim_game.get_result()
        return result if result is not None else 0.0

    def _add_dirichlet_noise(self, node: MCTSNode) -> None:
        if self.dirichlet_alpha is None or len(node.children) == 0:
            return
        alpha = self.dirichlet_alpha
        noise = np.random.dirichlet([alpha] * len(node.children))
        i = 0
        for action in node.children:
            node.children[action].P = (1 - self.epsilon) * node.children[action].P + self.epsilon * noise[i]
            i += 1

    def get_move(self, game: BaseGame, move_count: int = 0) -> int:
        best_action, policy = self.search(game)
        if move_count < self.temperature_threshold:
            legal_moves = game.get_legal_moves()
            visit_counts = np.array([policy[a] for a in legal_moves], dtype=np.float64)
            total = visit_counts.sum()
            if total > 0:
                probs = visit_counts / total
            else:
                probs = np.ones(len(legal_moves), dtype=np.float64) / len(legal_moves)
            idx = np.random.choice(len(legal_moves), p=probs)
            return legal_moves[idx]
        return best_action

    def get_search_tree(self, game: BaseGame, max_depth: int = 3, max_children: int = 5) -> dict:
        self.search(game)
        root = self._last_root
        if root is None:
            return {}

        def _build_tree(node: MCTSNode, depth: int) -> dict | None:
            if depth > max_depth:
                return None
            result: dict = {
                "action": node.action_that_led_here,
                "Q": node.Q,
                "N": node.N,
                "P": node.P,
                "children": [],
            }
            sorted_children = sorted(node.children.items(), key=lambda x: x[1].N, reverse=True)
            for action, child in sorted_children[:max_children]:
                child_tree = _build_tree(child, depth + 1)
                if child_tree is not None:
                    result["children"].append(child_tree)
            return result

        return _build_tree(root, 0)
