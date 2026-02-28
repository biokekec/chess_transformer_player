import chess
import random
import torch
from typing import Optional, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


class TransformerPlayer(Player):
    """
    Player v4: legal-move LM scoring + opening bias + safety + 1-ply material gain.

    Design goals:
    - ALWAYS returns a legal move (fallback ~0)
    - Avoids obviously bad early moves (opening bias)
    - Avoids hanging the queen for free (cheap tactical filter)
    - Converts RandomPlayer blunders by preferring captures / material gain
      among top-scored LM candidates (still not an engine).
    """

    PIECE_VALUE = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    def __init__(
        self,
        name: str = "PlayerV4",
        model_id: str = "distilgpt2",
        batch_size: int = 64,
        top_k: int = 10,          # consider more candidates than v3
        opening_plies: int = 12,  # slightly longer opening help
        max_candidates: int = 12  # how many top LM moves we consider for material filter
    ):
        super().__init__(name)
        self.model_id = model_id
        self.batch_size = batch_size
        self.top_k = top_k
        self.opening_plies = opening_plies
        self.max_candidates = max_candidates

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    # -------------------------
    # Lazy loading
    # -------------------------
    def _load_model(self):
        if self.model is None:
            print(f"[{self.name}] Loading {self.model_id} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

            self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)
            self.model.eval()

    # -------------------------
    # Prompt
    # -------------------------
    def _build_prompt(self, fen: str) -> str:
        return f"FEN: {fen}\nMove: "

    # -------------------------
    # Opening bias
    # -------------------------
    def _opening_choice(self, board: chess.Board, legal_uci: List[str]) -> Optional[str]:
        if len(board.move_stack) >= self.opening_plies:
            return None

        if board.turn == chess.WHITE:
            prefs = ["e2e4", "d2d4", "g1f3", "c2c4", "e2e3", "d2d3"]
        else:
            prefs = ["e7e5", "c7c5", "d7d5", "g8f6", "e7e6", "c7c6"]

        legal_set = set(legal_uci)
        for mv in prefs:
            if mv in legal_set:
                return mv
        return None

    # -------------------------
    # Safety: don't hang queen for free
    # -------------------------
    def _queen_hangs_for_free(self, board: chess.Board, my_color: bool) -> bool:
        queens = list(board.pieces(chess.QUEEN, my_color))
        if not queens:
            return False
        qsq = queens[0]

        attackers = len(board.attackers(not my_color, qsq))
        if attackers == 0:
            return False

        defenders = len(board.attackers(my_color, qsq))
        return defenders == 0

    def _passes_safety(self, board_before: chess.Board, move_uci: str) -> bool:
        my_color = board_before.turn
        mv = chess.Move.from_uci(move_uci)

        board_after = board_before.copy()
        board_after.push(mv)

        if self._queen_hangs_for_free(board_after, my_color):
            return False

        return True

    # -------------------------
    # Material helpers (1-ply)
    # -------------------------
    def _material(self, board: chess.Board, color: bool) -> int:
        total = 0
        for ptype, val in self.PIECE_VALUE.items():
            total += len(board.pieces(ptype, color)) * val
        return total

    def _material_balance(self, board: chess.Board, my_color: bool) -> int:
        return self._material(board, my_color) - self._material(board, not my_color)

    def _move_material_gain(self, board: chess.Board, move_uci: str) -> int:
        """
        Material gain from making move_uci (1-ply lookahead).
        Huge reward if it delivers checkmate.
        """
        my_color = board.turn
        before = self._material_balance(board, my_color)

        b2 = board.copy()
        b2.push(chess.Move.from_uci(move_uci))

        if b2.is_checkmate():
            return 10_000

        after = self._material_balance(b2, my_color)
        return after - before

    # -------------------------
    # LM Scoring
    # -------------------------
    @torch.inference_mode()
    def _score_moves(self, prompt: str, legal_uci: List[str]) -> List[Tuple[str, float]]:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        scored: List[Tuple[str, float]] = []

        for i in range(0, len(legal_uci), self.batch_size):
            chunk = legal_uci[i:i + self.batch_size]
            texts = [prompt + mv for mv in chunk]

            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                add_special_tokens=False,
            ).to(self.device)

            logits = self.model(**enc).logits
            logprobs = torch.log_softmax(logits, dim=-1)

            input_ids = enc.input_ids
            attn = enc.attention_mask

            token_lp = logprobs[:, :-1, :].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

            cont_mask = attn[:, 1:].clone()
            cut = max(prompt_len - 1, 0)
            cont_mask[:, :cut] = 0

            lengths = cont_mask.sum(dim=1).clamp_min(1)
            scores = (token_lp * cont_mask).sum(dim=1) / lengths

            for mv, sc in zip(chunk, scores.tolist()):
                scored.append((mv, float(sc)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # -------------------------
    # Main API
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_uci = [m.uci() for m in board.legal_moves]
        if not legal_uci:
            return None

        # Opening bias
        opening_mv = self._opening_choice(board, legal_uci)
        if opening_mv is not None:
            return opening_mv

        # Load model
        try:
            self._load_model()
        except Exception:
            return random.choice(legal_uci)

        prompt = self._build_prompt(fen)

        try:
            ranked = self._score_moves(prompt, legal_uci)

            # Consider top N ranked moves (LM) and pick best by:
            # 1) safety filter
            # 2) material gain (1-ply)
            # 3) LM score as tie-breaker
            candidates = ranked[: max(self.top_k, self.max_candidates)]

            best_mv = None
            best_gain = -10_000
            best_lm = -1e30

            for mv, lm_sc in candidates:
                if not self._passes_safety(board, mv):
                    continue

                gain = self._move_material_gain(board, mv)

                if (gain > best_gain) or (gain == best_gain and lm_sc > best_lm):
                    best_gain = gain
                    best_lm = lm_sc
                    best_mv = mv

            if best_mv is not None:
                return best_mv

            # fallback: best LM move (still legal)
            return ranked[0][0] if ranked else random.choice(legal_uci)

        except Exception:
            return random.choice(legal_uci)
