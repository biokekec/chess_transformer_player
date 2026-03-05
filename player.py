import random
from typing import Optional, List, Tuple, Dict

import chess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


class TransformerPlayer(Player):
    """
    TransformerPlayer: legal-move LM scoring + opening bias + safety + 1-ply material gain + check tie-break.

    Key properties:
    - ALWAYS returns a legal move (fallbacks ~0)
    - LM scores ONLY legal UCI moves by avg logprob of move tokens after the prompt (teacher forcing)
    - Lightweight heuristics: opening bias, safety (avoid hanging pieces for free), material gain, check tie-break
    - Deepcopy-safe: class-level caches prevent re-loading the model per game clone
    """

    # Deepcopy-safe shared caches (Game() deepcopies players)
    _MODEL_CACHE: Dict[tuple, AutoModelForCausalLM] = {}
    _TOKENIZER_CACHE: Dict[str, AutoTokenizer] = {}

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
        name: str = "TransformerPlayer",
        model_id: str = "jmarcinek/pythia160_ft",
        batch_size: int = 64,
        top_k: int = 16,
        opening_plies: int = 16,
        max_candidates: int = 24,
        max_length: int = 128,
        seed: int = 0,
    ):
        super().__init__(name)
        self.model_id = model_id
        self.batch_size = batch_size
        self.top_k = top_k
        self.opening_plies = opening_plies
        self.max_candidates = max_candidates
        self.max_length = max_length

        self.rng = random.Random(seed)

        self.device_key = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None

    # -------------------------
    # Lazy loading + caching
    # -------------------------
    def _load_model(self):
        if self.model is not None and self.tokenizer is not None:
            return

        key = (self.model_id, self.device_key)

        if key in self._MODEL_CACHE:
            self.model = self._MODEL_CACHE[key]
            self.tokenizer = self._TOKENIZER_CACHE[self.model_id]
            return

        tok = AutoTokenizer.from_pretrained(self.model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        if self.device_key == "cuda":
            mdl = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            mdl = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)

        mdl.eval()

        self._MODEL_CACHE[key] = mdl
        self._TOKENIZER_CACHE[self.model_id] = tok
        self.model = mdl
        self.tokenizer = tok

    # -------------------------
    # Prompt
    # -------------------------
    def _build_prompt(self, fen: str) -> str:
        return f"FEN: {fen}\nMove: "

    # -------------------------
    # Opening bias
    # -------------------------
    def _ply_from_fen(self, board: chess.Board) -> int:
        # ply = number of half-moves played so far (approx from fullmove number + side to move)
        # At start: fullmove=1, white to move -> ply=0
        # At start: fullmove=1, black to move -> ply=1
        return (board.fullmove_number - 1) * 2 + (0 if board.turn == chess.WHITE else 1)

    def _opening_choice(self, board: chess.Board, legal_uci: List[str]) -> Optional[str]:
        if self._ply_from_fen(board) >= self.opening_plies:
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
    # Safety: don't hang pieces for free (queen/rooks/minors)
    # -------------------------
    def _piece_hangs_for_free(self, board: chess.Board, my_color: bool, piece_type: int) -> bool:
        squares = list(board.pieces(piece_type, my_color))
        for sq in squares:
            attackers = len(board.attackers(not my_color, sq))
            if attackers == 0:
                continue
            defenders = len(board.attackers(my_color, sq))
            if defenders == 0:
                return True
        return False

    def _passes_safety(self, board_before: chess.Board, move_uci: str) -> bool:
        my_color = board_before.turn
        mv = chess.Move.from_uci(move_uci)

        board_after = board_before.copy(stack=False)
        board_after.push(mv)

        # Avoid leaving any of these pieces en prise with 0 defenders
        if self._piece_hangs_for_free(board_after, my_color, chess.QUEEN):
            return False
        if self._piece_hangs_for_free(board_after, my_color, chess.ROOK):
            return False
        if self._piece_hangs_for_free(board_after, my_color, chess.BISHOP):
            return False
        if self._piece_hangs_for_free(board_after, my_color, chess.KNIGHT):
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
        my_color = board.turn
        before = self._material_balance(board, my_color)

        b2 = board.copy(stack=False)
        b2.push(chess.Move.from_uci(move_uci))

        if b2.is_checkmate():
            return 10_000

        after = self._material_balance(b2, my_color)
        return after - before

    def _gives_check(self, board: chess.Board, move_uci: str) -> bool:
        b2 = board.copy(stack=False)
        b2.push(chess.Move.from_uci(move_uci))
        return b2.is_check()

    # -------------------------
    # LM Scoring (correct with LEFT padding)
    # -------------------------
    @torch.inference_mode()
    def _score_moves(self, prompt: str, legal_uci: List[str]) -> List[Tuple[str, float]]:
        tok = self.tokenizer
        mdl = self.model

        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        scored: List[Tuple[str, float]] = []

        for i in range(0, len(legal_uci), self.batch_size):
            chunk = legal_uci[i : i + self.batch_size]
            texts = [prompt + mv for mv in chunk]

            enc = tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )

            # move tensors to model device (works for device_map too)
            enc = {k: v.to(mdl.device) for k, v in enc.items()}

            logits = mdl(**enc).logits                      # [B, L, V]
            logprobs = torch.log_softmax(logits, dim=-1)    # [B, L, V]

            input_ids = enc["input_ids"]                   # [B, L]
            attn = enc["attention_mask"]                   # [B, L]
            B, L = input_ids.shape

            # token_lp[b, t] = log p(x_{t+1} | x_{<=t}) for each t in [0..L-2]
            token_lp = logprobs[:, :-1, :].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B, L-1]

            # Build continuation mask per row, accounting for left padding
            cont_mask = torch.zeros((B, L - 1), dtype=token_lp.dtype, device=token_lp.device)

            for b in range(B):
                seq_len = int(attn[b].sum().item())
                pad_len = L - seq_len

                # If truncation happened, prompt may be partially truncated; cap prompt_len_effective
                prompt_len_effective = min(prompt_len, seq_len)
                # First move token index in full sequence (0-based, including padding)
                move_start_pos = pad_len + prompt_len_effective
                # token_lp index that predicts token at move_start_pos is move_start_pos - 1
                start_lp = max(move_start_pos - 1, 0)

                # End of real tokens in token_lp space is (pad_len + seq_len - 1) exclusive
                end_lp = pad_len + seq_len - 1
                if start_lp < end_lp:
                    cont_mask[b, start_lp:end_lp] = 1.0

            lengths = cont_mask.sum(dim=1).clamp_min(1.0)
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
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None
            legal_uci = [m.uci() for m in legal_moves]
    
            # Opening bias
            opening_mv = self._opening_choice(board, legal_uci)
            if opening_mv is not None:
                return opening_mv
    
            # Load model
            try:
                self._load_model()
            except Exception:
                return self.rng.choice(legal_uci)
    
            prompt = self._build_prompt(fen)
    
            try:
                ranked = self._score_moves(prompt, legal_uci)
    
                # evaluate top max(top_k, max_candidates)
                candidates = ranked[: max(self.top_k, self.max_candidates)]
    
                best_mv = None
                best_gain = -10_000
                best_check = False
                best_lm = -1e30
    
                for mv, lm_sc in candidates:
                    if not self._passes_safety(board, mv):
                        continue
    
                    gain = self._move_material_gain(board, mv)
                    gives_check = self._gives_check(board, mv)
    
                    if best_mv is None:
                        best_mv, best_gain, best_check, best_lm = mv, gain, gives_check, lm_sc
                        continue
    
                    if gain > best_gain:
                        best_mv, best_gain, best_check, best_lm = mv, gain, gives_check, lm_sc
                    elif gain == best_gain:
                        if gives_check and not best_check:
                            best_mv, best_check, best_lm = mv, gives_check, lm_sc
                        elif gives_check == best_check and lm_sc > best_lm:
                            best_mv, best_lm = mv, lm_sc
    
                if best_mv is not None:
                    return best_mv
    
                # Fallback: best LM move (still legal)
                return ranked[0][0] if ranked else self.rng.choice(legal_uci)
    
            except Exception:
                return self.rng.choice(legal_uci)
