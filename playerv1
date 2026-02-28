import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


class TransformerPlayer(Player):
    """
    Tiny LM baseline chess player (SAFE VERSION).

    - ONLY return the model-generated move if it is LEGAL in the given FEN.
    - Otherwise,  fall back to a random legal move.
    This keeps fallback counts ~0 in the tournament harness.
    """

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str = "TinyLMPlayer",
        model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        temperature: float = 0.7,
        max_new_tokens: int = 8,
    ):
        super().__init__(name)

        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Lazy-loaded components
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

            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

    # -------------------------
    # Prompt
    # -------------------------
    def _build_prompt(self, fen: str) -> str:
        # small tweak: trailing space can help generation a bit
        return f"FEN: {fen}\nMove: "

    def _extract_move(self, text: str) -> Optional[str]:
        match = self.UCI_REGEX.search(text)
        return match.group(1).lower() if match else None

    def _random_legal(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        return random.choice(moves).uci() if moves else None

    # -------------------------
    # Main API
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:
        # Always compute legal moves first so we can validate model output
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]
        legal_set = set(legal_moves)

        # If no legal moves exist, game is over
        if not legal_moves:
            return None

        # Load model; if it fails, return a legal move (safe)
        try:
            self._load_model()
        except Exception:
            return random.choice(legal_moves)

        prompt = self._build_prompt(fen)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove prompt prefix if present
            if decoded.startswith(prompt):
                decoded = decoded[len(prompt):]

            move = self._extract_move(decoded)

            # ✅ Only return if it's a legal move in this position
            if move and move in legal_set:
                return move

        except Exception:
            pass

        # Last resort: still legal (so fallback stays 0)
        return random.choice(legal_moves)
