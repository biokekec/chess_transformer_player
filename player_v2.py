import chess
import random
import torch
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


class TransformerPlayer(Player):
    """
    Player v2: legal-move scoring (no free generation).
    Always returns a legal move by scoring each legal move under the LM
    and choosing the best one.
    """

    def __init__(
        self,
        name: str = "PlayerV2",
        model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        batch_size: int = 64,
    ):
        super().__init__(name)

        self.model_id = model_id
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            print(f"[{self.name}] Loading {self.model_id} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)
            self.model.eval()

    def _build_prompt(self, fen: str) -> str:
        return f"FEN: {fen}\nMove: "

    @torch.inference_mode()
    def _best_legal_by_logprob(self, prompt: str, legal_uci: List[str]) -> Optional[str]:
        if not legal_uci:
            return None

        # tokenize prompt once to get boundary
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        best_move = None
        best_score = -1e30

        for i in range(0, len(legal_uci), self.batch_size):
            chunk = legal_uci[i:i + self.batch_size]
            texts = [prompt + mv for mv in chunk]

            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(self.device)

            logits = self.model(**enc).logits  # [B, T, V]
            logprobs = torch.log_softmax(logits, dim=-1)

            input_ids = enc.input_ids
            attn = enc.attention_mask

            # log p(token_t | token_<t>) for actual tokens
            token_lp = logprobs[:, :-1, :].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B, T-1]

            # mask prompt+padding; keep continuation tokens only
            cont_mask = attn[:, 1:].clone()  # [B, T-1]
            cut = max(prompt_len - 1, 0)
            cont_mask[:, :cut] = 0

            lengths = cont_mask.sum(dim=1).clamp_min(1)
            scores = (token_lp * cont_mask).sum(dim=1) / lengths  # avg logprob per token

            j = int(torch.argmax(scores).item())
            score = float(scores[j].item())
            if score > best_score:
                best_score = score
                best_move = chunk[j]

        return best_move

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_uci = [m.uci() for m in board.legal_moves]
        if not legal_uci:
            return None

        try:
            self._load_model()
        except Exception:
            return random.choice(legal_uci)

        prompt = self._build_prompt(fen)

        try:
            mv = self._best_legal_by_logprob(prompt, legal_uci)
            if mv is not None:
                return mv
        except Exception:
            pass

        return random.choice(legal_uci)
