from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class FeatureConfig:
    min_k_ratio: float = 0.2
    max_length: int = 512
    device: str = "cpu"


class FeatureExtractor:
    def __init__(self, target_model_name: str, reference_model_name: str, config: FeatureConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        self.reference_tokenizer = AutoTokenizer.from_pretrained(reference_model_name)

        self.target_model = AutoModelForCausalLM.from_pretrained(target_model_name).to(self.device).eval()
        self.reference_model = AutoModelForCausalLM.from_pretrained(reference_model_name).to(self.device).eval()

        if self.target_tokenizer.pad_token is None:
            self.target_tokenizer.pad_token = self.target_tokenizer.eos_token
        if self.reference_tokenizer.pad_token is None:
            self.reference_tokenizer.pad_token = self.reference_tokenizer.eos_token

    @torch.no_grad()
    def _token_losses(self, text: str, model, tokenizer) -> np.ndarray:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        input_ids = enc["input_ids"].to(self.device)
        if input_ids.shape[1] < 2:
            return np.array([0.0], dtype=np.float32)
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        token_losses = (-token_log_probs).squeeze(0).detach().cpu().numpy()
        return token_losses.astype(np.float32)

    def _zlib_ratio(self, loss: float, text: str) -> float:
        compressed_len = max(1, len(zlib.compress(text.encode("utf-8"))))
        return float(loss / compressed_len)

    def extract_one(self, text: str) -> Dict[str, float]:
        target_losses = self._token_losses(text, self.target_model, self.target_tokenizer)
        reference_losses = self._token_losses(text, self.reference_model, self.reference_tokenizer)

        target_avg_loss = float(np.mean(target_losses))
        reference_avg_loss = float(np.mean(reference_losses))

        k = max(1, int(np.ceil(len(target_losses) * self.config.min_k_ratio)))
        min_k_losses = np.partition(target_losses, -k)[-k:]
        min_k_logprob = float(np.mean(-min_k_losses))

        return {
            "loss": target_avg_loss,
            "reference": target_avg_loss - reference_avg_loss,
            "min_k": min_k_logprob,
            "zlib": self._zlib_ratio(target_avg_loss, text),
        }

    def extract_batch(self, texts: Iterable[str]) -> List[Dict[str, float]]:
        return [self.extract_one(t) for t in texts]
