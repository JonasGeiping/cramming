"""Sanity Check architecture."""
import torch
from typing import Optional


class SanityCheckforPreTraining(torch.nn.Module):
    """Make big go fast."""

    def __init__(self, width, vocab_size):
        super().__init__()
        self.word_embedding = torch.nn.Embedding(vocab_size, width, padding_idx=0)
        self.transform = torch.nn.Linear(width, width, bias=False)

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        embeds = self.word_embedding(input_ids)
        outputs = self.transform(embeds)
        loss = outputs.mean()
        return dict(outputs=outputs, loss=loss)
