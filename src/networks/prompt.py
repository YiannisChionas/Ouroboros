"""Prompt pool for Learning to Prompt (L2P)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptPool(nn.Module):
    """Pool of M prompt key-value pairs selected via cosine similarity.

    Args:
        pool_size  (int): Number of prompts in the pool (M)
        top_k      (int): Number of prompts selected per forward pass (K)
        prompt_len (int): Number of tokens per prompt
        embed_dim  (int): Token embedding dimension
    """

    def __init__(self, pool_size: int, top_k: int, prompt_len: int, embed_dim: int):
        super().__init__()
        self.pool_size  = pool_size
        self.top_k      = top_k
        self.prompt_len = prompt_len
        self.embed_dim  = embed_dim

        self.keys   = nn.Parameter(torch.empty(pool_size, embed_dim))
        self.values = nn.Parameter(torch.empty(pool_size, prompt_len, embed_dim))

        nn.init.trunc_normal_(self.keys,   std=0.02)
        nn.init.trunc_normal_(self.values, std=0.02)

    def forward(self, query: torch.Tensor):
        """Select top-K prompts by cosine similarity to query.

        Args:
            query: (B, D) query vector (cls token before transformer blocks)

        Returns:
            selected:    (B, K*L, D) prompt tokens to prepend to sequence
            reduce_sim:  scalar mean cosine similarity (add -lamb*reduce_sim to loss)
        """
        q = F.normalize(query,     dim=-1)   # (B, D)
        k = F.normalize(self.keys, dim=-1)   # (M, D)
        sim = q @ k.T                        # (B, M)

        top_k_sim, top_k_idx = sim.topk(self.top_k, dim=-1)   # (B, K)
        reduce_sim = top_k_sim.mean()

        B, K = top_k_idx.shape
        L, D = self.prompt_len, self.embed_dim
        idx = top_k_idx.unsqueeze(-1).unsqueeze(-1).expand(B, K, L, D)  # (B,K,L,D)
        vals = self.values.unsqueeze(0).expand(B, -1, -1, -1)           # (B,M,L,D)
        selected = vals.gather(1, idx).view(B, K * L, D)                # (B,K*L,D)

        return selected, reduce_sim
