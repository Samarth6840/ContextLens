"""
Layer 3 — Creator–Brand Affinity Model (LightGCN-style).

Learns latent creator and brand embeddings from sparse creator×brand
co-occurrence data (creators who featured / engaged with a brand), and scores
how well a brand fits a given creator beyond raw graph co-occurrence.

The architecture is a LightGCN (He et al., "LightGCN: Simplifying and Powering
Graph Convolution Network for Recommendation", SIGIR 2020): a pure linear
message-passing GNN over the bipartite creator→brand graph — no feature
transformation, no nonlinearity, just normalized neighborhood aggregation. It
is trained with a pairwise (Bayesian Personalized Ranking, BPR) loss.

Cold start is handled explicitly and honestly: an UNTRAINED model scores
NOTHING (returns None) and the caller falls back to the category-affinity
heuristic. Only when fitted on real interactions does the learned signal
override/augment the graph signal. This keeps the Phase 2 upgrade from silently
replacing the honest v1 heuristic with an untrained model.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001 - torch optional for this module
    torch = None
    nn = None
    F = None
    _TORCH_AVAILABLE = False


def _normalized_adjacency(interactions: np.ndarray) -> np.ndarray:
    """Return the symmetrically normalized adjacency matrix D^-1/2 A D^-1/2.

    `interactions` is a (num_creators, num_brands) non-negative matrix. The
    bipartite adjacency has shape (n_c + n_b, n_c + n_b):
        [ 0        A  ]
        [ A^T      0  ]
    Row/col degree is computed per node in the bipartite sense so message
    normalization matches LightGCN.
    """
    n_creators, n_brands = interactions.shape
    n = n_creators + n_brands
    adj = np.zeros((n, n), dtype=np.float32)
    adj[:n_creators, n_creators:] = interactions
    adj[n_creators:, :n_creators] = interactions.T
    # Degree per node.
    deg = adj.sum(axis=1)
    deg[deg == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    # D^-1/2 A D^-1/2
    norm = adj * d_inv_sqrt[:, None]
    norm = d_inv_sqrt[None, :] * norm
    return norm


if _TORCH_AVAILABLE:
    class LightGCNModule(nn.Module):
        """Pure linear graph-convolution encoder over a creator/brand bipartite graph.

        Each layer aggregates normalized neighbor embeddings. The final embedding is
        the mean of layer-0 embeddings after each of `n_layers` propagation steps.
        """

        def __init__(
            self,
            n_creators: int,
            n_brands: int,
            embed_dim: int = 64,
            n_layers: int = 3,
        ):
            super().__init__()
            self.n_creators = n_creators
            self.n_brands = n_brands
            self.embed_dim = embed_dim
            self.n_layers = n_layers
            self.creator_emb = nn.Embedding(n_creators, embed_dim)
            self.brand_emb = nn.Embedding(n_brands, embed_dim)
            nn.init.normal_(self.creator_emb.weight, std=0.1)
            nn.init.normal_(self.brand_emb.weight, std=0.1)

        def forward(self, adj: torch.Tensor) -> torch.Tensor:
            """Propagate embeddings over the (sym-normalized) adjacency matrix.

            Args:
                adj: (N, N) float tensor adjacency, N = creators + brands.
            Returns:
                (N, embed_dim) final embeddings for all creators and brands.
            """
            all_emb = torch.cat([self.creator_emb.weight, self.brand_emb.weight], dim=0)
            # LightGCN: keep layer-0 output, then mean across all layers.
            embs = [all_emb]
            x = all_emb
            for _ in range(self.n_layers):
                x = adj @ x
                embs.append(x)
            final = torch.stack(embs, dim=0).mean(dim=0)
            return final

        def creator_embed(self, final: torch.Tensor) -> torch.Tensor:
            return final[: self.n_creators]

        def brand_embed(self, final: torch.Tensor) -> torch.Tensor:
            return final[self.n_creators :]

else:
    class LightGCNModule:  # type: ignore[no-redef]
        """Placeholder when torch is unavailable; construction always raises."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "LightGCNModule requires torch; install torch to use the "
                "creator-brand affinity model."
            )


class CreatorBrandAffinityModel:
    """Trainable LightGCN affinity model + honest cold-start fallback."""

    def __init__(
        self,
        embed_dim: int = 64,
        n_layers: int = 3,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ):
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "CreatorBrandAffinityModel requires torch; install torch to use it."
            )
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[LightGCNModule] = None
        self._creator_index: Dict[str, int] = {}
        self._brand_index: Dict[str, int] = {}
        self._interaction_matrix_cache: Optional[np.ndarray] = None
        self.fitted = False

    @property
    def is_fitted(self) -> bool:
        return self.fitted

    def fit(
        self,
        creator_ids: List[str],
        brand_ids: List[str],
        weights: Optional[List[float]] = None,
        epochs: int = 30,
        seed: int = 0,
    ) -> Dict:
        """Train the model on a list of (creator, brand[, weight]) co-interactions.

        Args:
            creator_ids: creator identifiers, len == n interactions.
            brand_ids: brand identifiers aligned with creator_ids.
            weights: optional positive interaction strength per pair (0-1).
            epochs: number of training epochs.

        Returns:
            dict summary: {"creators": n, "brands": n, "epochs": e, "loss": last}.
        """
        rng = np.random.default_rng(seed)
        pairs = list(zip(creator_ids, brand_ids))
        if not pairs:
            self.fitted = False
            return {"creators": 0, "brands": 0, "epochs": 0, "loss": 0.0}

        # Fix both weight init and negative sampling so two fits with the same
        # seed are reproducible (honest, auditable training).
        torch.manual_seed(seed)

        # Build indexes.
        for c in creator_ids:
            self._creator_index.setdefault(c, len(self._creator_index))
        for b in brand_ids:
            self._brand_index.setdefault(b, len(self._brand_index))

        n_creators = len(self._creator_index)
        n_brands = len(self._brand_index)

        # Interaction matrix (dense, small enough for v1 catalog scale).
        interactions = np.zeros((n_creators, n_brands), dtype=np.float32)
        for i, (c, b) in enumerate(pairs):
            w = weights[i] if weights else 1.0
            interactions[self._creator_index[c], self._brand_index[b]] += float(w)
        self._interaction_matrix_cache = interactions

        adj = _normalized_adjacency(interactions)
        adj_t = torch.as_tensor(adj, dtype=torch.float32, device=self.device)

        self._model = LightGCNModule(
            n_creators, n_brands, embed_dim=self.embed_dim, n_layers=self.n_layers
        ).to(self.device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        # Build training triples (creator, pos_brand, neg_brand) via sampling.
        pos_pairs = [(i, j) for i, j in zip(
            [self._creator_index[c] for c in creator_ids],
            [self._brand_index[b] for b in brand_ids],
        )]
        pos_set = set(pos_pairs)

        loss_val = 0.0
        self._model.train()
        for _ in range(epochs):
            opt.zero_grad()
            final = self._model(adj_t)
            cemb = self._model.creator_embed(final)
            bemb = self._model.brand_embed(final)
            # Sample one neg brand per positive per epoch.
            losses = []
            for (ci, pi) in pos_set:
                # Any brand not interacted with by this creator is a valid negative.
                neg_options = [
                    j for j in range(n_brands)
                    if (ci, j) not in pos_set and interactions[ci, j] == 0
                ]
                if not neg_options:
                    continue
                ni = int(rng.choice(neg_options))
                pos_score = (cemb[ci] * bemb[pi]).sum()
                neg_score = (cemb[ci] * bemb[ni]).sum()
                losses.append(F.softplus(-(pos_score - neg_score)))
                del pos_score, neg_score
            if not losses:
                loss_val = 0.0
                break
            batch = torch.stack(losses).mean()
            batch.backward()
            opt.step()
            loss_val = float(batch.detach().cpu())

        self.fitted = True
        return {
            "creators": n_creators,
            "brands": n_brands,
            "epochs": epochs,
            "loss": loss_val,
        }

    def predict_affinity(
        self,
        creator_id: str,
        brand_ids: List[str],
    ) -> Optional[List[float]]:
        """Score affinity of brands for a creator. Returns None if untrained or if
        the creator/brands are unseen (cold start) — caller falls back to graph."""
        if not self.fitted or self._model is None:
            return None
        if creator_id not in self._creator_index:
            return None
        with torch.no_grad():
            # Recompute embeddings (small graph — cheap).
            # Rebuild adjacency from stored interaction matrix.
            interactions = self._interaction_matrix
            adj = _normalized_adjacency(interactions)
            adj_t = torch.as_tensor(adj, dtype=torch.float32, device=self.device)
            self._model.eval()
            final = self._model(adj_t)
            bemb = self._model.brand_embed(final)
            ci = self._creator_index[creator_id]
            cvec = self._model.creator_embed(final)[ci]
            scores = []
            for b in brand_ids:
                if b not in self._brand_index:
                    scores.append(0.0)
                    continue
                bi = self._brand_index[b]
                s = float((cvec * bemb[bi]).sum().cpu())
                scores.append(max(0.0, float(np.tanh(s))))
            return scores

    @property
    def _interaction_matrix(self) -> np.ndarray:
        return self._interaction_matrix_cache
