"""Memorilla memory modality handlers.

``MemorillaEncoder`` embeds the documents of each payload with a frozen sentence-embedding model and compresses them
into a fixed number of memory vectors with a trainable ``memorilla.memory.MemoryModule``. ``IdentityProjection`` passes
those vectors through unchanged, since the memory module already projects into the decoder's embedding space.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from memorilla.memory import MemoryModule
from sentence_transformers import SentenceTransformer

from skyrl_train.modalities.handlers import ModalityEncoderProtocol, ModalityProjectorProtocol

PROJECTION_INIT_STD = 0.01


class MemorillaEncoder(nn.Module, ModalityEncoderProtocol):
    """Turns collections of text documents into memory vectors for the decoder.

    A payload is either a sequence of document strings or a mapping with a ``documents`` sequence and an optional
    ``query`` string. The memory module is conditioned on the embedding of the query when one is given and on the mean
    document embedding otherwise. A payload without documents yields zero vectors.

    Only the memory module is trainable. The embedding model is frozen, loaded on first use and kept out of
    ``state_dict()``, so checkpoints hold the memory module alone.
    """

    def __init__(
        self,
        modality_id: str,
        role: str,
        *,
        embedding_model: str = "Qwen/Qwen3-Embedding-4B",
        embedding_device: str = "cuda",
        embedding_dim: int = 2560,
        output_dim: int = 2560,
        num_memories: int = 8,
        num_heads: int = 8,
        num_self_attn_layers: int = 1,
        num_cross_attn_layers: int = 2,
        dropout: float = 0.1,
        checkpoint_path: str | None = None,
    ) -> None:
        """Build the memory module.

        Without a checkpoint, the output projection starts at N(0, ``PROJECTION_INIT_STD``) with zero bias, so the
        first memory tokens are small perturbations of the prompt rather than large random embeddings.

        Args:
            modality_id: Modality name in the SkyRL config.
            role: Handler role assigned by SkyRL (``"encoder"``).
            embedding_model: Sentence-embedding model used to embed documents and queries.
            embedding_device: Device for the embedding model.
            embedding_dim: Hidden size of the embedding model.
            output_dim: Hidden size of the decoder.
            num_memories: Memory tokens per payload; must equal the modality's ``max_placeholder_tokens``.
            num_heads: Attention heads in the memory module.
            num_self_attn_layers: Self-attention layers over the documents.
            num_cross_attn_layers: Cross-attention refinement blocks.
            dropout: Dropout probability in the memory module.
            checkpoint_path: Optional Memorilla checkpoint (directory or ``memory.pt``) to start from; its
                architecture must match the arguments above.
        """
        super().__init__()
        self.modality_id = modality_id
        self.role = role
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_memories = num_memories

        self.memory = MemoryModule(
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            num_memories=num_memories,
            num_heads=num_heads,
            num_self_attn_layers=num_self_attn_layers,
            num_cross_attn_layers=num_cross_attn_layers,
            dropout=dropout,
        )
        nn.init.normal_(self.memory.memory_projection.weight, mean=0.0, std=PROJECTION_INIT_STD)
        nn.init.zeros_(self.memory.memory_projection.bias)
        if checkpoint_path is not None:
            pretrained = MemoryModule.from_pretrained(checkpoint_path)
            self.memory.load_state_dict(pretrained.state_dict(), strict=True)
            logger.info("Loaded memory module for modality `{}` from {}.", modality_id, checkpoint_path)

        self._embedding_model_name = embedding_model
        self._embedding_device = embedding_device
        self._embedding_model: list[SentenceTransformer] = []  # a list keeps it out of the module tree

    def encode(self, payloads: Sequence[Any]) -> list[torch.Tensor]:
        """Compute memory vectors for a batch of payloads.

        Args:
            payloads: One payload per placeholder occurrence.

        Returns:
            One ``[num_memories, output_dim]`` tensor per payload, on the memory module's device and dtype.
        """
        reference = next(self.memory.parameters())
        device, dtype = reference.device, reference.dtype
        outputs = [torch.zeros(self.num_memories, self.output_dim, device=device, dtype=dtype) for _ in payloads]

        documents = [self._documents(payload) for payload in payloads]
        active = [index for index, docs in enumerate(documents) if docs]
        if not active:
            return outputs

        embeddings = self._embed([text for index in active for text in documents[index]], device, dtype)
        max_docs = max(len(documents[index]) for index in active)
        doc_embeds = torch.zeros(len(active), max_docs, self.embedding_dim, device=device, dtype=dtype)
        padding_mask = torch.ones(len(active), max_docs, dtype=torch.bool, device=device)
        offset = 0
        for row, index in enumerate(active):
            count = len(documents[index])
            doc_embeds[row, :count] = embeddings[offset : offset + count]
            padding_mask[row, :count] = False
            offset += count

        question_embeds = self._question_embeddings([payloads[index] for index in active], doc_embeds, padding_mask)
        with torch.enable_grad():
            memory = self.memory(
                doc_embeds=doc_embeds.detach(), doc_padding_mask=padding_mask, question_embeds=question_embeds.detach()
            )
        for row, index in enumerate(active):
            outputs[index] = memory[row].to(dtype=dtype)
        return outputs

    def _question_embeddings(
        self, payloads: Sequence[Any], doc_embeds: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return one query embedding per payload: its embedded query, or its mean document embedding.

        Args:
            payloads: Payloads that have at least one document.
            doc_embeds: Padded document embeddings ``[batch, num_docs, embedding_dim]``.
            padding_mask: ``[batch, num_docs]`` with True at padding positions.

        Returns:
            Query embeddings ``[batch, embedding_dim]``.
        """
        valid = (~padding_mask).unsqueeze(-1).float()
        question_embeds = ((doc_embeds * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)).to(doc_embeds.dtype)

        queries = [self._query(payload) for payload in payloads]
        rows = [row for row, query in enumerate(queries) if query]
        if rows:
            embedded = self._embed([queries[row] for row in rows], doc_embeds.device, doc_embeds.dtype)
            question_embeds[rows] = embedded
        return question_embeds

    def _embed(self, texts: list[str], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Embed texts with the frozen embedding model.

        Args:
            texts: Texts to embed.
            device: Device of the returned tensor.
            dtype: Dtype of the returned tensor.

        Returns:
            Embeddings ``[len(texts), embedding_dim]``.
        """
        if not self._embedding_model:
            logger.info("Loading embedding model {} on {}.", self._embedding_model_name, self._embedding_device)
            self._embedding_model.append(
                SentenceTransformer(
                    self._embedding_model_name,
                    model_kwargs={"device_map": self._embedding_device, "torch_dtype": torch.bfloat16},
                    tokenizer_kwargs={"padding_side": "left"},
                )
            )
        with torch.no_grad():
            embeddings = self._embedding_model[0].encode(
                texts, convert_to_tensor=True, show_progress_bar=False, device=self._embedding_device
            )
        return embeddings.to(device=device, dtype=dtype)

    @staticmethod
    def _documents(payload: Any) -> list[str]:
        """Return the non-empty, stripped document texts of a payload.

        Args:
            payload: A sequence of strings, a mapping with ``documents``, a single string or None.

        Returns:
            The document texts.
        """
        if isinstance(payload, Mapping):
            payload = payload.get("documents", [])
        if isinstance(payload, str):
            payload = [payload]
        if not isinstance(payload, Sequence):
            return []
        return [text.strip() for text in payload if isinstance(text, str) and text.strip()]

    @staticmethod
    def _query(payload: Any) -> str:
        """Return the stripped query of a payload, or an empty string when it has none.

        Args:
            payload: A memory payload.

        Returns:
            The query text.
        """
        query = payload.get("query") if isinstance(payload, Mapping) else None
        return query.strip() if isinstance(query, str) else ""


class IdentityProjection(nn.Module, ModalityProjectorProtocol):
    """Passes memory vectors through unchanged."""

    def __init__(self, modality_id: str, role: str) -> None:
        """Store the handler identity.

        Args:
            modality_id: Modality name in the SkyRL config.
            role: Handler role assigned by SkyRL (``"projection"``).
        """
        super().__init__()
        self.modality_id = modality_id
        self.role = role

    def project(self, features: torch.Tensor) -> torch.Tensor:
        """Return the memory vectors of one occurrence.

        Args:
            features: ``[num_memories, hidden_size]`` memory vectors.

        Returns:
            The same tensor.

        Raises:
            ValueError: If ``features`` is not two-dimensional.
        """
        if features.dim() != 2:
            raise ValueError(f"Projection for `{self.modality_id}` expects a 2D tensor, got {tuple(features.shape)}.")
        return features
