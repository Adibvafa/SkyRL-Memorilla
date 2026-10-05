"""Tests for the Memorilla memory modality handlers, with a stand-in embedding model."""

from pathlib import Path

import pytest
import torch

handlers = pytest.importorskip("skyrl_train.examples.modalities.memorilla_handlers")

EMBEDDING_DIM = 32
OUTPUT_DIM = 16
NUM_MEMORIES = 4


class HashEmbedder:
    """Deterministic stand-in for a sentence-embedding model."""

    def encode(self, texts: list[str], convert_to_tensor: bool, show_progress_bar: bool, device: str) -> torch.Tensor:
        """Return one pseudo-random vector per text, seeded by its content."""
        vectors = [
            torch.randn(EMBEDDING_DIM, generator=torch.Generator().manual_seed(sum(map(ord, text)))) for text in texts
        ]
        return torch.stack(vectors)


def make_encoder(**kwargs: object) -> "handlers.MemorillaEncoder":
    """Build a small encoder in evaluation mode with the stand-in embedding model."""
    encoder = handlers.MemorillaEncoder(
        "memorilla",
        "encoder",
        embedding_device="cpu",
        embedding_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        num_memories=NUM_MEMORIES,
        num_heads=4,
        **kwargs,
    )
    encoder._embedding_model.append(HashEmbedder())
    return encoder.eval()


def test_outputs_match_memory_module_with_mean_document_query() -> None:
    encoder = make_encoder()
    payloads = [["first document", "second document", " "], [], "a single document"]
    outputs = encoder.encode(payloads)

    assert [tuple(output.shape) for output in outputs] == [(NUM_MEMORIES, OUTPUT_DIM)] * 3
    assert torch.count_nonzero(outputs[1]) == 0

    docs = HashEmbedder().encode(["first document", "second document"], True, False, "cpu").unsqueeze(0)
    with torch.no_grad():
        expected = encoder.memory(doc_embeds=docs, doc_padding_mask=None, question_embeds=docs.mean(dim=1))[0]
    torch.testing.assert_close(outputs[0], expected)


def test_query_payload_uses_query_embedding() -> None:
    encoder = make_encoder()
    output = encoder.encode([{"documents": ["a document"], "query": "where is the key"}])[0]

    docs = HashEmbedder().encode(["a document"], True, False, "cpu").unsqueeze(0)
    query = HashEmbedder().encode(["where is the key"], True, False, "cpu")
    with torch.no_grad():
        expected = encoder.memory(doc_embeds=docs, doc_padding_mask=None, question_embeds=query)[0]
    torch.testing.assert_close(output, expected)


def test_only_memory_module_is_trainable_state() -> None:
    encoder = make_encoder().train()
    assert all(key.startswith("memory.") for key in encoder.state_dict())
    assert encoder.memory.memory_projection.weight.std().item() < 0.05

    encoder.encode([["a document", "another document"]])[0].sum().backward()
    assert all(parameter.grad is not None for parameter in encoder.memory.parameters())


def test_checkpoint_path_loads_weights(tmp_path: Path) -> None:
    source = make_encoder()
    source.memory.save(tmp_path / "memory")

    loaded = make_encoder(checkpoint_path=str(tmp_path / "memory"))
    for name, tensor in source.memory.state_dict().items():
        torch.testing.assert_close(loaded.memory.state_dict()[name], tensor)


def test_identity_projection_checks_shape() -> None:
    projection = handlers.IdentityProjection("memorilla", "projection")
    features = torch.ones(NUM_MEMORIES, OUTPUT_DIM)
    assert projection.project(features) is features
    with pytest.raises(ValueError):
        projection.project(features.unsqueeze(0))
