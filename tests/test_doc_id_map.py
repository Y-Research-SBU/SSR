import numpy as np

from ssr.retrieval.doc_id_map import DocIdMap
from ssr.retrieval.sparse_repr import SparseTokenEmbeddings
from ssr.retrieval.sparse_tensors import sparse_list_to_flat_coo


def test_doc_id_map_string_ids():
    m = DocIdMap()
    assert m.assign("doc-a") == 0
    assert m.assign("doc-b") == 1
    assert m.assign("doc-a") == 0
    gids, start = m.global_starts_for_batch(["doc-c", "doc-d"])
    assert start == 2
    assert list(gids) == [2, 3]
    assert m.external_id(1) == "doc-b"


def test_sparse_list_to_flat_coo_row_layout():
    doc = SparseTokenEmbeddings(
        indices=np.array([[0, 1], [-1, -1]], dtype=np.int32),
        values=np.array([[1.0, 0.5], [0.0, 0.0]], dtype=np.float32),
        n_latents=8,
    )
    coo = sparse_list_to_flat_coo([doc], doc_tokens=2)
    coo = coo.coalesce()
    assert coo.shape == (2, 8)
    assert coo._nnz() == 2
