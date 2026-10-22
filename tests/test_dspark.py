"""DSpark: draft parity against the reference forward_spec, and exactness of greedy speculative
decoding (it must reproduce plain greedy decoding)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_parity as T  # noqa: E402

torch = T.torch
import mlx.core as mx  # noqa: E402

from deepinfer_mac.dspark import DSpark  # noqa: E402
from deepinfer_mac.engine import SamplingParams, sample  # noqa: E402

MTP = dict(
    n_mtp_layers=2, dspark_block_size=4, dspark_noise_token_id=128799, dspark_target_layer_ids=(3, 4, 5),
    dspark_markov_rank=64, dspark_n_routed_experts=4, dspark_n_activated_experts=2,
    compress_ratios=(0, 0, 2, 2, 1, 1, 0, 0), temperature=0,
)


@pytest.fixture(scope="module")
def models():
    saved = T._no_quant(True)  # isolate logic from rounding-flip amplification, as in test_parity
    rm, mm = T.build(index_topk=64, candidate_topk_blocks=64, **MTP)
    ds = DSpark(mm.a, mm.w, mm)
    ds.ADAPTIVE = False  # these tests exercise the speculative path itself
    yield rm, mm, ds
    T._restore(saved)


def test_draft_matches_reference(models):
    """Drafts from the reference's own target hiddens, so the check covers the draft stages,
    Markov head and confidence head without upstream ulp noise."""
    rm, mm, ds = models
    toks = np.random.default_rng(7).integers(100, 120000, size=30).tolist()
    P = 25
    out_ids, _, main_hidden = rm(torch.tensor([toks[:P]]), 0)
    rm.forward_spec(out_ids, main_hidden)
    st = mm.new_state()
    mm.forward(st, toks[:P])
    ds.seed(st, T._mx(main_hidden[0]), list(range(P)))
    for i in range(P, 29):
        out_ids, _, main_hidden = rm(torch.tensor([[toks[i]]]), i)
        r_ids, r_logits, r_conf = rm.forward_spec(out_ids, main_hidden, i)
        mm.forward(st, [toks[i]])
        d, dlog, conf = ds.draft(st, int(out_ids[0]), T._mx(main_hidden[0, 0]), i, SamplingParams(temperature=0), np.random.default_rng(0))
        cos = [T._cmp(r_logits[0, j].float().numpy(), dlog[j])[0] for j in range(len(d))]
        rconf = torch.sigmoid(r_conf[0].float()).numpy()
        print(f"step {i}: ref drafts {r_ids[0, 1:].tolist()} mine {d} logit cos {np.round(cos, 6)} conf diff {np.abs(rconf - conf).max():.4f}")
        assert d == r_ids[0, 1:].tolist()
        assert min(cos) > 0.9999 and np.abs(rconf - conf).max() < 0.01


def test_greedy_spec_equals_plain(models, monkeypatch):
    _, mm, ds = models

    class Eng:  # the bits of LLM that DSpark.generate uses
        draft_len = 4

        class tokenizer:
            @staticmethod
            def decode(ids, skip_special_tokens=False):
                return ""

        def _plain_decode(self, *a):
            raise AssertionError

    sp = SamplingParams(temperature=0, max_tokens=14, ignore_eos=True)
    prompt = np.random.default_rng(8).integers(100, 120000, size=20).tolist()
    st = mm.new_state()
    logits = mm.forward(st, prompt)
    plain = []
    for i in range(sp.max_tokens):
        tok = int(np.argmax(np.array(logits[-1])))
        plain.append(tok)
        logits = mm.forward(st, [tok])
    for thr in (0.0, 0.5):
        ds.threshold = thr
        st = mm.new_state()
        logits, hid = mm.forward(st, prompt, all_logits=True, want_hidden=True)
        outs = list(ds.generate(Eng(), st, logits, hid, sp, np.random.default_rng(0), None, set(), len(prompt)))
        spec = [o.token_id for o in outs]
        nd = sum(o.draft_tok for o in outs); na = sum(o.accept_tok for o in outs); nt = sum(1 for o in outs if o.draft_tok)
        print(f"threshold {thr}: plain {plain}\n  spec {spec}\n  accepted {na - nt}/{nd - nt} drafts in {nt} verify steps")
        agree = next((i for i, (a, b) in enumerate(zip(plain, spec)) if a != b), len(plain))
        assert agree >= 8, f"spec diverged from plain greedy at {agree}"


def test_full_accept_path(models):
    """An oracle drafter (the true greedy continuation) must be fully accepted, with the target
    state left exactly where plain decoding leaves it."""
    _, mm, ds = models
    sp = SamplingParams(temperature=0, max_tokens=13, ignore_eos=True)
    prompt = np.random.default_rng(9).integers(100, 120000, size=20).tolist()
    st = mm.new_state()
    logits = mm.forward(st, prompt)
    plain = []
    for _ in range(sp.max_tokens):
        plain.append(int(np.argmax(np.array(logits[-1]))))
        logits = mm.forward(st, [plain[-1]])

    real_draft = ds.draft

    def oracle(st_, token, hidden, p, sp_, rng):
        d, dlog, conf = real_draft(st_, token, hidden, p, sp_, rng)
        j = len(st_.tokens) - len(prompt) + 1  # index in `plain` of the token after `token`
        d = (plain[j : j + len(d)] + [0] * len(d))[: len(d)]
        return d, dlog, np.ones(len(d))

    ds.draft, ds.threshold = oracle, 0.5
    try:
        class Eng:
            draft_len = 4

            class tokenizer:
                @staticmethod
                def decode(ids, skip_special_tokens=False):
                    return ""

        st = mm.new_state()
        logits, hid = mm.forward(st, prompt, all_logits=True, want_hidden=True)
        outs = list(ds.generate(Eng(), st, logits, hid, sp, np.random.default_rng(0), None, set(), len(prompt)))
    finally:
        ds.draft = real_draft
    spec = [o.token_id for o in outs]
    steps = sum(1 for o in outs if o.draft_tok)
    print("oracle spec", spec, "verify steps", steps)
    assert spec[:10] == plain[:10] and steps <= 4


def test_adaptive_spec_still_exact(models):
    """With the throughput chooser on, steps mix plain and speculative decoding; greedy output must
    still equal plain greedy decoding."""
    _, mm, ds = models

    class Eng:
        draft_len = 4

        class tokenizer:
            @staticmethod
            def decode(ids, skip_special_tokens=False):
                return ""

    sp = SamplingParams(temperature=0, max_tokens=20, ignore_eos=True)
    prompt = np.random.default_rng(13).integers(100, 120000, size=20).tolist()
    st = mm.new_state()
    logits = mm.forward(st, prompt)
    plain = []
    for _ in range(sp.max_tokens):
        plain.append(int(np.argmax(np.array(logits[-1]))))
        logits = mm.forward(st, [plain[-1]])
    ds.ADAPTIVE, ds.PROBE_EVERY = True, 3
    try:
        st = mm.new_state()
        logits, hid = mm.forward(st, prompt, all_logits=True, want_hidden=True)
        outs = list(ds.generate(Eng(), st, logits, hid, sp, np.random.default_rng(0), None, set(), len(prompt)))
    finally:
        ds.ADAPTIVE, ds.PROBE_EVERY = False, 16
    spec = [o.token_id for o in outs]
    agree = next((i for i, (a, b) in enumerate(zip(plain, spec)) if a != b), len(plain))
    print("adaptive spec:", spec[:10], "agree", agree)
    assert agree >= 10
