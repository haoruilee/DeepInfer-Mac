import sys, numpy as np, mlx.core as mx
sys.path.insert(0, "tests"); sys.path.insert(0, ".")
import test_parity as T, test_dspark as D
torch = T.torch
saved = T._no_quant(True)
rm, mm = T.build(index_topk=64, candidate_topk_blocks=64, **D.MTP)
ds = D.DSpark(mm.a, mm.w, mm)
toks = np.random.default_rng(7).integers(100, 120000, size=30).tolist(); P = 25
out_ids, _, main_hidden = rm(torch.tensor([toks[:P]]), 0); rm.forward_spec(out_ids, main_hidden)
st = mm.new_state(); _, hid = mm.forward(st, toks[:P], all_logits=True, want_hidden=True)
print("prefill hidden cos", T._cmp(main_hidden[0].float().numpy(), np.array(hid.astype(mx.float32)))[0])
ds.seed(st, T._mx(main_hidden[0]), list(range(P)))  # seed with the REFERENCE hiddens
out_ids, rl, mh_ref = rm(torch.tensor([[toks[P]]]), P)
r_ids, r_logits, r_conf = rm.forward_spec(out_ids, mh_ref, P)
ml, mh = mm.forward(st, [toks[P]], want_hidden=True)
print("decode hidden cos", T._cmp(mh_ref[0].float().numpy(), np.array(mh.astype(mx.float32)))[0])
d, dlog, conf = ds.draft(st, int(out_ids[0]), T._mx(mh_ref[0, 0]), P, D.SamplingParams(temperature=0), np.random.default_rng(0))
print("with reference hiddens: logit cos", [round(T._cmp(r_logits[0, j].float().numpy(), dlog[j])[0], 6) for j in range(4)])
print("conf", torch.sigmoid(r_conf[0].float()).numpy().round(4), conf.round(4), "drafts", r_ids[0,1:].tolist(), d)
