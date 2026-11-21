"""数据管线：明文(<=1024B) <-> LZMA 定参数压缩流(<=1024B)。

只保留明文与压缩流都不超过 MAX_LEN 的样本对，输出定长张量。
"""

import json
import lzma
import os
import random

import torch

MAX_LEN = 1024
PAD = 256
VOCAB = 257
FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 6}]


def compress(raw: bytes) -> bytes:
    return lzma.compress(raw, format=lzma.FORMAT_RAW, filters=FILTERS)


def decompress(stream: bytes) -> bytes:
    return lzma.decompress(stream, format=lzma.FORMAT_RAW, filters=FILTERS)


WORDS = (
    "the of and to in is are was were be been being for on with as by at from that this "
    "not have has had will would can could should may might must do does did done make made "
    "system buffer stream block frame index token packet channel session header payload "
    "error warn info debug trace retry abort commit merge branch remote local origin "
    "quick brown fox jumps over lazy dog while parsing manifest schema version digest "
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda sigma omega "
    "cache flush drain queue worker thread lock mutex atomic barrier shard replica "
).split()

KEYS = [
    "id", "name", "type", "size", "offset", "codec", "level", "flags", "hash", "path",
    "count", "status", "region", "epoch", "shard", "digest", "payload", "version",
]


def _pick_len(rng, lo, hi):
    return rng.randint(lo, hi)


def gen_text(rng):
    n = _pick_len(rng, 40, 900)
    out = []
    size = 0
    while size < n:
        line = " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 12)))
        line = line.capitalize() + rng.choice([".", ".", ".", "?", "!", ";", ":"])
        out.append(line)
        size += len(line) + 1
    return "\n".join(out).encode()[:n]


def gen_log(rng):
    n = _pick_len(rng, 120, 1000)
    out = []
    size = 0
    ts = rng.randint(1_600_000_000, 1_800_000_000)
    idx = rng.randint(1, 5000)
    while size < n:
        line = "[%d] worker-%d %s seq=%d size=%d %s" % (
            ts,
            rng.randint(0, 15),
            rng.choice(["flush", "commit", "retry", "ack", "drop", "merge"]),
            idx,
            rng.randint(1, 4096),
            rng.choice(WORDS),
        )
        out.append(line)
        size += len(line) + 1
        ts += rng.randint(1, 9)
        idx += rng.randint(1, 3)
    return "\n".join(out).encode()[:n]


def gen_json(rng):
    n = _pick_len(rng, 60, 950)
    records = []
    size = 2
    while size < n:
        rec = {}
        for _ in range(rng.randint(2, 6)):
            k = rng.choice(KEYS)
            v = rng.choice(
                [
                    rng.randint(0, 10**6),
                    rng.choice(WORDS),
                    "%08x" % rng.getrandbits(32),
                    rng.random() < 0.5,
                    None,
                ]
            )
            rec[k] = v
        text = json.dumps(rec, sort_keys=True)
        records.append(text)
        size += len(text) + 2
    return ("[" + ",\n ".join(records) + "]").encode()[:n]


def gen_repeat(rng):
    n = _pick_len(rng, 32, 1000)
    mode = rng.randint(0, 2)
    if mode == 0:
        return bytes([rng.getrandbits(8)]) * n
    if mode == 1:
        pat = bytes(rng.getrandbits(8) for _ in range(rng.randint(2, 8)))
        return (pat * (n // len(pat) + 1))[:n]
    block = bytes(rng.getrandbits(8) for _ in range(rng.randint(8, 32)))
    reps = n // len(block) + 1
    return (block * reps)[:n]


def gen_random(rng):
    return bytes(rng.getrandbits(8) for _ in range(_pick_len(rng, 16, 1000)))


def gen_mixed(rng):
    n = _pick_len(rng, 80, 1000)
    parts = []
    size = 0
    while size < n:
        k = rng.randint(0, 3)
        if k == 0:
            p = rng.choice(WORDS).encode() * rng.randint(1, 6)
        elif k == 1:
            p = bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 12)))
        elif k == 2:
            p = ("%d" % rng.randint(0, 999)).encode()
        else:
            p = rng.choice([b"{}", b"[]", b"\n", b"\x00", b"=", b", "])
        parts.append(p)
        size += len(p)
    return b"".join(parts)[:n]


GENERATORS = [gen_text, gen_text, gen_log, gen_json, gen_repeat, gen_random, gen_mixed]


def sample_raw(rng):
    return rng.choice(GENERATORS)(rng)


def build_split(n, seed, seen, max_bytes=MAX_LEN):
    rng = random.Random(seed)
    raws, comps = [], []
    tries = 0
    while len(raws) < n and tries < n * 400:
        tries += 1
        raw = sample_raw(rng)
        if not raw or len(raw) > max_bytes:
            continue
        comp = compress(raw)
        if not comp or len(comp) > max_bytes or comp in seen:
            continue
        seen.add(comp)
        raws.append(raw)
        comps.append(comp)
    if len(raws) < n:
        raise SystemExit("只生成了 %d 条唯一样本，目标 %d" % (len(raws), n))
    return raws, comps


def to_tensors(comps, raws):
    n = len(raws)
    inp = torch.full((n, MAX_LEN), PAD, dtype=torch.long)
    out = torch.full((n, MAX_LEN), PAD, dtype=torch.long)
    in_len = torch.tensor([len(c) for c in comps], dtype=torch.long)
    out_len = torch.tensor([len(r) for r in raws], dtype=torch.long)
    for i, (c, r) in enumerate(zip(comps, raws)):
        inp[i, : len(c)] = torch.frombuffer(bytearray(c), dtype=torch.uint8).long()
        out[i, : len(r)] = torch.frombuffer(bytearray(r), dtype=torch.uint8).long()
    bucket = torch.tensor([bucket_of(c) for c in comps], dtype=torch.long)
    return {
        "inp": inp,
        "in_len": in_len,
        "out": out,
        "out_len": out_len,
        "bucket": bucket,
    }


def bucket_of(comp: bytes) -> int:
    """LZMA2 首控制字节：0x01/0x02 = 原样存储块（明文就躺在流里），其余 = 压缩块。"""
    return 1 if comp[0] in (0x01, 0x02) else 0


def build(n_train, n_test, save_path, max_bytes=MAX_LEN, seeds=(11, 12)):
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    seen = set()
    tr_raw, tr_comp = build_split(n_train, seeds[0], seen, max_bytes)
    te_raw, te_comp = build_split(n_test, seeds[1], seen, max_bytes)
    meta = {
        "n_train": n_train,
        "n_test": n_test,
        "max_len": MAX_LEN,
        "max_bytes": max_bytes,
        "filters": FILTERS,
        "avg_raw": sum(map(len, tr_raw + te_raw)) / (n_train + n_test),
        "avg_comp": sum(map(len, tr_comp + te_comp)) / (n_train + n_test),
        "uncompressed_share": {
            "train": sum(map(bucket_of, tr_comp)) / n_train,
            "test": sum(map(bucket_of, te_comp)) / n_test,
        },
    }
    torch.save(
        {
            "train": to_tensors(tr_comp, tr_raw),
            "test": to_tensors(te_comp, te_raw),
            "meta": meta,
        },
        save_path,
    )
    return meta


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/dataset.pt"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    max_bytes = int(sys.argv[4]) if len(sys.argv) > 4 else MAX_LEN
    meta = build(n_train, n_test, path, max_bytes)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    probe = b"hello lzma, hello lzma, hello lzma" * 3
    assert decompress(compress(probe)) == probe
    print("roundtrip ok:", len(probe), "->", len(compress(probe)))
