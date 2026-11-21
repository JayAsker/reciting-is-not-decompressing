# Reciting is not decompressing

Code, evidence and figures for the satirical note

> *Reciting Is Not Decompressing: A Case Study in Copy Heads, Lookup Tables,
> and the Temptation to Report a Rounding Error as a Result*

**DOI (note):** https://doi.org/10.5281/zenodo.24423042

## What this is, and what it is not

This repository is the **artifact for a satire**. It accompanies a short piece
that walks through one deliberately small experiment — a byte-level transformer
asked to invert LZMA2 — and then declines to describe the outcome as a result.

Read the note before you read the numbers below, or the numbers will look like a
claim. They are not. The joke is in the framing; **the measurements are real**,
which is the only reason they are worth shipping. Every figure in the note is
regenerated from `evidence/*/results.json` by `make_figs.py`; nothing is
hand-typed except a single hash-table lookup constant, flagged in the source.

**Do not cite this as prior work on neural LZMA2 decompression.** There is no
prior work here. There is a negative result, a learned copy head, and a lookup
table that beats both.

The registered record is `upload_type: other` precisely so that no indexer
files it under `publication`. If you found this repository through a citation
manager, something has gone wrong and you should tell the author.

## The experiment in one paragraph

Compress short plaintexts with `lzma.FILTER_LZMA2` at preset 6, keep only pairs
where both the plaintext and the compressed stream fit in 1024 bytes, and train a
causal byte transformer to map stream to plaintext. Because LZMA2 emits **stored**
chunks when compression does not help, a fraction of the "compressed" inputs
contain their own answer verbatim. A hash table keyed on the input recovers
exactly those samples and nothing else. The transformer eventually learns to do
the same, at roughly three orders of magnitude the cost per sample. On samples
whose answer is not sitting in the input, it reaches 0% whole-sample exact match.

## Layout

```
data.py                 corpus builder: synthetic plaintext -> LZMA2 pairs -> .pt tensors
model.py                ByteTransformer (byte embeddings, causal blocks, length head)
train.py                training loop, bucketed batching, hash baseline, timing probes
make_figs.py            regenerates figs/curve.* and figs/cost.* from evidence
evidence/<run>/results.json   archived metrics, one file per run
figs/bytes.png          pre-rendered byte strip (needs a checkpoint; shipped as-is)
figs/curve.*, cost.*    produced by make_figs.py
```

`figs/bytes.png` is the one artifact not reproduced from JSON: it is a picture of
model output, so it requires weights, and no weights are shipped here. It is
included so the note compiles.

## Requirements

Python 3.10+, `torch` (CPU is enough to smoke-test), `matplotlib`. No other
dependencies. The corpus is generated from a seeded RNG inside `data.py`, so
there is nothing to download and no license to negotiate.

```bash
pip install torch matplotlib
```

## Smoke test (about 20 seconds on CPU, or ~17 s on a small GPU)

```bash
python data.py smoke.pt 8 8 256
python train.py --data smoke.pt --out smoke_run --epochs 1 --eval-every 1 \
    --batch-size 4 --d-model 32 --layers 1 --heads 2
```

Expected shape of the output (exact numbers vary by machine and seed):

```
device=cuda params=63297 train=8 test=8 avg_raw=163 avg_comp=126
ep    1 loss 5.7026  train match 0.0000 byte 0.0034 | test match 0.0000 byte 0.0043 ...
=== final ===
train: file_match=0.0000 (0/8)  byte_acc=0.0034  len_mae=327.7B  len_exact=0.0000
hash table: train=1.0000 test=0.0000 size=8
nn: 1.87 ms/sample (bs1, 1024 slots) 0.46 ms (bs16 bucketed) | lzma: 34.3 us/sample | ratio 54x
```

One epoch on eight samples is a plumbing check, not a result. Note that it
already shows the whole phenomenon: the hash table scores 1.0 on train and 0.0 on
test, and the network is far slower than the codec.

## Reproducing the four archived runs

The corpus is small and the seeds are fixed. Run from this directory.

```bash
# long-form corpus: 64 train / 512 test, plaintext up to 1024 bytes
python data.py data/dataset.pt 64 512

# short corpus: 512 / 512, plaintext up to 256 bytes (the 1024-slot tensor is unchanged)
python data.py data/short.pt 512 512 256
```

Then, matching `evidence/<run>/results.json` one for one:

```bash
python train.py --data data/dataset.pt --out runs/main \
    --epochs 600 --eval-every 20 --batch-size 8 --lr 5e-4 \
    --d-model 256 --layers 3 --heads 8

python train.py --data data/short.pt --out runs/nocopy_short \
    --epochs 100 --eval-every 20 --batch-size 8 --lr 1e-3 \
    --d-model 192 --layers 2 --heads 8

python train.py --data data/short.pt --out runs/copy_short \
    --epochs 100 --eval-every 10 --batch-size 8 --lr 1e-3 \
    --d-model 192 --layers 2 --heads 8 --copy

python train.py --data data/short.pt --out runs/copy_rel \
    --epochs 100 --eval-every 10 --batch-size 8 --lr 1e-3 \
    --d-model 192 --layers 2 --heads 8 --copy --copy-rel-only
```

Training stops early the moment train-split whole-sample exact match reaches
100% (the test split is never consulted for stopping). `copy_rel` never reaches
it, which is why it is the only run that stops on the epoch budget.

| run | corpus | params | stopped | train EM | test EM | test byte acc |
|---|---|---|---|---|---|---|
| `main` | `dataset.pt` 64/512 | 2,829,569 | ep 120 | 1.0000 | 0.0000 | 0.03844 |
| `nocopy_short` | `short.pt` 512/512 | 1,222,721 | ep 80 | 1.0000 | 0.0000 | 0.06147 |
| `copy_short` | `short.pt` 512/512 | 1,297,091 | ep 70 | 1.0000 | 0.0000 | 0.06520 |
| `copy_rel` | `short.pt` 512/512 | 1,222,979 | no early stop | 0.98242 | 0.15430 | 0.19053 |

The single non-zero test number in that table is `copy_rel` at 15.43%, and it is
entirely accounted for by samples whose plaintext was stored verbatim in the
input stream: 79 of 94 stored test samples match, 0 of 418 compressed ones. The
stored share of the test split is 0.18359. Multiply and you get the reported
figure without a model.

## Timing

`results.json` carries two probe types per run: `time_model/per_sample_bs1`,
`time_model/per_sample_bs16_bucketed`, and `time_lzma/per_sample`. Wall-clock
numbers are machine-dependent — the archived ones were taken on one CUDA device
and are quoted in the note as ratios, not as absolute claims.

## License

MIT, see `LICENSE`. The text and figures of the note itself are CC BY 4.0 under
the Zenodo record.
