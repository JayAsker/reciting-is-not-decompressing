"""训练/评估：LZMA 定点映射的神经解压器，附哈希查表 baseline 与耗时对比。

停止规则：训练集文件级完全匹配率首次达到 100% 的那一刻停手（不偷看测试集）。
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.nn.functional as F

import data
from data import MAX_LEN, PAD
from model import ByteTransformer, n_params


def bucketed_batches(lengths, batch_size, rng=None, shuffle=False):
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(batches)
        for b in batches:
            rng.shuffle(b)
    return batches


def split_tensors(split, device):
    return (
        split["inp"].to(device),
        split["in_len"].to(device),
        split["out"].to(device),
        split["out_len"].to(device),
    )


@torch.no_grad()
def evaluate(model, split, device, batch_size=16):
    model.eval()
    inp, in_len, out, out_len = split_tensors(split, device)
    n = inp.shape[0]
    file_match = 0
    correct = 0
    total = 0
    len_err = 0.0
    len_exact = 0
    buck = split.get("bucket")
    buck_n = {0: 0, 1: 0}
    buck_hit = {0: 0, 1: 0}
    for idx in bucketed_batches(in_len.tolist(), batch_size):
        idx = torch.tensor(idx, device=device)
        logp, length = model(inp[idx], in_len[idx])
        pred = logp.argmax(-1)
        y = out[idx]
        mask = torch.arange(MAX_LEN, device=device)[None] < out_len[idx][:, None]
        correct += int(((pred == y) & mask).sum())
        total += int(mask.sum())
        ok = ((pred == y) | ~mask).all(dim=1)
        file_match += int(ok.sum())
        len_err += float((length - out_len[idx].float()).abs().sum())
        len_exact += int((length.round().long() == out_len[idx]).sum())
        if buck is not None:
            b = buck.to(device)[idx]
            for k in (0, 1):
                sel = b == k
                buck_n[k] += int(sel.sum())
                buck_hit[k] += int(ok[sel].sum())
    res = {
        "n": n,
        "file_match": file_match / n,
        "byte_acc": correct / max(total, 1),
        "len_mae": len_err / n,
        "len_exact": len_exact / n,
    }
    if buck is not None:
        res["by_bucket"] = {
            "compressed": {"n": buck_n[0], "file_match": buck_hit[0] / max(buck_n[0], 1)},
            "uncompressed": {"n": buck_n[1], "file_match": buck_hit[1] / max(buck_n[1], 1)},
        }
    return res


def hash_baseline(train, test):
    table = {}
    inp, out, out_len = train["inp"], train["out"], train["out_len"]
    for i in range(inp.shape[0]):
        table[bytes(inp[i, : int(train["in_len"][i])].tolist())] = bytes(
            out[i, : int(out_len[i])].tolist()
        )
    out_c = {}
    for name, split in (("train", train), ("test", test)):
        hit = 0
        right = 0
        for i in range(split["inp"].shape[0]):
            key = bytes(split["inp"][i, : int(split["in_len"][i])].tolist())
            truth = bytes(split["out"][i, : int(split["out_len"][i])].tolist())
            got = table.get(key)
            if got is not None:
                hit += 1
                right += int(got == truth)
        out_c[name] = {"table_hit": hit, "file_match": right / split["inp"].shape[0]}
    out_c["table_size"] = len(table)
    return out_c


def time_model(model, split, device):
    model.eval()
    inp, in_len, _, _ = split_tensors(split, device)
    n = inp.shape[0]
    for i in range(min(8, n)):
        model(inp[i : i + 1, :MAX_LEN], in_len[i : i + 1])
    if device.type == "cuda":
        torch.cuda.synchronize()
    single = []
    for i in range(n):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(inp[i : i + 1, :MAX_LEN], in_len[i : i + 1])
        if device.type == "cuda":
            torch.cuda.synchronize()
        single.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    bs = 16
    done = 0
    for idx in bucketed_batches(in_len.tolist(), bs):
        idx = torch.tensor(idx, device=device)
        lb = int(in_len[idx].max())
        model(inp[idx, :lb], in_len[idx])
        done += len(idx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    bulk = (time.perf_counter() - t0) / done
    return {"per_sample_bs1": statistics.median(single), "per_sample_bs16_bucketed": bulk}


def time_lzma(split):
    streams = [
        bytes(split["inp"][i, : int(split["in_len"][i])].tolist())
        for i in range(split["inp"].shape[0])
    ]
    ts = []
    for s in streams:
        t0 = time.perf_counter()
        data.decompress(s)
        ts.append(time.perf_counter() - t0)
    return {"per_sample": statistics.median(ts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.pt")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--len-loss", type=float, default=0.1)
    ap.add_argument("--copy", action="store_true", help="启用 pointer-generator 拷贝头")
    ap.add_argument(
        "--copy-rel-only", action="store_true", help="拷贝分数只由相对位置偏置决定（纯偏移指针）"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.data, weights_only=False)
    train, test, meta = blob["train"], blob["test"], blob["meta"]
    os.makedirs(args.out, exist_ok=True)

    model = ByteTransformer(
        args.d_model, args.layers, args.heads, copy=args.copy, copy_rel_only=args.copy_rel_only
    ).to(device)
    print(
        "device=%s params=%d train=%d test=%d avg_raw=%.0f avg_comp=%.0f"
        % (
            device,
            n_params(model),
            meta["n_train"],
            meta["n_test"],
            meta["avg_raw"],
            meta["avg_comp"],
        ),
        flush=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * ((meta["n_train"] + args.batch_size - 1) // args.batch_size),
        pct_start=0.05, anneal_strategy="cos",
    )

    inp, in_len, out, out_len = split_tensors(train, device)
    rng = __import__("random").Random(args.seed)
    batches = bucketed_batches(in_len.tolist(), args.batch_size, rng, shuffle=True)

    history = []
    full_epoch = None
    t_start = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        byte_sum = 0
        tok_sum = 0
        for idx in batches:
            idx = torch.tensor(idx, device=device)
            lb = int(max(in_len[idx].max(), out_len[idx].max()))
            x, y = inp[idx, :lb], out[idx, :lb]
            mask = torch.arange(lb, device=device)[None] < out_len[idx][:, None]
            logp, length = model(x, in_len[idx])
            nll = -logp.gather(2, y.clamp(max=255).unsqueeze(-1)).squeeze(-1)
            ce = (nll * mask).sum() / mask.sum()
            le = F.mse_loss(length / MAX_LEN, out_len[idx].to(length.dtype) / MAX_LEN)
            loss = ce + args.len_loss * le
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += loss.item()
            with torch.no_grad():
                byte_sum += int(((logp.argmax(-1) == y) & mask).sum())
                tok_sum += int(mask.sum())

        if ep % args.eval_every == 0 or ep == args.epochs or full_epoch is not None:
            tr = evaluate(model, train, device)
            te = evaluate(model, test, device)
            br = te.get("by_bucket", {})
            row = {
                "epoch": ep,
                "loss": loss_sum / len(batches),
                "train_byte_acc": tr["byte_acc"],
                "train_file_match": tr["file_match"],
                "test_byte_acc": te["byte_acc"],
                "test_file_match": te["file_match"],
                "test_uncompressed_match": br.get("uncompressed", {}).get("file_match"),
                "test_compressed_match": br.get("compressed", {}).get("file_match"),
                "sec": round(time.perf_counter() - t_start, 1),
            }
            history.append(row)
            print(
                "ep %4d loss %.4f  train match %.4f byte %.4f | test match %.4f byte %.4f "
                "[uncomp %.4f comp %.4f]  (%.0fs)"
                % (
                    ep,
                    row["loss"],
                    row["train_file_match"],
                    row["train_byte_acc"],
                    row["test_file_match"],
                    row["test_byte_acc"],
                    row["test_uncompressed_match"],
                    row["test_compressed_match"],
                    row["sec"],
                ),
                flush=True,
            )
            if full_epoch is None and tr["file_match"] == 1.0:
                full_epoch = ep
                torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
                print("train file-match hit 1.0 at epoch %d, stop." % ep, flush=True)
                break

    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    final = {
        "train": evaluate(model, train, device),
        "test": evaluate(model, test, device),
    }
    result = {
        "meta": meta,
        "args": vars(args),
        "params": n_params(model),
        "device": str(device),
        "stop_epoch": full_epoch,
        "train_seconds": round(time.perf_counter() - t_start, 1),
        "final": final,
        "hash_baseline": hash_baseline(train, test),
        "time_model": time_model(model, test, device),
        "time_lzma": time_lzma(test),
        "history": history,
    }
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n=== final ===")
    for name in ("train", "test"):
        f_ = final[name]
        print(
            "%s: file_match=%.4f (%d/%d)  byte_acc=%.4f  len_mae=%.1fB  len_exact=%.4f"
            % (
                name,
                f_["file_match"],
                round(f_["file_match"] * f_["n"]),
                f_["n"],
                f_["byte_acc"],
                f_["len_mae"],
                f_["len_exact"],
            )
        )
        for k, v in f_.get("by_bucket", {}).items():
            print(
                "   %-12s n=%3d  file_match=%.4f"
                % (k, v["n"], v["file_match"])
            )
    hb = result["hash_baseline"]
    print(
        "hash table: train=%.4f test=%.4f size=%d"
        % (hb["train"]["file_match"], hb["test"]["file_match"], hb["table_size"])
    )
    tm, tl = result["time_model"], result["time_lzma"]
    print(
        "nn: %.2f ms/sample (bs1, 1024 slots) %.2f ms (bs16 bucketed) | lzma: %.1f us/sample | ratio %.0fx"
        % (
            tm["per_sample_bs1"] * 1e3,
            tm["per_sample_bs16_bucketed"] * 1e3,
            tl["per_sample"] * 1e6,
            tm["per_sample_bs1"] / tl["per_sample"],
        )
    )


if __name__ == "__main__":
    main()
