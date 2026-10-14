"""Benchmark through the Wattson gateway at several concurrency levels.

For each concurrency level C, sends `waves` rounds of C simultaneous
requests, then reads Wattson's per-request records and writes a CSV plus a
summary table.

    python scripts/load_test.py --label hf-serial --concurrency 1,2,4 --max-tokens 128

Standard library only.
"""

import argparse
import csv
import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.request

PROMPT = "Explain in detail how a GPU executes a matrix multiplication."


def get(url):
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


def send_request(gateway, model, max_tokens, ids, errors):
    body = json.dumps({
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "ignore_eos": True,  # fixed output length makes runs comparable
        "messages": [{"role": "user", "content": PROMPT}],
    }).encode()
    req = urllib.request.Request(f"{gateway}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            ids.append(int(resp.headers["x-wattson-request-id"]))
            for _ in resp:  # read the whole stream
                pass
    except urllib.error.HTTPError as err:  # e.g. wrong model name
        errors.append(f"HTTP {err.code}: {err.read()[:200].decode(errors='replace')}")
    except Exception as err:  # keep going; report at the end
        errors.append(str(err))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default="http://127.0.0.1:8080")
    parser.add_argument("--label", required=True, help="name for this run, e.g. hf-serial")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="model name; vLLM rejects requests for any other name")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--waves", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    energy = get(f"{args.gateway}/energy")
    print(f"energy source: {energy['source']}")
    print("measuring idle power for 5 s ...")
    e0 = get(f"{args.gateway}/energy")["total_energy_j"]
    time.sleep(5)
    idle_w = (get(f"{args.gateway}/energy")["total_energy_j"] - e0) / 5
    print(f"idle power: {idle_w:.1f} W\n")

    level_of = {}  # request id -> concurrency level
    wall = {}
    for level in [int(c) for c in args.concurrency.split(",")]:
        start = time.time()
        for wave in range(args.waves):
            ids, errors = [], []
            threads = [threading.Thread(target=send_request,
                                        args=(args.gateway, args.model, args.max_tokens, ids, errors))
                       for _ in range(level)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            for i in ids:
                level_of[i] = level
            if errors:
                print(f"  concurrency {level} wave {wave + 1}: {len(errors)} errors, e.g. {errors[0]}")
        wall[level] = time.time() - start
        print(f"concurrency {level}: done in {wall[level]:.1f} s")

    time.sleep(0.5)  # let the gateway record the last requests
    records = [r for r in get(f"{args.gateway}/requests") if r["id"] in level_of]

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.label}.csv")
    fields = ["label", "concurrency", "id", "completion_tokens", "ttft_ms", "total_ms",
              "decode_tokens_per_s", "energy_j", "naive_energy_j", "joules_per_token", "completed"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in sorted(records, key=lambda r: r["id"]):
            writer.writerow({"label": args.label, "concurrency": level_of[r["id"]],
                             **{k: r.get(k) for k in fields[2:]}})

    print(f"\nsource: {energy['source']}   idle: {idle_w:.1f} W   saved: {path}\n")
    print(f"{'conc':>4} {'reqs':>4} {'J/token':>8} {'J/req':>7} {'TTFT ms':>8} "
          f"{'tok/s/req':>9} {'total tok/s':>11}")
    for level in sorted(wall):
        rs = [r for r in records if level_of[r["id"]] == level and r["completion_tokens"]]
        if not rs:
            continue
        tokens = sum(r["completion_tokens"] for r in rs)
        print(f"{level:>4} {len(rs):>4} "
              f"{sum(r['energy_j'] for r in rs) / tokens:>8.3f} "
              f"{statistics.mean(r['energy_j'] for r in rs):>7.1f} "
              f"{statistics.mean(r['ttft_ms'] or 0 for r in rs):>8.0f} "
              f"{statistics.mean(r['decode_tokens_per_s'] or 0 for r in rs):>9.1f} "
              f"{tokens / wall[level]:>11.1f}")


if __name__ == "__main__":
    main()
