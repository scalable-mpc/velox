#!/usr/bin/env python3
"""Benchmark the threshold_mldsa application on local fixtures of n parties.

For every (parties n, signatures S) point of the grid this

  1. predicts the peak memory of all parties together from the points
     already measured (see `predicted_mb`). When the prediction is at or
     above the memory budget, it does not run the point and records the
     status "skipped". The machine must never run out of memory;
  2. makes the node configuration testdata/<n>/ with the config binary when
     it is missing;
  3. runs scripts/test_threshold_mldsa.sh (key generation, the preprocessing
     of S·B attempts, and the signatures of S messages; each party runs
     under /usr/bin/time) and waits for the syncer to report the output
     phase. A memory guard adds up the resident memory of all parties each
     second. When the total passes the budget, it stops the run and records
     the status "over budget". The guard is only a safety stop: the
     prediction must keep the runs below the budget;
  4. reads the phase latencies off the syncer log, the plan off party 0's log
     (op-depths, engine depths, edaBits, random bits), and off every party's
     log the time of the signing op-depths 6 to 9, the first passing attempt
     of each message, the number of signed messages and the RustCrypto
     results; and off every party's /usr/bin/time output its peak memory;

and appends one row per run to a CSV. Repetitions are separate rows; the
report step takes medians. The points run in order of growing memory: n
first, then S. Rows already in the CSV count: a new run does only the
repetitions that are missing, so a stopped grid continues where it stopped.
Skipped rows do not count: a new run decides them again, for example with a
larger budget.

    scripts/bench_threshold_mldsa.py run    [--reps 3] [--parties 4,7,10,13,16] [--signatures 1,2,4]
                                            [--attempts 16] [--memory-limit-gb 10] [--out logs/bench_threshold_mldsa.csv]
    scripts/bench_threshold_mldsa.py report [--out logs/bench_threshold_mldsa.csv]

Needs release builds of threshold_mldsa and config.
"""

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_ops import ROOT, COMP, phase_latencies_in  # noqa: E402

FIELDS = ["parties", "signatures", "attempts", "rep", "status", "op_depths", "engine_depths", "edabits",
          "random_bits", "preprocessing_ms", "circuit_ms", "verification_ms", "total_ms", "signing_ms",
          "signed", "parties_done", "rejected_parties", "passed_attempts",
          "peak_party_mb", "median_party_mb", "peak_total_mb", "predicted_total_mb"]

MB = 1 << 20


def party_log(n, party):
    return os.path.join(ROOT, "logs", f"party-{party}-mldsa_n_{n}_{COMP}.log")


def rss_file(n, party):
    return os.path.join(ROOT, "logs", f"rss-{party}-mldsa_n_{n}_{COMP}.txt")


def syncer_log(n):
    return os.path.join(ROOT, "logs", f"syncer_mldsa_n_{n}_{COMP}.log")


def read(path):
    return open(path).read() if os.path.exists(path) else ""


def kill_all():
    subprocess.run(["killall", "-9", "threshold_mldsa"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ensure_config(n):
    """Writes testdata/<n>/ with the config binary, as in README step 5."""
    target = os.path.join(ROOT, "testdata", str(n))
    if os.path.exists(os.path.join(target, "nodes-0.json")):
        return
    os.makedirs(target, exist_ok=True)
    subprocess.run(["./target/release/config", "--base_port", "15000", "--client_base_port", "19000",
                    "--client_run_port", "19500", "--NumNodes", str(n), "--blocksize", "100", "--delay", "100",
                    "--target", f"testdata/{n}/", "--local", "true"], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)


def total_rss_mb():
    """The resident memory of all threshold_mldsa processes, in MB."""
    out = subprocess.run(["ps", "-axo", "rss=,comm="], capture_output=True, text=True).stdout
    kb = sum(int(line.split(None, 1)[0]) for line in out.splitlines()
             if line.strip().endswith("threshold_mldsa"))
    return kb / 1024


def peak_party_mb(n, party):
    """The party's peak memory from /usr/bin/time: bytes on macOS, kbytes on Linux."""
    text = read(rss_file(n, party))
    m = re.search(r"(\d+)\s+maximum resident set size", text)
    if m:
        return int(m[1]) / MB
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    return int(m[1]) / 1024 if m else None


def run_point(n, signatures, attempts, rep, timeout, grace, limit_mb):
    ensure_config(n)
    kill_all()
    time.sleep(20)  # sockets from the previous run have to close first
    subprocess.run(["bash", "scripts/test_threshold_mldsa.sh", str(n), str(COMP), str(attempts), str(signatures)],
                   cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    start, status, peak_total, phases = time.time(), "timeout", 0.0, {}
    while time.time() - start < timeout:
        time.sleep(1)
        peak_total = max(peak_total, total_rss_mb())
        if peak_total > limit_mb:
            status = "over budget"
            break
        phases = phase_latencies_in(syncer_log(n))
        if "output" in phases:
            status = "ok"
            break
        if any("panicked" in read(rss_file(n, p)) for p in range(n)):
            status = "failed"
            break
    if status == "ok":
        time.sleep(grace)  # the syncer reports at n − t parties; let the stragglers finish
    kill_all()
    time.sleep(2)  # /usr/bin/time writes its output after the party is killed

    row = {"parties": n, "signatures": signatures, "attempts": attempts, "rep": rep, "status": status,
           "peak_total_mb": round(peak_total)}
    m = re.search(r"Planner: (\d+) op-depths over (\d+) engine depths, (\d+) edaBits \((\d+) random bits\)",
                  read(party_log(n, 0)))
    if m:
        row.update(op_depths=m[1], engine_depths=m[2], edabits=m[3], random_bits=m[4])
    pre, online = phases.get("Preprocessing"), phases.get("Online")
    verification, output = phases.get("verification"), phases.get("output")
    row.update(preprocessing_ms=pre, total_ms=output,
               circuit_ms=online - pre if pre and online else None,
               verification_ms=verification - online if online and verification else None)
    signing, signed, done, rejected = [], [], 0, 0
    for party in range(n):
        text = read(party_log(n, party))
        m = re.search(r"signing op-depths 6 to 9 took (\d+) ms; first passing attempt of each message: \[(.*)\]", text)
        if m:
            signing.append(int(m[1]))
            row["passed_attempts"] = m[2]
        m = re.search(r"(\d+) of \d+ messages signed", text)
        if m:
            done += 1
            signed.append(int(m[1]))
        rejected += "RustCrypto ml-dsa rejects" in text or "accepts it: false" in text
    memory = [mb for mb in (peak_party_mb(n, p) for p in range(n)) if mb is not None]
    row.update(signing_ms=statistics.median(signing) if signing else None,
               signed=max(signed) if signed else None, parties_done=done, rejected_parties=rejected,
               peak_party_mb=round(max(memory)) if memory else None,
               median_party_mb=round(statistics.median(memory)) if memory else None)
    return row


def predicted_mb(rows, n, signatures):
    """The predicted peak memory of all parties of the point (n, S), in MB,
    or None when no point is measured yet.

    A measured point gives the median of its runs. A run that the guard
    stopped counts as infinite memory, so that point and all points that
    are predicted from it are skipped. For a new point, each of these is an
    estimate from above, and the prediction is the smallest of them:

    - from the largest smaller n with the same S: the memory times n/n'.
      The memory of one party fell from n = 4 to n = 10 in the runs of
      issue 19, so the total grows at most linearly in n;
    - from the largest smaller S with the same n: the memory times the
      growth from S' to S at a smaller n, when one is measured, or else
      times S/S'.
    """
    measured = {}
    for r in rows:
        if r["status"] in ("ok", "over budget") and r["peak_total_mb"]:
            mb = float("inf") if r["status"] == "over budget" else float(r["peak_total_mb"])
            measured.setdefault((int(r["parties"]), int(r["signatures"])), []).append(mb)
    measured = {point: statistics.median(mbs) for point, mbs in measured.items()}
    if (n, signatures) in measured:
        return measured[(n, signatures)]
    estimates = []
    smaller_n = [m for m, s in measured if s == signatures and m < n]
    if smaller_n:
        m = max(smaller_n)
        estimates.append(measured[(m, signatures)] * n / m)
    smaller_s = [s for m, s in measured if m == n and s < signatures]
    if smaller_s:
        s = max(smaller_s)
        growth = signatures / s
        both = [m for m, t in measured if t == signatures and m < n and (m, s) in measured]
        if both:
            m = max(both)
            growth = measured[(m, signatures)] / measured[(m, s)]
        estimates.append(measured[(n, s)] * growth)
    return min(estimates) if estimates else None


def run(args):
    out = os.path.join(ROOT, args.out)
    rows = list(csv.DictReader(open(out))) if os.path.exists(out) else []
    rows = [r for r in rows if r["status"] != "skipped"]
    limit_mb = args.memory_limit_gb * 1024
    with open(out, "w", newline="") as f:
        # Write the kept rows again, so that the header has the columns of FIELDS.
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
        for n in [int(p) for p in args.parties.split(",")]:
            for signatures in [int(s) for s in args.signatures.split(",")]:
                done = sum(int(r["parties"]) == n and int(r["signatures"]) == signatures
                           and int(r["attempts"]) == args.attempts for r in rows)
                for rep in range(done, args.reps):
                    predicted = predicted_mb(rows, n, signatures)
                    if predicted is not None and predicted >= limit_mb:
                        row = {"parties": n, "signatures": signatures, "attempts": args.attempts, "rep": rep,
                               "status": "skipped"}
                    else:
                        row = run_point(n, signatures, args.attempts, rep, args.timeout, args.grace, limit_mb)
                    row["predicted_total_mb"] = round(predicted) if predicted not in (None, float("inf")) else None
                    print(row, flush=True)
                    w.writerow(row)
                    f.flush()
                    rows.append({k: "" if row.get(k) is None else str(row[k]) for k in FIELDS})


def median(group, column):
    values = [float(r[column]) for r in group if r.get(column)]
    return f"{statistics.median(values):.0f}" if values else "–"


def report(args):
    rows = list(csv.DictReader(open(os.path.join(ROOT, args.out))))
    columns = ["preprocessing_ms", "circuit_ms", "signing_ms", "verification_ms", "total_ms",
               "peak_party_mb", "peak_total_mb", "predicted_total_mb"]
    print("| n | S | runs | status | random bits | edaBits | " + " | ".join(columns) + " | signed |")
    print("|---" * (len(columns) + 7) + "|")
    for n, s in sorted({(int(r["parties"]), int(r["signatures"])) for r in rows}):
        group = [r for r in rows if int(r["parties"]) == n and int(r["signatures"]) == s]
        ok = [r for r in group if r["status"] == "ok"]
        statuses = ", ".join(f"{sum(r['status'] == x for r in group)} {x}"
                             for x in sorted({r["status"] for r in group}))
        signed = sum(int(r["signed"]) for r in ok if r["signed"])
        first = group[0]
        print(f"| {n} | {s} | {len(group)} | {statuses} | {first['random_bits'] or '–'} | {first['edabits'] or '–'} | "
              + " | ".join(median(ok or group, c) for c in columns) + f" | {signed}/{s * len(ok)} |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--parties", default="4,7,10,13,16")
    r.add_argument("--signatures", default="1,2,4")
    r.add_argument("--attempts", type=int, default=16)
    r.add_argument("--memory-limit-gb", type=float, default=10)
    r.add_argument("--timeout", type=int, default=1800)
    r.add_argument("--grace", type=int, default=5)
    r.add_argument("--out", default="logs/bench_threshold_mldsa.csv")
    q = sub.add_parser("report")
    q.add_argument("--out", default="logs/bench_threshold_mldsa.csv")
    args = p.parse_args()
    run(args) if args.command == "run" else report(args)


if __name__ == "__main__":
    main()
