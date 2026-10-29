#!/usr/bin/env python3
"""Benchmark the threshold_mldsa application on the local 10-party fixture.

For every number of signing attempts B this

  1. runs scripts/test_threshold_mldsa.sh (key generation, the preprocessing
     of B attempts, and the signature of one message) and waits for the
     syncer to report the output phase;
  2. reads the phase latencies off the syncer log, the plan off party 0's log
     (op-depths, engine depths, edaBits, random bits), and off every party's
     log the time of the signing op-depths 6 to 9, the first passing attempt
     and whether RustCrypto ml-dsa accepts the signature;

and appends one row per run to a CSV. Repetitions are separate rows; the
report step takes medians.

    scripts/bench_threshold_mldsa.py run    [--reps 3] [--attempts 1,4,16] [--out logs/bench_threshold_mldsa.csv]
    scripts/bench_threshold_mldsa.py report [--out logs/bench_threshold_mldsa.csv]

Needs a release build of threshold_mldsa and the testdata/10 fixture.
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
from bench_ops import ROOT, N, COMP, phase_latencies_in  # noqa: E402

FIELDS = ["attempts", "rep", "op_depths", "engine_depths", "edabits", "random_bits",
          "preprocessing_ms", "circuit_ms", "verification_ms", "total_ms",
          "signing_ms", "passed_attempt", "accepted_parties", "no_signature_parties"]


def party_log(party):
    return os.path.join(ROOT, "logs", f"party-{party}-mldsa_n_{N}_{COMP}.log")


def syncer_log():
    return os.path.join(ROOT, "logs", f"syncer_mldsa_n_{N}_{COMP}.log")


def kill_all():
    subprocess.run(["killall", "-9", "threshold_mldsa"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_point(attempts, rep, timeout, grace):
    kill_all()
    time.sleep(20)  # sockets from the previous run have to close first
    subprocess.run(["bash", "scripts/test_threshold_mldsa.sh", str(N), str(COMP), str(attempts)], cwd=ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    start = time.time()
    phases = {}
    while time.time() - start < timeout:
        time.sleep(3)
        phases = phase_latencies_in(syncer_log())
        if "output" in phases:
            break
    time.sleep(grace)  # the syncer reports at n − t parties; let the stragglers finish
    kill_all()

    row = {"attempts": attempts, "rep": rep}
    p0 = open(party_log(0)).read() if os.path.exists(party_log(0)) else ""
    m = re.search(r"Planner: (\d+) op-depths over (\d+) engine depths, (\d+) edaBits \((\d+) random bits\)", p0)
    if m:
        row.update(op_depths=m[1], engine_depths=m[2], edabits=m[3], random_bits=m[4])
    pre, online = phases.get("Preprocessing"), phases.get("Online")
    verification, output = phases.get("verification"), phases.get("output")
    row.update(preprocessing_ms=pre, total_ms=output,
               circuit_ms=online - pre if pre and online else None,
               verification_ms=verification - online if online and verification else None)
    signing, accepted, no_signature = [], 0, 0
    for party in range(N):
        text = open(party_log(party)).read() if os.path.exists(party_log(party)) else ""
        m = re.search(r"signing op-depths 6 to 9 took (\d+) ms; first passing attempt of each message: \[(.*)\]", text)
        if m:
            signing.append(int(m[1]))
            row["passed_attempt"] = m[2]
        accepted += "RustCrypto ml-dsa accepts it: true" in text
        no_signature += "no signature" in text
    row.update(signing_ms=statistics.median(signing) if signing else None,
               accepted_parties=accepted, no_signature_parties=no_signature)
    return row


def run(args):
    out = os.path.join(ROOT, args.out)
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for attempts in [int(a) for a in args.attempts.split(",")]:
            for rep in range(args.reps):
                row = run_point(attempts, rep, args.timeout, args.grace)
                print(row, flush=True)
                w.writerow(row)
                f.flush()


def report(args):
    rows = list(csv.DictReader(open(os.path.join(ROOT, args.out))))
    columns = ["preprocessing_ms", "circuit_ms", "signing_ms", "verification_ms", "total_ms"]
    print("| B | runs | random bits | edaBits | engine depths | " + " | ".join(columns) + " | signed |")
    print("|---" * (len(columns) + 6) + "|")
    for attempts in sorted({int(r["attempts"]) for r in rows}):
        group = [r for r in rows if int(r["attempts"]) == attempts]
        medians = []
        for c in columns:
            values = [float(r[c]) for r in group if r[c]]
            medians.append(f"{statistics.median(values):.0f}" if values else "–")
        signed = sum(int(r["accepted_parties"]) == N for r in group)
        first = group[0]
        print(f"| {attempts} | {len(group)} | {first['random_bits']} | {first['edabits']} | {first['engine_depths']} | "
              + " | ".join(medians) + f" | {signed}/{len(group)} |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--attempts", default="1,4,16")
    r.add_argument("--timeout", type=int, default=600)
    r.add_argument("--grace", type=int, default=5)
    r.add_argument("--out", default="logs/bench_threshold_mldsa.csv")
    q = sub.add_parser("report")
    q.add_argument("--out", default="logs/bench_threshold_mldsa.csv")
    args = p.parse_args()
    run(args) if args.command == "run" else report(args)


if __name__ == "__main__":
    main()
