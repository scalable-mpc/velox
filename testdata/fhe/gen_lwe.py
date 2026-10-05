#!/usr/bin/env python3
"""Write an LWE key and ciphertexts for the fhe_decrypt application.

    python3 testdata/fhe/gen_lwe.py [--num N] [--seed S] [--out DIR]

Writes three files into DIR (default testdata/fhe):

  key.txt          the n = 1024 key bits, one per line; party 0 deals them.
  ciphertexts.txt  one ciphertext per line, "a_1 ... a_n b" in decimal.
  plaintexts.txt   the bit each ciphertext encrypts, one per line.

A ciphertext encrypts a bit mu as b = <a, s> + mu * 2^63 + e (mod 2^64), a
uniform, e a rounded Gaussian of standard deviation 2^40. The first six
ciphertexts put the noise at the ends of the decryptable range [-2^62, 2^62)
and at zero, for both bits; the remaining N - 6 are random.
"""

import argparse
import os
import random

N_LWE = 1024
Q = 1 << 64
SIGMA = 2.0 ** 40
BOUNDARY_NOISE = [-(1 << 62), 0, (1 << 62) - 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num", type=int, default=64, help="number of ciphertexts (at least 6)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()
    if args.num < len(BOUNDARY_NOISE) * 2:
        parser.error(f"--num must be at least {len(BOUNDARY_NOISE) * 2}")

    rng = random.Random(args.seed)
    key = [rng.randrange(2) for _ in range(N_LWE)]
    cases = [(mu, e) for mu in (0, 1) for e in BOUNDARY_NOISE]
    cases += [(rng.randrange(2), round(rng.gauss(0, SIGMA))) for _ in range(args.num - len(cases))]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "key.txt"), "w") as f:
        f.writelines(f"{s}\n" for s in key)
    with open(os.path.join(args.out, "ciphertexts.txt"), "w") as cts, \
         open(os.path.join(args.out, "plaintexts.txt"), "w") as pts:
        for mu, e in cases:
            a = [rng.randrange(Q) for _ in range(N_LWE)]
            b = (sum(ai * si for ai, si in zip(a, key)) + mu * (1 << 63) + e) % Q
            cts.write(" ".join(map(str, a + [b])) + "\n")
            pts.write(f"{mu}\n")
    print(f"wrote {len(cases)} ciphertexts and a {N_LWE}-bit key to {args.out}")


if __name__ == "__main__":
    main()
