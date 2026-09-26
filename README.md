# Scalable Anonymous Broadcast from  Asynchronous MPC using Velox

<img src="images/velox_logo.png" width="400"/>

This repository implements anonymous broadcast using Velox, an asynchronous MPC protocol accepted for publication in ACM CCS 2025. The paper is available [here](https://eprint.iacr.org/2025/1630). 
If you utilize this repository, please consider citing our work. 
```
@misc{bandarupalli2025velox,
      author = {Akhil Bandarupalli and Xiaoyu Ji and Aniket Kate and Chen-Da Liu-Zhang and Daniel Pollmann and Yifan Song},
      title = {Velox: Scalable Fair Asynchronous MPC from Lightweight Cryptography},
      howpublished = {ACM CCS 2025},
      year = {2025},
      note = {\url{https://eprint.iacr.org/2025/1630}},
      url = {https://eprint.iacr.org/2025/1630}
}
```

This code has been written as a research prototype and has not been vetted for security. Therefore, this repository can contain serious security vulnerabilities. Please use at your own risk. 

# Quick Start
We describe the steps to run this artifact. 

## Hardware and OS setup
1. This artifact has been run and tested on `x86_64` and `x64` architectures. However, we are unaware of any issues that would prevent this artifact from running on `x86` architectures. 

2. This artifact has been run and tested on Ubuntu OS (versions 20,22,24) following the Debian distro. However, we are unaware of any issues that would prevent this artifact from running on Fedora distros like CentOS and Red Hat Linux. 

## Rust installation and Cargo setup
The repository uses the `Cargo` build tool. The compatibility between dependencies has been tested for Rust version `1.83.0`.

3. **Install Rust and Cargo**: Run the set of following commands to install the toolchain required to compile code written in Rust and create binary executable files. 
```bash
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get -y autoremove
sudo apt-get -y install build-essential
sudo apt-get -y install cmake
sudo apt-get -y install curl
# Install rust (non-interactive)
curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
rustup install 1.83.0
rustup override set 1.83.0
```
4. Build the repository using the following command. The command should be run in the directory containing the `Cargo.toml` file. 
```bash
cargo build --release
mkdir logs
```
If the build fails because of lack of `lgmp` files, install the `libgmp3-dev` dependency using the following command and try again.
```
sudo apt-get install libgmp3-dev
```

5. **Generate Configuration Files**: Next, generate configuration files for nodes in the system using the following command. Run the following commands to create configuration files and necessary directories for logs storage. 
```bash
mkdir testdata/hyb_4
mkdir logs/
./target/release/config --base_port 15000 --client_base_port 19000 --client_run_port 19500 --NumNodes 4 --blocksize 100 --delay 100 --target testdata/hyb_4/ --local true
```

## Running the code
6. **Generate Inputs**: Generate files containing the inputs of each party. These files need to be placed in `testdata/inputs/` directory. A sample code in `python` has been provided to automatically generate these inputs. Navigate to the `testdata/inputs/` directory and run the following command. 
```bash
cd testdata/inputs/
python3 inp_gen.py
```
This command generates input text files of the form `input_{$i}.txt` in the `testdata/inputs/` folder. 

7. **Run the protocol**: After generating the configuration files, run the script `test.sh` in the scripts folder.
The protocol takes the following command line arguments.
- num_parties: The number of parties $n$ participating in the protocol. 
- num_messages: The anonymity set size `k`, which corresponds to the number of inputs to mix.  
- batchsize: ACSS parameter deciding number of secrets to be batched within each ACSS instance. 
- compression_factor: The degree of the polynomial in the multiplication tuple verification phase. A higher degree implies lower round complexity but higher computation complexity. 
```bash
./scripts/test.sh {num_parties} {num_messages} {batchsize} {compression_factor}
```
Substitute `{num_parties}` with the number of parties and `{num_messages}` with the `k` value, where `k` is the number of messages.  
Example values include `k=256,512,1024...`. 
An example run can be the following. 
```bash
./scripts/test.sh 4 256 1000 10
```
This script starts `n=4` parties. 
Each party $i$ reads the first `k/n` inputs from its input file `testdata/inputs/inputs_{$i}.txt`. 
Then, parties start the mixing protocol with `k` inputs. 

**Note: Each line in the input file must be less than 31 bytes. This is because the protocol converts the input into a finite field element. The code currently operates on a 254-bit finite field, so if the input is bigger, the encoding will fail.**

8. **Check results in logs**: The termination latencies of each protocol phase are logged into the `syncer-{}.log` file in logs directory. 
Please wait for a minute before checking the logfile.  
The output of individual parties can be found in individual log files `party-0-{}.log,...`. 
The `syncer-{}.log` file will contain phase-wise latencies of the protocol. 
As mentioned in the paper, the protocol contains four phases: (a) Preprocessing, (b) Online, (c) Verification, and (d) Output. 
The `syncer` module records the latency (in milliseconds) of each phase and will print it out to the log file in the following format. 
```
INFO [node::syncer] All n nodes completed the protocol for ID: 1 with latency [2961, 3241, 3457], status {"Preprocessing"}, and value {[]}
```
The array of latencies indicate the time at which each party terminated the protocol. 
In the output phase, the `syncer-{}.log` file will also contain the output of the protocol - a set of shuffled messages input to the protocol. 

9. **Kill processes**: Before running the protocol with another configuration, kill all processes running on the requested ports. 
```bash
sudo lsof -ti:15000-19500 | xargs kill -9
```

# Comparison with other works
Velox is an asynchronous MPC protocol with Fairness that can tolerate $t<\frac{n}{3}$ faulty parties. 
The protocol will terminate when no party behaves maliciously. 
On the other hand, the adversary may abort the protocol - i.e. it can prevent honest parties from getting the output, but will not learn anything about the output of the computation. 
Velox makes progress at network speed and can be deployed in unreliable and unpredictable networks like the internet. 
In contrast, protocols like MP-SPDZ and Turbopack require the network to be synchronous, with a time bound on message delivery, for retaining safety. 

Velox also does not rely on a trusted setup to run preprocessing. 
The protocol starts with a preprocessing phase where it prepares the necessary sharings. 
In contrast, prior asynchronous MPC protocols like HoneyBadgerMPC either have a non-robust synchronous preprocessing phase or rely on a trusted dealer to produce Beaver triples. 
A recent work DumboMPC (USENIX Security'25) improved HoneyBadgerMPC's preprocessing phase.
But their protocol's non-robust preprocessing phase requires $36\times$ the latency of Velox, just to prepare Beaver triples. 
Therefore, Velox is most suited for real-time performance at network speed, in wide-area networks like the internet. 

# Repository Structure

This repository implements the Velox asynchronous MPC engine and the applications built on it. Here's a high-level overview of the directory structure:

```
velox/
├── fields/                   # Finite-field arithmetic (M31, M61 and extensions, prime fields), polynomials, GEMM (SIMD/CUDA)
├── secret_sharing/           # Secret-sharing building blocks
│   ├── acss_ab/              # Asynchronous Complete Secret Sharing with Abort
│   │   └── src/protocol/     #   dealing (init, avss), CTRBC, AVID dispersal, RA, per-instance state
│   ├── avid_ab/              # Asynchronous Verifiable Information Dispersal with Abort
│   │   └── src/              #   protocol/ (init, echo, ready, state), handlers/, rs.rs (Reed–Solomon)
│   └── sh2t/                 # Degree-2t sharing with Abort
│       └── src/protocol/     #   dealing (init), CTRBC, AVID dispersal, RA, per-instance state
│
├── mpc/                      # The MPC engine: runs the protocol phases for a hosted Application
│   └── src/
│       ├── protocol/
│       │   ├── rand_sharings/          # Preprocessing: random sharings, masks, random bits, per-depth reservation
│       │   ├── multiplication/         # Linear, quadratic, weak and masked multiplication; output reconstruction
│       │   ├── online_phase/           # Input dealing, depth scheduling, application reveals
│       │   ├── public_reconstruction/  # Batched public opening shared by multiplication and reveal
│       │   └── tuple_verification/     # Tuple compression, common coin, reveal check
│       ├── handlers/         # Message and syncer handlers
│       ├── context.rs        # Per-party engine state and constants
│       ├── input.rs          # Reading parties' input files
│       └── msg.rs, process_msg.rs      # Wire messages and their dispatch
│
├── planner/                  # The Planner: comparison, min/max, truncation, fixed-point ops over the engine
│   ├── src/
│   │   ├── api/              # engine.rs (the engine's Application trait), application.rs (PlannerApplication, Op)
│   │   ├── ops/              # One file per op: add, mul, reveal, mask_reveal, compare, drelu, max, min, truncate, fixed_mul
│   │   ├── primitives/       # edaBits, the carry-tree bitwise less-than, Mersenne-prime arithmetic
│   │   ├── plan.rs           # Compiles op-depths into engine rounds
│   │   └── planner.rs        # The Application the engine hosts, driving the PlannerApplication
│   ├── tests/plaintext.rs    # The Planner against a plaintext engine
│   └── README.md             # The Planner's API
│
├── apps/                     # Applications, each a PlannerApplication with its own binary (lib.rs + main.rs)
│   ├── anonymous_broadcast/  # Butterfly mixing network over k messages
│   ├── bristol_circuit/      # Evaluates any .arith circuit (parser.rs)
│   ├── btx_setup/            # Batched threshold encryption setup (commitments.rs: shares lifted into the pairing groups)
│   ├── erc20/                # Private ERC20 transfers with secure comparisons
│   └── reveal_probe/         # End-to-end check of the public reveal
│
├── velox/                    # The engine as one dependency for applications: re-exports, CLI arguments, node launch, syncer
├── circuit/                  # The .arith circuit IR (docs/CIRCUIT_FORMAT.md)
├── config/                   # The config binary, which generates per-node configuration files
├── benchmark/                # AWS benchmarking infrastructure and analysis tools
├── testdata/                 # Node configurations (testdata/<n>), circuits, erc20 ledgers, party inputs
├── scripts/                  # test.sh, test_btx.sh, and the bench_ops.py / bench_erc20.py benchmarks
├── docs/                     # Circuit format, comparison design, and other design notes
├── logs/                     # Runtime logs from protocol execution
└── images/                   # Project assets (logo, etc.)
```

## Applications

Each application is a `PlannerApplication` with its own binary under `apps/`,
run through `scripts/test.sh` (`APP_BIN=<name>`):

| application | what it computes | fields |
|---|---|---|
| `anonymous_broadcast` | a butterfly mixing network: `k` messages shuffled anonymously | any (`m61` default) |
| `bristol_circuit` | any `.arith` circuit, including comparison, ReLU, max/min, truncation and fixed-point gates ([`docs/CIRCUIT_FORMAT.md`](docs/CIRCUIT_FORMAT.md)) | `m61base`, `m31base` |
| `btx_setup` | the key setup of batched threshold encryption: shares of `τ¹ … τ^{2B}`; with `--delta`, Policharla's indexed variant (shares of `sk·τ^d`, `sk⁻¹·τ^i`, `β`) | any (`bls381` default) |
| `erc20` | private ERC20 payments: secret balances and amounts, every transfer checked for funds with secure comparisons (`apps/erc20/src/lib.rs`) | `m61base`, `m31base` |
| `reveal_probe` | an end-to-end check of the public reveal | any |

### Running each application

Every application runs through `scripts/test.sh {num_parties} {label} {comp} [rand_batches]`,
except `btx_setup`, which has `scripts/test_btx.sh`. Environment variables pick
the application. The config directory defaults to `testdata/<num_parties>`
(`testdata/10` ships with the repository; generate others with the `config`
binary as in step 5, using `--target testdata/<n>/`). `comp` must be at least 2;
use 10.

```bash
# anonymous_broadcast (default; field m61). The 2nd argument is --messages, the anonymity set size k.
./scripts/test.sh 10 256 10

# bristol_circuit (runs when CIRCUIT is set; field m61base by default). The 2nd argument only labels the logs.
CIRCUIT=testdata/circuits/comparison.arith ./scripts/test.sh 10 cmp 10
FIELD=m31base CIRCUIT=testdata/circuits/simple_mul.arith ./scripts/test.sh 10 mul 10

# erc20 (m61base or m31base; set FIELD, since test.sh otherwise defaults to m61)
APP_BIN=erc20 APP_ARG="--ledger testdata/erc20/example.ledger" FIELD=m61base ./scripts/test.sh 10 erc20 10

# reveal_probe (any field)
APP_BIN=reveal_probe APP_ARG="--values 4" ./scripts/test.sh 10 probe 10

# btx_setup (field bls381 by default): {num_parties} {batch_size B} {comp}
./scripts/test_btx.sh 10 16 10
DELTA=4 ./scripts/test_btx.sh 10 16 10    # indexed variant, delta <= B
```

Each party reads its inputs from `testdata/inputs/`: `input_<id>.txt` for
anonymous broadcast, `circuit_input_<id>.txt` for circuits, and
`erc20_input_<id>.txt` for erc20 (balances, then transfer amounts). Anonymous
broadcast and circuits fall back to random inputs when a file is missing, and
erc20 falls back to zeros. `TESTDIR`, `TYPE` (build profile, default `release`) and
`FIELD` override the defaults.

The syncer writes phase latencies to `logs/syncer_n_<n>_<label>_<comp>.log`
(`logs/syncer_btx_n_<n>_<B>_<comp>.log` for btx_setup), and each party writes
its output to `logs/party-<i>-...log`. `test.sh` kills the previous run's
processes, but its kill list leaves out `btx_setup`, which only `test_btx.sh`
kills. If ports are still held, free them as in step 9.
`scripts/bench_ops.py` and `scripts/bench_erc20.py` benchmark circuits and
erc20 on the 10-party fixture.

Secure comparison, truncation and fixed-point multiplication (issue #5) are
described in [`docs/comparison.md`](docs/comparison.md); the Planner's API in
[`planner/README.md`](planner/README.md); circuits with those operations as
gates in [`docs/CIRCUIT_FORMAT.md`](docs/CIRCUIT_FORMAT.md).

# Running in AWS
Please refer to the `benchmark/` directory for instructions to run benchmarks on AWS.

## Performance Results
The following results were achieved in a single-region AWS testbed with `n=16` parties, each party running on a `c5.4xlarge` device with 16 cores and 32 GB RAM.

| k (Anonymity Set Size) | Time (seconds) |
|------------------------|----------------|
| 256                    | 1.32           |
| 512                    | 2.40           |
| 1024                   | 5.62           |


# Dependencies in the codebase
The artifact is organized into the following modules of code.

1. The config directory contains code pertaining to configuring each node in the distributed system. 
Each node requires information about port to use, network addresses of other nodes, symmetric keys to establish pairwise authenticated channels between nodes, and protocol specific configuration parameters. 
Code related to managing and parsing these parameters is in the config directory. 
This library has been borrowed from the libchatter (https://github.com/libdist-rs/libchatter-rs) repository.

2. Networking: This repository uses the libnet-rs (https://github.com/libdist-rs/libnet-rs) networking library. 
Similar libraries include networking library from the narwhal (https://github.com/MystenLabs/sui/tree/main/narwhal/) repository. The nodes use the tcp protocol to send messages to each other.

3. The protocol directory contains code that implements the building blocks of the codebase. 
The protocol employs ACSS, AVID, and Sh2t protocols, which build on smaller building blocks like Reliable Broadcast, Reliable Agreement, and Asynchronous consensus. 
These building blocks have been implemented in the Secure Distributed Computing repository (https://github.com/scalable-mpc/secure-distributed-computing-protocols). 

# Architecture
The following architecture diagram describes the components of Velox and their dependencies. 
The diagram can be interpreted as a Directed Graph with source vertices.
Each source vertex has been implemented using the composing building blocks from the Secure Distributed Computing Repository (https://github.com/scalable-mpc/secure-distributed-computing-protocols).

<img src="images/MPC_component_flow_updated.png"/>

The components Reliable Broadcast, Asynchronous Verifiable Information Dispersal, and Asynchronous Common Subset have been implemented in Secure Distributed Computing Protocols repository (https://github.com/scalable-mpc/secure-distributed-computing-protocols).
The remaining components have been implemented in this repository. 
