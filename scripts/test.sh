# A script to test quickly

killall anonymous_broadcast bristol_circuit &> /dev/null
rm -rf /tmp/*.db &> /dev/null
vals=(27000 27100 27200 27300)

#rand=$(gshuf -i 1000-150000000 -n 1)
TESTDIR=${TESTDIR:="testdata/$1"}
TYPE=${TYPE:="release"}

# Finite field to run over: m61 (default), stark252, or bn254.
FIELD=${FIELD:="m61"}

# Optional 4th arg: number of random-sharing sub-batches (--rand-batches).
# When omitted, the node falls back to mpc::NUM_RAND_BATCHES.
RAND_BATCHES_ARG=${4:+--rand-batches $4}

# Applications own their binaries. Set CIRCUIT to a .arith file to run the
# bristol_circuit application; otherwise anonymous_broadcast runs, and `--messages`
# is its anonymity set size. The syncer role is served by whichever binary is in
# play, via `--protocol sync`.
if [ -n "${CIRCUIT:-}" ]; then
    APP_BIN=bristol_circuit
    APP_ARG="--circuit $CIRCUIT"
else
    APP_BIN=anonymous_broadcast
    APP_ARG="--messages $2"
fi

# Run the syncer now
./target/$TYPE/$APP_BIN \
    --config $TESTDIR/nodes-0.json \
    --ip ip_file \
    --protocol sync \
    --syncer $TESTDIR/syncer \
    $APP_ARG \
    --comp $3 \
    --field $FIELD \
    $RAND_BATCHES_ARG \
    --byzantine false > logs/syncer_n_$1_$2_$3.log &

for((i=0;i<$1;i++)); do
./target/$TYPE/$APP_BIN \
    --config $TESTDIR/nodes-$i.json \
    --ip ip_file \
    --protocol mpc \
    --syncer $TESTDIR/syncer \
    $APP_ARG \
    --comp $3 \
    --field $FIELD \
    $RAND_BATCHES_ARG \
    --byzantine false > logs/party-$i-n_$1_$2_$3.log &
done

# Kill all nodes sudo lsof -ti:7000-7015 | xargs kill -9
