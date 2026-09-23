# Run the btx_setup application locally: n parties generate shares of
# tau^1..tau^{2B}; each prints its shares and its commitments to them into its
# own log, logs/party-<id>-btx_n_<n>_<B>_<comp>.log.
#
#   ./scripts/test_btx.sh {num_parties} {batch_size} {compression_factor}
#
# DELTA=<d> runs Policharla's indexed variant with index radius d <= B.
#
# Same knobs as test.sh: TESTDIR for the config directory, TYPE for the build
# profile, FIELD for the field (defaults to bls381, where the scheme lives).

# A previous run's parties keep their ports until killed; give the sockets a
# moment to close before binding them again.
killall -9 btx_setup &> /dev/null
sleep 1
rm -rf /tmp/*.db &> /dev/null

TESTDIR=${TESTDIR:="testdata/$1"}
TYPE=${TYPE:="release"}
FIELD=${FIELD:="bls381"}

# Optional 4th arg: number of random-sharing sub-batches (--rand-batches).
RAND_BATCHES_ARG=${4:+--rand-batches $4}

APP_BIN=btx_setup
APP_ARG="--batch_size $2${DELTA:+ --delta $DELTA}"

./target/$TYPE/$APP_BIN \
    --config $TESTDIR/nodes-0.json \
    --ip ip_file \
    --protocol sync \
    --syncer $TESTDIR/syncer \
    $APP_ARG \
    --comp $3 \
    --field $FIELD \
    $RAND_BATCHES_ARG \
    --byzantine false > logs/syncer_btx_n_$1_$2_$3.log &

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
    --byzantine false > logs/party-$i-btx_n_$1_$2_$3.log &
done
