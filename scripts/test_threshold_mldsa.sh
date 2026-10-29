# Run the threshold_mldsa application locally: n parties make an ML-DSA-44
# key, prepare the signing attempts and sign the messages. Each party logs the
# hash of the public key on the line "ThresholdMlDsa: public key of 1312
# bytes, tr = ..." (all parties must log the same hash), and the result on
# the line "ThresholdMlDsa: <k> of <S> messages signed". Each signature has
# its line "ThresholdMlDsa: message <i>: signature of 2420 bytes ...
# RustCrypto ml-dsa accepts it: true" (or "message <i>: no signature: ...").
#
#   ./scripts/test_threshold_mldsa.sh {num_parties} {compression_factor} [attempts] [signatures]
#
# attempts is --attempts, the signing attempts for each message (default
# 16). signatures is --signatures, the messages to sign (default 1).
#
# Logs: logs/party-<id>-mldsa_n_<n>_<comp>.log. Each party runs under
# /usr/bin/time, which writes the party's peak memory ("maximum resident set
# size": bytes on macOS, kbytes on Linux) and its stderr to
# logs/rss-<id>-mldsa_n_<n>_<comp>.txt when the party exits or is killed.
#
# Same knobs as test.sh: TESTDIR for the config directory, TYPE for the build
# profile. The field is fixed: the ML-DSA field F_q, q = 8380417.

# A previous run's parties keep their ports until killed; give the sockets a
# moment to close before binding them again.
killall -9 threshold_mldsa &> /dev/null
sleep 1
rm -rf /tmp/*.db &> /dev/null

TESTDIR=${TESTDIR:="testdata/$1"}
TYPE=${TYPE:="release"}

APP_BIN=threshold_mldsa

ATTEMPTS_ARG="--attempts ${3:-16} --signatures ${4:-1}"

if [ "$(uname)" = "Darwin" ]; then TIME_CMD="/usr/bin/time -l"; else TIME_CMD="/usr/bin/time -v"; fi

./target/$TYPE/$APP_BIN \
    --config $TESTDIR/nodes-0.json \
    --ip ip_file \
    --protocol sync \
    --syncer $TESTDIR/syncer \
    --comp $2 \
    $ATTEMPTS_ARG \
    --byzantine false > logs/syncer_mldsa_n_$1_$2.log &

for((i=0;i<$1;i++)); do
$TIME_CMD ./target/$TYPE/$APP_BIN \
    --config $TESTDIR/nodes-$i.json \
    --ip ip_file \
    --protocol mpc \
    --syncer $TESTDIR/syncer \
    --comp $2 \
    $ATTEMPTS_ARG \
    --byzantine false > logs/party-$i-mldsa_n_$1_$2.log 2> logs/rss-$i-mldsa_n_$1_$2.txt &
done
