# Run the fhe_decrypt application locally: n parties threshold-decrypt every
# ciphertext in testdata/fhe/ciphertexts.txt (write it, the key and the
# expected plaintexts with testdata/fhe/gen_lwe.py). Each party logs the bits
# it decrypted on the line "FheDecrypt: plaintexts ...", and the online
# latency on the line before.
#
#   ./scripts/test_fhe_decrypt.sh {num_parties} {compression_factor} {method}
#
# method is carry (the Planner's Mod2m, the default) or table (the paper's
# lookup tables). Logs are named after it: logs/party-<id>-fhe_<method>_n_<n>_<comp>.log.
#
# Same knobs as test.sh: TESTDIR for the config directory, TYPE for the build
# profile. The field is fixed: Mersenne-127.

# A previous run's parties keep their ports until killed; give the sockets a
# moment to close before binding them again.
killall -9 fhe_decrypt &> /dev/null
sleep 1
rm -rf /tmp/*.db &> /dev/null

TESTDIR=${TESTDIR:="testdata/$1"}
TYPE=${TYPE:="release"}

METHOD=${3:-carry}

# Optional 4th arg: number of random-sharing sub-batches (--rand-batches).
RAND_BATCHES_ARG=${4:+--rand-batches $4}

APP_BIN=fhe_decrypt

./target/$TYPE/$APP_BIN \
    --config $TESTDIR/nodes-0.json \
    --ip ip_file \
    --protocol sync \
    --syncer $TESTDIR/syncer \
    --comp $2 \
    --method $METHOD \
    $RAND_BATCHES_ARG \
    --byzantine false > logs/syncer_fhe_${METHOD}_n_$1_$2.log &

for((i=0;i<$1;i++)); do
./target/$TYPE/$APP_BIN \
    --config $TESTDIR/nodes-$i.json \
    --ip ip_file \
    --protocol mpc \
    --syncer $TESTDIR/syncer \
    --comp $2 \
    --method $METHOD \
    $RAND_BATCHES_ARG \
    --byzantine false > logs/party-$i-fhe_${METHOD}_n_$1_$2.log &
done
