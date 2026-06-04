#!/bin/bash
# Usage: bash run.sh --gpus cuda:0 cuda:1 --n_run 10
EPOCHS=2
LR=1e-4
BATCH_SIZE=8
RUNS=10
GPUS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n_run)
            RUNS="$2"
            shift 2
            ;;
        --gpus)
            shift
            while [[ $# -gt 0 && ! $1 == --* ]]; do
                GPUS+=("$1")
                shift
            done
            ;;
        *)
            # Treat positional arguments as GPUs for backward compatibility
            GPUS+=("$1")
            shift
            ;;
    esac
done

NUM_GPUS=${#GPUS[@]}

if [ $NUM_GPUS -eq 0 ]; then
    GPUS=("cuda:0")
    NUM_GPUS=1
fi

# Clear previous results to ensure summary is fresh
rm -rf results

echo "Starting $RUNS-cycle training/testing pipeline on $NUM_GPUS GPUs: ${GPUS[*]}"
echo "Configuration: Epochs=$EPOCHS, LR=$LR, Batch Size=$BATCH_SIZE"
export PYTHONPATH=$PYTHONPATH:$(pwd)

for i in $(seq 1 $RUNS)
do
    SEED=$((RANDOM % 1000))
    echo "========================================"
    echo "Iteration $i/$RUNS | Seed: $SEED"
    echo "========================================"

    # train visa & test mvtec
    echo "[-] Training Visa..."
    torchrun --nproc_per_node=$NUM_GPUS train.py \
        --dataset visa \
        --split test \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch_size "$BATCH_SIZE" \
        --seed "$SEED" \
        --gpus "${GPUS[@]}"

    echo "[-] Testing MVTec with Visa checkpoint..."
    torchrun --nproc_per_node=$NUM_GPUS test.py \
        --dataset mvtec \
        --batch_size "$BATCH_SIZE" \
        --seed "$SEED" \
        --ckpt_path "checkpoints/train_visa/model_$SEED.pth" \
        --gpus "${GPUS[@]}"

    # train mvtec & test others
    echo "[-] Training MVTec..."
    torchrun --nproc_per_node=$NUM_GPUS train.py \
        --dataset mvtec \
        --split test \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch_size "$BATCH_SIZE" \
        --seed "$SEED" \
        --gpus "${GPUS[@]}"

    for ds in visa btad dagm mpdd dtd
    do
        echo "[-] Testing $ds with MVTec checkpoint..."
        torchrun --nproc_per_node=$NUM_GPUS test.py \
            --dataset "$ds" \
            --batch_size "$BATCH_SIZE" \
            --seed "$SEED" \
            --ckpt_path "checkpoints/train_mvtec/model_$SEED.pth" \
            --gpus "${GPUS[@]}"
    done
done