#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

test -f scripts/serve-one-spark.sh
test -f overlay/patch_kpool_vllm_backports.py
test -f overlay/patch_partial_prefix_hits.py
grep -q ONE_SPARK_PARTIAL_APC scripts/serve-one-spark.sh

export ACCEPT_DFLASH2_NC_LICENSE=1
export IMAGE="glm53-one-spark-vllm:cocho"
# Set build to 1 only if image does not exist
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  export BUILD=1
else
  export BUILD=0
fi

export MODEL_DIR="$HF_HOME/hub/models--turboderp--GLM-5.3-Flash-exl3"
export ONE_SPARK_MODEL_PATH="/model/snapshots/51058cd551c7e570d87bd32a4adee720edce2349"

export DFLASH_DIR="$HF_HOME/hub/models--incoai--GLM-5.3-Flash-DFlash2"
export ONE_SPARK_DRAFT_PATH="/draft/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a"

export HOST="0.0.0.0"
export PORT=8000
export CONTAINER="glm53-one-spark"

export ONE_SPARK_UTIL=0.85
export ONE_SPARK_CTX=524288
export ONE_SPARK_SEQS=1
export ONE_SPARK_ASYNC=0
export ONE_SPARK_DRAFT_BLOCK=1024
export ONE_SPARK_PARTIAL_APC=1
export ONE_SPARK_MAMBA_SEED_FIX=1
export ONE_SPARK_K=5
export GLM53_INDEXER_WORKSPACE="rightsize"

./start.sh