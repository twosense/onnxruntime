set -e -o -x

SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"
BUILD_CONFIG=$1

source /usr/local/miniconda3/bin/activate lotus-py35
if [ $? -ne 0 ]; then
    /usr/local/miniconda3/bin/conda env create \
        --file $SCRIPT_DIR/Conda/conda-linux-lotus-py35-environment.yml \
        --name lotus-py35 \
        --quiet \
        --force
    source /usr/local/miniconda3/bin/activate lotus-py35
fi

python $SCRIPT_DIR/../../build.py --build_dir /home/lotusdev/gpubuild \
    --config $BUILD_CONFIG \
    --skip_submodule_sync \
    --enable_pybind \
    --use_cuda \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr/local/cudnn-7.0/cuda

source /usr/local/miniconda3/bin/deactivate lotus-py35

exit $?