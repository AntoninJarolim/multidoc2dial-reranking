#!/bin/bash

install_dependencies() {
    cd ..
    # Download and install Conda if not already installed
    if ! command -v conda &> /dev/null; then
        echo "Downloading and installing Conda..."
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p ./conda
        rm miniconda.sh
    else
        echo "Conda is already installed."
    fi

    # Initialize Conda
    source /mnt/data/xjarol06_firllm/conda/etc/profile.d/conda.sh

    # Create and activate Conda environment
    echo "Creating Conda environment from environment.yml..."
    conda env create -f multidoc2dial-reranking/environment.yml

    # Proceed with other setup tasks if needed
    # For example, copying data or running other installation scripts
    cd multidoc2dial-reranking
}

# Check if --install option is provided
if [[ "$1" == "--install" ]]; then
    install_dependencies
fi

DIR="data"
if [ ! -d "$DIR" ]; then
    echo "Creating directory $DIR"
    mkdir -p "$DIR"
fi

# Copy data if not already present
if [ ! -d "data/naver_trecdl22-crossencoder-debertav3" ]; then
    echo "Copying training data from homedir to this directory"
    cp -r ~/md2d_data/naver_trecdl22-crossencoder-debertav3 data/naver_trecdl22-crossencoder-debertav3
fi

source /mnt/data/xjarol06_firllm/conda/etc/profile.d/conda.sh
conda activate md2d-fresh 
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
export PYTHONPATH=$(pwd) # assuming you are in root repository folder
cp ~/.host_config.json .host_config.json
git pull

MONGODBSERVER=pcknot6.fit.vutbr.cz
DB_KEY=ce

# Start hyperopt-mongo-worker
hyperopt-mongo-worker --mongo=$MONGODBSERVER:1234/$DB_KEY --poll-interval=3


