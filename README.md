# TKTR

This repository is build for TKTR: End-to-End Multi-Object Tracking withTransformer"

## Installation

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n TKTR python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate TKTR 
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

## Usage

### Dataset preparation

Please download MOT17 and crowdhuman datasets


### Training

#### Training on single node

We only use 1 GPU on Tesla V100 to train as following:

```bash
python main.py --output_dir exps2/crowd+xn --resume exps2/crowd+xn/checkpoint.pth    --batch_size 4
```



### Evaluation

You can get the pretrained model of TKTR , then run following command to evaluate it on MOT17 validation set:

```bash
python src/track_half.py mot --load_model <your path>/checkpoint0134.pth  --conf_thres 0.6
```

Then, you can get the result same as paper.