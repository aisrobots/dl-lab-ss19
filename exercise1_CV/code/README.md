# Installation guide: TF pool

These steps will set up the environment when you work on the pool computer at the technical faculty

1. Clone this repository and create a virtual env to work in
    ```Shell
    git clone https://github.com/aisrobots/dl-lab-ss19
    virtualenv --no-site-packages -p python3 venv
    ```
2. Activate environment and install dependencies

    ```Shell
    source venv/bin/activate
    pip install --upgrade pip
    pip install numpy matplotlib Pillow torch torchvision
    ```
    
3. Verify correct dataset setup by visually inspecting samples of the dataset with

    ```Shell
    cd dl-lab-ss19/exercise1_CV/code
    python3 show_data.py
    python3 show_data_seg.py
    ```
    
5. Download the pretrained model and visualize its predictions

    ```Shell
    wget https://lmb.informatik.uni-freiburg.de/data/DL_Labcourse/trained_net.model
    python3 run_forward.py
    ```

When all of the steps above were successful you are ready to start implementing the exercises lined out in the accompanying pdf file.
    
# Installation guide: Elsewhere

For the TF pool we created a central instance of the dataset to be used, 
but if you prefer to use the Google Cloud or use your personal computer you will need to download and setup the dataset.
Please note that using a computer with a proper Nvidia GPU is imperative for these assignments to keep training times in an acceptable regime. 
Training a network to complete convergence with a good GPU will take approx. 6 hours.

1. Download dataset from [here](https://lmb.informatik.uni-freiburg.de/data/DL_Labcourse/coco_subset.zip)  (~870 MB)
2. Adapt path to dataset ("COCO_PATH") in model/data.py and in model/data_seg.py


# Running experiments in the TF pool
Using the TF pool hardware is the preferred option for this exercise and therefore we give you a short summary on how to run jobs there.
If you are not in the pool connect to the pool via ssh

    ssh username@login.informatik.uni-freiburg.de

You should use the machines tfpool25-46 or tfpool53-63 as they provide the best GPUs. So connect to one of them 
    
    ssh tfpoolXX
    
You can check the GPU status with nvidia-smi. It will tell you about which processes currently occupy the GPU.

    nvidia-smi
    
If the GPU is occupied by other processes, i.e. somebody else is training on this machine already, chose another machine.
Once you found a free one you can use it for training. It would be possible to start the training script right away, 
but this would require to keep the shell open during the whole training procedure. Therefore it is best practice to    
start a screen session that will allow you to close the connection and keep the process running meanwhile.

    screen -aS t01
    
Now you can start the training script you have written.

    python3 train.py
    
Use the hotkeys CRTL + A + D to detach from the session.
If you want to get back to the process you can do so by attaching to the process again.

    screen -r t01 
    
Please keep in mind, that there is no guarantee that your job will complete, because sometime the pool computer will be shut down by other students.
Therefore you might want to write your training procedure such that it saves its current state occasionally and implement an option to resume training from a saved state.
