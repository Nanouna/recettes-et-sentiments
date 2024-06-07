
# Setup to make tensorflow work with GPU on WSL
Author: @VeMBe


> [!WARNING] IMPORTANT
> - This Setup is valid as of 3/12/2024, things might change in the mean time
> - I did it on Windows 11
> - It will download 4gb+ of data



## TLDR:
- You don't need to change your regular NVidia drivers
- install CUDA toolkit
- install cuDNN
- `pip install "tensorflow[and-cuda]"`
- feel the power

## Sources
- [This blogpost](https://discuss.tensorflow.org/t/tensorflow-2-13-0-does-not-find-gpu-with-cuda-12-1/18939)
- [Video from NVidia](https://www.youtube.com/watch?v=JaHVsZa2jTc) on how to install CUDA easily
- [NVidia CUDA download link](https://developer.nvidia.com/cuda-downloads) (WARNING: select linux => WSL-Ubuntu!)
- [NVidia cuDNN download link](https://developer.nvidia.com/cudnn-downloads)
- [tensorflow WSL install link](https://www.tensorflow.org/install/pip)

## 1) Pre-checks

### Check if you have an nvidia GPU

```bash
nvidia-smi # should return a table displaying your GPU resources
```



You can manually  verify if your GPU is in the [list](https://developer.nvidia.com/cuda-gpus) (but the `nvidia-smi` command is most robust)

### Update your NVidia Drivers
Just your regular update on the Windows side, normally you should have the NVidia GeForce software, check for updates and update your drivers


> [!tip] quick info
> If you don't update your drivers, the `nvidia-smi` command will show your maximum CUDA version available for your driver, which will be likely less than the CUDA version we'll download ==> will create errors


## 2) Install CUDA

https://www.cherryservers.com/blog/install-cuda-ubuntu


[NVidia CUDA download link](https://developer.nvidia.com/cuda-downloads)
This install uses a run file, which is apparently safer as it prevents downloading all of WSL by mistake

Follow [this](https://www.youtube.com/watch?v=JaHVsZa2jTc)  video:
```bash
sudo apt update

# install build-essential, needed to install using a run-file
sudo apt install build-essential

# get command from https://developer.nvidia.com/cuda-downloads
# Choose: linux->x86_64->WSL-Ubuntu->2.0->runfile(local)
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
# Run install, follow instructions on screen
sudo sh cuda_12.4.0_550.54.14_linux.run

# check that install went correctly, put correct version in path (<cuda-12.4>)
/usr/local/cuda-12.4/extras/demo_suite/deviceQuery
# should print GPU info
```

Current CUDA version: 12.4

## 3) Install cuDNN

NVidia cuDNN download link](https://developer.nvidia.com/cudnn-downloads)


```bash
# get command from https://developer.nvidia.com/cudnn-downloads
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb

sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# use correct install based on your CUDA version
sudo apt-get -y install cudnn-cuda-12
#choose another if you're not on CUDA 12


# Export CUDNN_PATH and LD_LIBRARY_PATH for tensorflow
export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pyenv prefix)/lib/:$CUDNN_PATH/lib

exec zsh # restart shell
# If you don't run those commands, tensorflow might not see your GPU still

```

Current cuDNN version: 9.0.0

## 4) install Tensorflow

> [!NOTE] Notes
> - I haven't tried just using the regular tensorflow version
> - I downloaded tensorflow[and-cuda] in a fresh environment to make sure it worked fine
> - I don't know how a full setup install would go in terms of dependency conflicts

**At this point** I would reboot the whole computer just to be sure (for Windows11, actual restart, not a shut-down + boot-up)

Install Tensorflow with the specific CUDA tag (It probably works without, but you never know)

[tensorflow WSL docs](https://www.tensorflow.org/install/pip)
```bash
# install tensorflow
pip install "tensorflow[and-cuda]"

# Verify the installation:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
# you'll still have some warnings, but in the end you get a GPU detected
```

## And you're done!
