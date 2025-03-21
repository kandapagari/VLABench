# VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks (Early Preview-Version!)

<span style="font-size:16px"> 🚨 <span style="color:#AB4459;">**NOTICE:**</span> 🎁 The early preview version is released on my birthday (12.25) as a gift for myself🎄! Most codes are still under management or even reconstruction for a more robust and user-friendly version.（Sorry, I’ve been so busy these days). The Complete Version will be open-sourced around the Chinese Lunar New Year🧧! </br> <span style="font-size:14px;font-style: italic;">I don’t like the phrase "code coming soon"; it often feels like I’ll never actually see the code on GitHub, which can be quite frustrating. So this early version is my promise.</span></span>

###  🎓 [**Paper**](docs/pdf/paper.pdf) |  🌐 [**Project Website**](https://vlabench.github.io/) ｜ 🤗 [**Hugging Face**](https://huggingface.co/datasets/VLABench/eval_vlm_v0)
<img src="docs/images/Figure1_overview.png" width="100%" />


## News
* **2024/12/25** The preview verison of VLABench has been released! This version is a gift for my birthday, happy birthday to myself and merry chrismas to u!🎁🎉 The preview version showcases most of the designed tasks and structure, but the functionalities are still being managed and tested. I aim to provide you with a highly user-friendly and efficient evaluation tool, so I kindly ask for your patience during this process. Thank you for your understanding, and I look forward to delivering a polished and seamless experience soon!

## Installation

### Install VLABench
1. Prepare conda environment
```sh
conda create -n vlabench python=3.10
conda activate vlabench

git clone https://github.com/OpenMOSS/VLABench.git
cd VLABench
pip install -r requirements.txt
pip install -e .
```
2. Download the assets
```sh
python script/download_assetes.py
```
The script will automatically download the necessary assets and unzip them into the correct directory.

### Issues with octo
Some experiences to create octo evaluation env:
```sh
    conda env remove -n octo
    conda create -n octo python=3.10
    conda activate octo
    pip install -e .
    pip install "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html flax==0.7.5 
    pip install tensorflow==2.15.0 pip install dlimp@git+https://github.com/kvablack/dlimp@5edaa4691567873d495633f2708982b42edf1972 
    pip install distrax==0.1.5 
    pip install tensorflow_probability==0.23.0 
    pip install scipy==1.12.0 
    pip install einops==0.6.1
    pip install transformers==4.34.1 
    pip install ml_collections==0.1.0 
    pip install wandb==0.12.14 
    pip install matplotlib 
    pip install gym==0.26 
    pip install plotly==5.16.1
    pip install orbax-checkpoint==0.4.0
```
Note: Line 5 "cuda12_pip" may be replaced by other proper version according to your machine. Refer to  [jax installation](https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu).

Make sure jax version=0.4.20 and flax version=0.7.5

    pip show jax flax jaxlib

Run this to verify installation successful

    python -c "from octo.model.octo_model import OctoModel; model = OctoModel.load_pretrained('hf://rail-berkeley/octo-base-1.5'); print('Model loaded successfully')"

## Data Collection
### Run scripts to generate hdf5 dataset with multi-processing
We provide a brief tutorial in `tutorials/2.auto_trajectory_generate.ipynb` and the whole codes are in `scripts/trajectory_generation.py`. Trajectory generation can be sped up several times by using multiple processes. A naive way to use it is: 
```sh
sh data_generation.sh
```
Currently, the version does not support multi-processing environment within the code. We will optimize the collection efficiency as much as possible in future updates. After running the script, each trajectory will be stored as a hdf5 file in the directory you specify.

### Convert to rlds format
Due to some frameworks such as [Octo](https://github.com/octo-models/octo) and [Openvla](https://github.com/openvla/openvla) using data in the RLDS format for training, we refer to the process from [rlds_dataset_builder](https://github.com/kpertsch/rlds_dataset_builder) to provide an example of converting the aforementioned HDF5 dataset into RLDS format data.
First, run 
```sh
python scripts/convert_to_rlds.py --task [list] --save_dir /your/path/to/dataset
```
This will create a python file including the task rlds-builder in the directory.
Then
```sh
cd /your/path/to/dataset/task

tfds build
```
This process consumes a long time with only single process, and we are testing multithreading mthod yet. The codes of original repo seem to have some bugs. 

### Convert to Lerobot format
Following the Libero dataset process way of [openpi](https://github.com/Physical-Intelligence/openpi), we offer a simple way to convert hdf5 data files into lerobot format.
Run the script by 
```sh
python scripts/convert_to_lerobot.py --dataset-name [your-dataset-name] --dataset-path /your/path/to/dataset --max-files 100
```
The processed Lerobot dataset will be stored defaultly in your `HF_HOME/lerobot/dataset-name`.

## Recent Work Todo
- [x] Organize the functional code sections. 
    - [x] Reconstruct the efficient, user-friendly, and comprehensive evaluation framework. 
    - [x] Manage the automatic data workflow for existing tasks.
    - [x] Improve the DSL of skill libarary.
- [x] Release the trejectory and evaluation scripts.
- [ ] Release the left few tasks not released in preview version.
- [ ] Test the interface of humanoid and dual-arm manipulation.
- [ ] Integrate the commonly used VLA models for facilitate replication. (Continously update)
- [ ] Leaderboard of VLAs and VLMs in the standard evaluation 
    - [ ] Release standard evaluation datasets/episodes, in different dimension and difficulty level.
    - [ ] Release standard finetune dataset.

## Expandation 
VLABench adopts a flexible modular framework for task construction, offering high adaptability. You can follow the process outlined in [tutorial 6](tutorials/6.expandation.ipynb).
<!-- ### Register New Entity
1. Process the obj file with `obj2mjcf`(https://github.com/kevinzakka/obj2mjcf). Here is an use demo, `obj2mjcf --verbose --obj-dir your_own_obj_dir --compile-model --save-mjcf --decompose`
2. Put the processed xml files/directory to somewhere under VLABench/assets/meshes.
3. If it's a new class of entity, please register a entity class in VLABench/tasks/components with global register. Then, import the class in the `VLABench/tasks/components/__init__.py`.
4. Register it in `VLABench/configs/constant.py` for global access.

### Register New Task
1. Create new task class file under `VLABench/tasks/hierarchical_tasks`. And register it with global register in `VLABench/utils/register.py`. Notice that if the current condition can not met your requirement, you should write a single Condition class in `VLABench/tasks/condition.py`.
2. Import the new task class in `VLABench/tasks/hierarchical_tasks/__init__.py`. -->

## Evaluate
I am currently updating the evaluation process, which includes making the tools more user-friendly, speeding up the entire evaluation workflow, and implementing a more comprehensive scoring system.
```sh
python scirpts/eval.py --n-sample 20 --model your_model_script
```

## Citation
```bibtex
@misc{zhang2024vlabench,
      title={VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks}, 
      author={Shiduo Zhang and Zhe Xu and Peiju Liu and Xiaopeng Yu and Yuan Li and Qinghui Gao and Zhaoye Fei and Zhangyue Yin and Zuxuan Wu and Yu-Gang Jiang and Xipeng Qiu},
      year={2024},
      eprint={2412.18194},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2412.18194}, 
}
```