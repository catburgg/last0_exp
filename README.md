<div align="center">

# LaST​<sub>0</sub>​: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision-Language-Action Model

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

[🌐**Project Page**](https://vla-last0.github.io/) | [✍️**Paper(Arxiv)**](https://arxiv.org/abs/2601.05248v2) | [🎥**Demo**](https://vla-last0.github.io/)

Zhuoyang Liu, Jiaming Liu, Hao Chen, Jiale Yu, Ziyu Guo, Chengkai Hou, Chenyang Gu, Xiangju Mi, Renrui Zhang, Kun Wu, Zhengping Che, Jian Tang, Pheng-Ann Heng, Shanghang Zhang

</div>

![](asset/method.png)
**🤖LaST​<sub>0</sub>​ is a framework that enables efficient reasoning before acting through a Latent Spatio-Temporal Chain-of-Thought (CoT), capturing fine-grained physical and robotic dynamics that are often difficult to verbalize.** Specifically, we introduce a token-efficient latent CoT space that models future visual dynamics, 3D structural information, and robot proprioceptive states, and further extends these representations across time to enable temporally consistent implicit reasoning trajectories. Furthermore, LaST<sub>0</sub> adopts a dual-system architecture implemented via a Mixture-of-Transformers design, where a reasoning expert conducts low-frequency latent inference and an acting expert generates high-frequency actions conditioned on robotics-oriented latent representations. To facilitate coordination, LaST<sub>0</sub> is trained with heterogeneous operation frequencies, enabling adaptive switching during deployment.

## ✨ News ✨

- [2026/00/00] The code of LaST​<sub>0</sub> is released! Including training and evaluating on various benchmarks. We also release our pretrained and finetuned checkpoints! 🚀
- [2026/02/02] A new version of LaST​<sub>0</sub> is updated on arxiv, more experiments added!🚀
- [2026/01/08] is now live on arXiv! The code is also comming soon.🚀

## 📦 Installation

```bash
cd DoubleRL-VLA
conda create -n double_rl python=3.10
conda activate double_rl

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

cd ./LIFT3D/third_party/RLBench
pip install -e .
cd ../..
pip install -e .

pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

The code is built using Python 3.10, we also recommand to use Python above Python 3.10. We require PyTorch >= 2.2.0 and CUDA >= 12.0 (It may run with lower versions, but we have not tested it).
We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and create an environment as follows:

```bash
conda create --name last0 python=3.10
```

Next, clone our repo and install the required packages with the following commands:

```bash
git clone https://github.com/ZhuoyangLiu2005/last0
cd last0
pip install -e .
```

If you need to use the traning code, please also install the [Flash Attention](https://github.com/Dao-AILab/flash-attention):

```bash
# Training additionally requires Flash-Attention 2 (https://github.com/Dao-AILab/flash-attention)
pip install packaging ninja

# Verify Ninja --> should return exit code "0"
ninja --version; echo $?

# Install Flash Attention 2
# =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install "flash-attn==2.5.5" --no-build-isolation
```

## 🧩 Framework

Our code is built based on [OpenVLA](https://github.com/openvla/openvla) and [CogACT](https://github.com/microsoft/CogACT) and is organized in the following framework:

- `conf`: config files for hydridvla training
- `scripts`: scripts for training and testing
- `training`: contains strategies for training
- `models`: contains hybridvla models, including backbone & diffusion & vlm & vla
- `util`: contains different kinds of tools funtion
- `vla`: from openvla's vla structure, including action tokenizer, etc.

## 💡Getting Started

1. 训练看/gpfs/0607-cluster/chenhao/DoubleRL-VLA/scripts下面的train.sh，选择一个train.py。用train_janus_no_siglip_encoder.py或者train_janus_no_siglip_encoder_diff.py。no_gen_encoder和two_vis_encoder已经停止更新了，勿用。
2. rlbench的测试用/gpfs/0607-cluster/chenhao/DoubleRL-VLA/scripts/test_rlbench.sh，选择对应的test.py。
3. 用/gpfs/0607-cluster/chenhao/DoubleRL-VLA/utils/gen_rlbench_json_stats.py生成训练数据，需要唯一源数据，rlbench的npy数据，我用的源数据在/gpfs/0607-cluster/chenhao/data/rlbench/keyframe_fast_slow_chunk8_addlast_0806/for_rlds。
4. 真机参考/gpfs/0607-cluster/chenhao/DoubleRL-VLA/utils/gen_real_world_json_stat.py生成数据。

We release our pretrained model's parameters as follows:

- [Robotic Large-Scale Pretrained Checkpoint](https://pan.baidu.com/s/134S9y8UwoNlyw3yUKozbRw?pwd=1spu)
- [Simulation-Finetuned Checkpoint](https://pan.baidu.com/s/1f5zpPKoAJDRIHFIH602Bqg?pwd=3ca1)

Our model requires PIL image and text prompt as input, please refer to the code below for the minimal inference :

```python
# also see scripts/test_toy.py
from PIL import Image
from vla import load_vla
import torch
import numpy as np
import time

model = load_vla(
        '<absolute-path-to-ckpt>',
        load_for_training=False,
        future_action_window_size=0,
        use_diff=True, # choose weither to use diff
        action_dim=7,
        )

# (Optional) use "model.vlm = model.vlm.to(torch.bfloat16)" to load vlm in bf16

model.to('cuda:0').eval()

example_image: Image.Image = Image.open('<path-to-Hybrid-VLA>/assets/000.png') 
example_prompt = "close the laptop"
example_cur_robot_state = np.array([ 0.27849028, -0.00815899,  1.47193933, -3.14159094,  0.24234043,  3.14158629,  1.        ])
actions_diff, actions_ar, _ = model.predict_action(
            front_image=example_image,
            instruction=example_prompt,
            unnorm_key = 'rlbench',
            cfg_scale = 0.0, 
            use_ddim = True,
            num_ddim_steps = 4,
            action_dim = 7,
            cur_robot_state = example_cur_robot_state,
            predict_mode = 'diff+ar' # or 'ar' or 'diff'
            )
    
print(actions_diff)
```

## 📈 Fully Fine-Tuning

To fully fine-tune the pretrained models, we use PyTorch Fully Sharded Data Parallel(FSDP).The training script used is from [CogACT](https://github.com/microsoft/CogACT).

First, download our pretrain model, and change `--pretrained_checkpoint` to your local ckpt absolute path.

Next, create a Hugging Face user access token and export the token value. Make sure your token have right access to [llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b) repo.

```python
# create .hf_token file and put your user access token in.
cd <path-to-Hybrid-VLA>
vim .hf_token
```

Then launch the training script. We use one node with 8 A100 GPUs as an example.

```sh
# parameter
export PYTHONPATH=<path-to-Hybrid-VLA>:$PYTHONPATH

FUTURE_ACTION_STEPS=0
SETTING=<training-setting>
FREEZE_VISON=true
FREEZE_LLM=false
LOAD_DIT=false
ACTION_TOKENIZER_EXIST=true
USE_DIFF=true
AR_DIFF_LOSS=true
REPEATED_DIFFUSION_STEPS=4
CLASS_DROPOUT_PROB=0.0

DATA_MIX=rlbench
TASK=<your-task-name>
NUM_GPUS=8
NODES=1
BATCH_SIZE=32
EPOCHS=300
LEARNING_RATE=2e-5
ACTION_DIM=7

DATA_ROOT=<your-rlds-data>
EXP_ROOT=<runs-dir> #for example, ./runs
# launch
torchrun --standalone --nnodes ${NODES} --nproc-per-node ${NUM_GPUS} train.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix ${DATA_MIX} \
  --vla.base_vlm prism-dinosiglip-224px+7b \
  --need_to_sub 0 \
  --vla.expected_world_size $((${NUM_GPUS} * ${NODES})) \
  --vla.per_device_batch_size ${BATCH_SIZE} \
  --vla.global_batch_size $((${NUM_GPUS} * ${NODES} * ${BATCH_SIZE})) \
  --vla.learning_rate ${LEARNING_RATE} \
  --vla.epochs ${EPOCHS} \
  --vla.freeze_vision_backbone ${FREEZE_VISON} \
  --vla.freeze_llm_backbone ${FREEZE_LLM} \
  --data_root_dir ${DATA_ROOT}/${TASK} \
  --run_root_dir ${EXP_ROOT} \
  --run_id exp_${TASK}_${SETTING} \
  --image_aug false \
  --wandb_project hybridvla \
  --wandb_entity <your-w&b-account> \
  --save_interval 100 \
  --action_dim ${ACTION_DIM} \
  --repeated_diffusion_steps ${REPEATED_DIFFUSION_STEPS} \
  --action_tokenizer_exist ${ACTION_TOKENIZER_EXIST} \
  --future_action_window_size ${FUTURE_ACTION_STEPS} \
  --class_dropout_prob ${CLASS_DROPOUT_PROB} \
  --use_diff ${USE_DIFF} \
  --ar_diff_loss ${AR_DIFF_LOSS} \
  --is_resume False \
  --pretrained_checkpoint <absolute-path-to-ckpt>
```

## 🔍Test in RLBench

We evaluated our hybridvla in [RLBench](https://github.com/stepjam/RLBench), which based on the CoppeliaSim simulator. Install the virtual environment for testing in RLBench according to the following steps and begin your test. Thanks to the amazing work [LIFT3D](https://github.com/PKU-HMI-Lab/LIFT3D).

```bash
conda create --name hybridvla_test python=3.11
conda activate hybridvla_test

cd Hybrid-VLA
pip install -r test_env_requirements.txt
pip install git+https://github.com/moojink/dlimp_openvla@040105d256bd28866cc6620621a3d5f7b6b91b46
pip install git+https://github.com/arnoldland/openvla@5603207085d55148682e2a35b868ad77d7b42ece

export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

cd LIFT3D/third_party/RLBench
pip install -e .
cd ../../..

cd LIFT3D
pip install -e .
cd ..
```

See the ``scripts/sim.py`` for more details.

We have documented the test results: [Test_Result](https://pan.baidu.com/s/15-kMaHyHqCSSj3YTwhxvWQ?pwd=c9r2). For more implementation details, please see ``test.sh`` and ``scripts/sim.py``.

## 📊 Run on Different Datasets

You may want to train the model on different datasets, thus you need to adjust the code to your own dataset. Here we take bridgev2 dataset as an example:

First, assume that your dataset have been fully prepared with the RLDS format. You should modify the following files:

- `vla/datasets/rlds/oxe/configs.py`

```python
# === Individual Dataset Configs ===
OXE_DATASET_CONFIGS = {
    "rlbench": {
        "image_obs_keys": {"primary": "front_image", "wrist": "wrist_image","secondary": "wrist_image_left"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
   # ... other dataset configs
    "bridgev2": {  # Bridge V2 Dataset
        "image_obs_keys": {"primary": "image_0", "secondary": "image_1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    # ... other dataset configs
}
```

- `vla/datasets/rlds/oxe/mixtures.py`

```python
OXE_NAMED_MIXTURES: Dict[str, List[Tuple[str, float]]] = {
    # === Bridge V2 Dataset ===
    "bridgev2": [
        ("bridgev2", 1.0),                                    # Version of Bridge V2 in Open-X GCP Bucket
    ],
    # === RLBench Dataset ===
    "rlbench": [
         ("rlbench", 1.0),                                   # Original Version of Bridge V2 from Project Website
    ],
    # other dataset mixtures
}
```

- `vla/datasets/rlds/oxe/transforms.py`

```python
def bridge_v2_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    
    trajectory = relabel_bridge_actions(trajectory)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    
    # Note: build trajectory["observation"]["proprio"] here additionally 
    # for we'll use the robot state key 'proprio' in vla/datasets/rlds/config.py
    # you can adjust the state_obs_keys to change this logic adaptively
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["EEF_state"],
            trajectory["observation"]["gripper_state"],
        ),
        axis = -1,
    )
    return trajectory

# === Registry ===
OXE_STANDARDIZATION_TRANSFORMS = {
    "bridgev2": bridge_v2_dataset_transform,
    ### other transform registries
    "rlbench": identity_transform,
}
```

- Finally, modify the training script

You only need to change the `DATA_MIX` and remember to carefully adjust the `data_root_dir`

```sh
export PYTHONPATH=<path-to-Hybrid-VLA>:$PYTHONPATH

FUTURE_ACTION_STEPS=0
SETTING=<training-setting>
FREEZE_VISON=true
FREEZE_LLM=false
LOAD_DIT=false
ACTION_TOKENIZER_EXIST=true
USE_DIFF=true
AR_DIFF_LOSS=true
REPEATED_DIFFUSION_STEPS=4
CLASS_DROPOUT_PROB=0.0

DATA_MIX=bridgev2 # 'rlbench' -> 'bridgev2', corresponds to the key value in vla/datasets/rlds/oxe/mixtures.py
TASK=<your-task-name>
NUM_GPUS=8
NODES=1
BATCH_SIZE=32
EPOCHS=300
LEARNING_RATE=2e-5
ACTION_DIM=7

DATA_ROOT=<your-rlds-data>
EXP_ROOT=<runs-dir> #for example, ./runs
# launch
torchrun --standalone --nnodes ${NODES} --nproc-per-node ${NUM_GPUS} train.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix ${DATA_MIX} \
  --vla.base_vlm prism-dinosiglip-224px+7b \
  --need_to_sub 0 \
  --vla.expected_world_size $((${NUM_GPUS} * ${NODES})) \
  --vla.per_device_batch_size ${BATCH_SIZE} \
  --vla.global_batch_size $((${NUM_GPUS} * ${NODES} * ${BATCH_SIZE})) \
  --vla.learning_rate ${LEARNING_RATE} \
  --vla.epochs ${EPOCHS} \
  --vla.freeze_vision_backbone ${FREEZE_VISON} \
  --vla.freeze_llm_backbone ${FREEZE_LLM} \
  --data_root_dir ${DATA_ROOT}/${TASK} \ # remember to put the bridgev2 dataset under this dir
  --run_root_dir ${EXP_ROOT} \
  --run_id exp_${TASK}_${SETTING} \
  --image_aug false \
  --wandb_project hybridvla \
  --wandb_entity <your-w&b-account> \
  --save_interval 100 \
  --action_dim ${ACTION_DIM} \
  --repeated_diffusion_steps ${REPEATED_DIFFUSION_STEPS} \
  --action_tokenizer_exist ${ACTION_TOKENIZER_EXIST} \
  --future_action_window_size ${FUTURE_ACTION_STEPS} \
  --class_dropout_prob ${CLASS_DROPOUT_PROB} \
  --use_diff ${USE_DIFF} \
  --ar_diff_loss ${AR_DIFF_LOSS} \
  --is_resume False \
  --pretrained_checkpoint <absolute-path-to-ckpt>
```

## 📜️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 BibTeX

```tex
@misc{liu2026last0latentspatiotemporalchainofthought,
      title={LaST$_{0}$: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision-Language-Action Model}, 
      author={Zhuoyang Liu and Jiaming Liu and Hao Chen and Jiale Yu and Ziyu Guo and Chengkai Hou and Chenyang Gu and Xiangju Mi and Renrui Zhang and Kun Wu and Zhengping Che and Jian Tang and Pheng-Ann Heng and Shanghang Zhang},
      year={2026},
      eprint={2601.05248},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2601.05248}, 
}
```


