## 💡Getting Started
```bash
1. 训练看/gpfs/0607-cluster/chenhao/DoubleRL-VLA/scripts下面的train.sh，选择一个train.py。用train_janus_no_siglip_encoder.py或者train_janus_no_siglip_encoder_diff.py。no_gen_encoder和two_vis_encoder已经停止更新了，勿用。

2. rlbench的测试用/gpfs/0607-cluster/chenhao/DoubleRL-VLA/scripts/test_rlbench.sh，选择对应的test.py。

3. 用/gpfs/0607-cluster/chenhao/DoubleRL-VLA/utils/gen_rlbench_json_stats.py生成训练数据，需要唯一源数据，rlbench的npy数据，我用的源数据在/gpfs/0607-cluster/chenhao/data/rlbench/keyframe_fast_slow_chunk8_addlast_0806/for_rlds。

4. 真机参考/gpfs/0607-cluster/chenhao/DoubleRL-VLA/utils/gen_real_world_json_stat.py生成数据。
```


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