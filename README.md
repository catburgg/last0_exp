```bash
cd double_rl
conda create -n double_rl python=3.10
conda activate double_rl

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

cd ./LIFT3D/third_party/RLBench
pip install -e .
cd ./LIFT3D
pip install -e .

pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```