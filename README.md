# myNet_It_normal

## Setup
Please run the following pip and conda install commands:<br />
```
conda create -n envMyNet_3d python=3.7
conda activate envMyNet_3d
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub(安装nvidiacub)
conda install pytorch3d -c pytorch3d(安装pytorch3d)
conda install pyg -c pyg
pip install point-cloud-utils==0.27.0
pip install plyfile
pip install pandas
```