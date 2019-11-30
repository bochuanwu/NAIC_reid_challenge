--inplace# desci-reid

#Installation
Make sure your conda is installed.

```
conda create --name descireid python=3.6
source activate descireid
conda install pytorch torchvision cudatoolkit -c pytorch
conda install numpy Cython Pillow scipy matplotlib
pip install opencv-python future yacs gdown

cd ./metrics/rank_cylib
python setup.py build_ext --inplace
```