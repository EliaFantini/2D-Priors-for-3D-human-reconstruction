## Procedure to build the environment needed to run the code

```
conda create -n orig_pifu python=3.8
conda activate orig_pifu 

conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pillow
conda install scikit-image
conda install tqdm


pip install numpy cython
pip install menpo
pip install opencv-python
pip install pyembree
pip install PyOpenGL
pip install ffmpeg
pip install rembg[gpu]
pip install timm
pip install einops
```
