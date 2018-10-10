# Tensorflow implementation of "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
Fast artistic style transfer by using feed forward network.

<img src="https://github.com/antlerros/tensorflow-fast-neuralstyle/raw/master/images/content/tubingen.jpg" height="200px">

<img src="https://github.com/antlerros/tensorflow-fast-neuralstyle/raw/master/images/styles/Robert_D.jpg" height="200px">
<img src="https://github.com/antlerros/tensorflow-fast-neuralstyle/raw/master/images/output/RobertD_output.jpg" height="200px">

- input image size: 1024x768
- process time(CPU): 2.246 sec (Core i5-5257U)
- process time(GPU): 1.728 sec (GPU GRID K520)


## Requirement
- [Tensorflow 1.0](https://github.com/tensorflow/tensorflow)
- [Pillow](https://github.com/python-pillow/Pillow)
- [Numpy](https://github.com/numpy/numpy)
- [Scipy](https://github.com/scipy/scipy)


## Prerequisite
In this implementation, the VGG model part was based on [Tensorflow VGG16 and VGG19](https://github.com/machrisaa/tensorflow-vgg). Please add this as a submodule and follow the instructions there to have vgg16 model. Make sure the name of the module in your project matches the one in line 6 of`custom_vgg16.py`.

## Train a style model
Need to train one image transformation network model per one style target.
According to the paper, the models are trained on the [Microsoft COCO dataset](http://mscoco.org/dataset/#download). 
Also, it will save the transformation model, including the trained weights, for later use (in C++) in ```pbs``` directory, while the checkpoint files would be saved in ```ckpts/<style_name>/```. 


- ```<style_name>.pb``` is saved by default. To turn off, add argument ```-pb 0```.
- To train a model from scratch.
```
python train.py -s <style_image_path> -d <training_dataset_directory> -g 0

```
- To load a pre-trained model, specify the checkpoint to load. Negative checkpoint value suggests using the latest checkpoint.
```
python train.py -s <style_image_path> -d  <training_dataset_directory> -c <checkpoint_to_load>
```

## Generate a stylized image

### Load with .pb file
```
python generate.py -i <input_image_path> -o <output_image_path> -s <style_name> -pb 1
```

### Load with checkpoint files
- By default, the latest checkpoint file is used (negative value for the checkpoint argument). 
```
python generate.py <input_image_path> -s <style_name> -o <output_image_path> -c <checkpoint_to_load>
```

## Difference between implementation and paper
- Convolution kernel size 4 instead of 3.
- Training with batchsize(n >= 2) causes unstable result.

## License
MIT

## Reference
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)

Codes written in this repository based on following nice works, thanks to the author.

- [Chainer implementation of "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"](https://github.com/yusuketomoto/chainer-fast-neuralstyle)
