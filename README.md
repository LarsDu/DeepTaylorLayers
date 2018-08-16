# DeepTaylorLayers

Lawrence Du
Written in 2016

An implementation of DeepTaylor decomposition (https://www.sciencedirect.com/science/article/pii/S0031320316303582)
using Tensorflow. 

Useful for creating heatmaps of Convolutional Neural Networks (CNN) for visualizing the strengths of certain pixels in making a classification decision.

Example of usage is under demos/mnist_example1.py

#### To train:
```
cd demos
python mnist_example1.py --mode=train
```


#### To visualize results:
```
python mnist_example1.py --mode=visualize

```


![Example image 5](https://raw.github.com/LarsDu/DeepTaylorLayers/master/img/ex5.png)
![Example image 9](https://raw.github.com/LarsDu/DeepTaylorLayers/master/img/ex9.png)
