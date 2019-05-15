# Simple Semantic Segmentation
This repository implements the minimal code to do semantic segmentation. 

In semantic segmentation, each pixel of an input image must be assigned to an output class. For example, in segmenting tumors in CT-scans, each pixel must be assigned to class "healthy tissue" and "pathological tissue". Other examples include street scene segmentation, where each pixel belings to "background", "pedestrian", "car, "building" and so on. 

Most applications of semantic segmentation work within a larger pipeline. Therefore, semantic segmentation might seem a complicated algorithm. However, in this repository we use about 300-400 lines of code to show the core of the segmentation.

# Model
Each pixel must be assigned a label. Therefore, the output size is of same order as the input size. Say the input is __Height x Width x Channels__, then the output is __Height x Width x Classes__. Usually, other problems in machine learning require only a small output. For example, classification requires only __num_classes__ output and regression requires a single number.
To deal with these large outputs, two solutions are

 * Divide the problem into __Height*Width__ small problems. Train some algorithm to classify a patch into a single label and run this algorithm over the entire input.
 * Use a neural network to connect each pixel in the input with each pixel in the output. This repository implements this approach.

# Data
We pick a small dataset in order to show the concept. In fact, we create our own dataset. Most public datasets involve images of thousands of pixels. They require much more computation power than the average person has on his/her laptop. 

Our data combines the digits from MNIST with the background of CIFAR10. This resembles the usual foreground-background segmentation task. Below are some examples from the data.

The datagenerator randomly overlays a CIFAR image and an MNIST digit. The images and the offsets are chosen from uniform distributions.

# Convolutional neural network
As we deal with images, we model with a convolutional neural network (CNN). A CNN maps with filters between layers of neurons. This is a natural choice for images. Most concepts in the natural consist of hierarchical patterns. Moreover, with CNN's we can scale to arbitrary height and widths. You can read more about that in [this paper](https://arxiv.org/abs/1411.4038).

# Code
The code consist of three scripts

  * _model.py_ describes the model using Tensorflow library
  * *load_data.py* describes the generation and sampling of the data
  * *main.py* describes the main code: instantiating the model and data generator and trains the model.

# Further reading

  * [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
  * [Chapter 9: Convolutional Neural Networks; Deep learning book](http://www.deeplearningbook.org/)
 
Note that you have to download the data yourself:
  * [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)
  * [MNIST](http://yann.lecun.com/exdb/mnist/)

# Images

Samples of the input data
![input data](https://github.com/RobRomijnders/segm/blob/master/im/input_data.png?raw=true)
![segmentation results](https://github.com/RobRomijnders/segm/blob/master/im/segm_result.png?raw=true)

# python-mnistによるMNISTデータの使用方法
    - pip install from mnist import MNIST
    - webからダウンロードしたmnistのubyteファイルを置く
        - このファイルのパスを引数としてMNIST(パス)としてデータダウンロード
    - ディレクトリ構造
        - segm
            - data
                - cifar
                    - data_...
                - mnist
                    - t10k0image...
# Graph Convolution層の実装参考
    - https://netres-bigdata.hatenablog.com/entry/2019/01/04/154554
        - 100ノードを2値分類する程度の小規模ネットワーク

# 実装進行表
    - 画像からgrid行列を作成->OK
        - 隣接行列とグラフラプラシアン生成
    - sparse行列へ変換
    - graph convolution部を実装
    - 性能検証