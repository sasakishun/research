1. ループや多重エッジを持たない重み付き無向グラフに対してはGraph Fourier変換、
2. Graph Fourier変換後，Convolution Theoremを適用
3. うまくパラメータ付けすることで、ノードの隣接関係を考慮したGraph Convolutionも定義可能

Graph Fourier変換とは、
グラフ上の信号（ノードが持つベクター値）に対して定義される,Fourier変換に似た操作．
Fourier変換は波形信号を周波数成分ごとに成分分解する変換、
Graph Fourier変換はグラフ上の信号を「ゆるやかな信号」や「急峻な信号」へ成分分解する変換。

Convolution Theoremとは，
ConvolutionはFourier Domainにおける要素積に相当するという定理のこと


1.　グラフ上の信号に対してGraph Fourier変換
2.　1に対して何か(Fourier領域において表現されたConvolution)と要素積をとる
3.　2に対して逆Graph Fourier変換を施す

