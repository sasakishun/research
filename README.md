# research
研究開発用レポジトリ（2017/10~）
  - データセットなどのまとめサイト（https://github.com/arXivTimes/arXivTimes ）

研究テーマ（2018/10-）
  - graph neural networksやgraph convolutional neural networksなどのグラフ構造分野
    - graph convolution の実装法(https://omedstu.jimdo.com/2018/05/13/keras%E3%81%AB%E3%82%88%E3%82%8Bgraph-convolutional-networks/ )
    - graph convolutional networksとは（https://www.slideshare.net/KCSKeioComputerSocie/graph-convolutional-network ）
    - graoh neural networksの動画（https://www.youtube.com/watch?v=cWIeTMklzNg ）
    - graph neural networksの日本語解説（https://www.slideshare.net/DeepLearningJP2016/dlrelational-inductive-biases-deep-learning-and-graph-networks-104442091 ）
    - graph neural networkの元論文（http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1015.7227&rep=rep1&type=pdf ）
    - ラプラシアン=微分演算子
    
    - graph convolutionのgithubコード（https://github.com/tkipf/gcn ）
    - Graph Convolutional Networks for Classification with a Structured LebalSpace (https://www.slideshare.net/KazusaTaketoshi/graph-convolutional-networks-for-classification-with-a-structured-lebalspace )
    - Graph フーリエ変換(The Emerging Field of Signal Processing on Graphs: Extending High-Dimensional Data Analysis to Networks and Other Irregular Domains )（https://arxiv.org/pdf/1211.0053.pdf）
    
勉強したこと
  - ホップフィールドネットワーク(10/21)
  - ボルツマンマシン(10/21)
  - Graph Convolution(Graphフーリエ変換使用、"The Emerging Field of Signal Processing on Graphs") (10/23) 
  - Graph Convolution(直観的分類、“Modeling Relational Data with Graph Convolutional Networks”)(10/23)
  - Generative Adversarial Networks（10/29）

勉強すること
  - seq2seq
  - Attention
  - q-learning
  - コントラスティブダイバージェンス
  - Variational Auto Encoder
  - GANの応用
  - FaceNetsの実装
  
# 読むリスト
  - Neural Message Passing for Quantum Chemistry(https://arxiv.org/pdf/1704.01212.pdf )
    - message passing algorithmというものを用いる。入力は隣接行列とノードの持つ信号。
  - The Emerging Field of Signal Processing on Graphs: Extending High-Dimensional Data Analysis to Networks and Other Irregular Domains(https://arxiv.org/pdf/1211.0053.pdf)
    - graph spectral domainsについての議論、グラフから高次元データを抽出する方法の議論とサーベイ
  - Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering(https://arxiv.org/pdf/1606.09375.pdf )
    - 画像や動画などの低次元グリッドなどから、ソーシャルネットや単語埋め込みベクトルなどの高次元ネットワークを抽出する手法
  - Semi-Supervised Classification with Graph Convolutional Networks(https://arxiv.org/pdf/1609.02907.pdf )
    - グラフ畳み込みをスケーラブルにした半教師ありクラス分類
  - Modeling Relational Data with Graph Convolutional Networks(https://arxiv.org/pdf/1703.06103.pdf )
    - R-GCNsを用いて不完全な知識グラフ（Wikipedia）を補完する。ここではエンティティのクラス分類とエンティティ間の関係推論を行う。
    この手法では関係ごとに異なる重みで周辺ノードを畳み込み、各ノードごとに特徴を出力する。その特徴を次の層のノード値とする。
    このように畳み込みの結果、隣接ノード値を考慮したノード値が次の層へ出力される。畳み込みをしても層の前後でグラフの形は変化しない。あくまで隣接ノードを考慮した値にノードが更新されていくだけである。そして畳み込みをするたびに隣接ノードが考慮されていくが、その隣接ノードは別の隣接ノードを考慮しているので、間接的に隣の隣のノードも考慮できる。このように畳み込みをするごとに（畳み込み回数）近傍ノードを考慮したノード値が出力される。
  - Structured Sequence Modeling with Graph Convolutional Recurrent Networks(https://arxiv.org/pdf/1612.07659.pdf )
    - GraphCNNとRNNの組み合わせ手法。GraphCNNでデータ構造を抽出し、RNNでその動的なパターン変化を学習。
    Moving MNISTと自然言語生成の2タスクで実験。
  - Dynamic Graph Convolutional Networks(https://arxiv.org/pdf/1704.06199.pdf )
    - 1つのモデルで異なるクラス分類タスクに対応するには、
    グラフなどで表現される構造情報を取り扱う必要がある。
    しかし既存のNNでは動的に変化するグラフ構造を取り扱うことが出来ない。これに対しLSTMとGraphCNNを組み合わせることで、グラフが更新されても対応可能な手法を提案する。
  - Encoding Sentences with Graph Convolutional Networks for Semantic Role Labeling(https://arxiv.org/pdf/1703.04826.pdf )
    - Semantic role labeling(SRL)は主語述語関係を識別するタスクでNLPでは重要視される。
    著者は意味だけでなく構文情報も重要になると考え、意味を抽出するLSTMに構文情報を処理できるGraphCNNを組み合わせる。
    （構文情報とは何の事なのか？）→主語述語関係などのルールベースのような対応関係
  - Representation Learning: A Review and New Perspectives(https://arxiv.org/pdf/1206.5538.pdf )    

# 深く勉強中
- Siamese Network https://qiita.com/TatsuyaMizuguchi17/items/f6ef9d7884b4cf4b364e
