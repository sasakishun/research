
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
  - コントラスティブダイバージェンス（Siamese Networkで使用、ユークリッド距離の代わり）
    - マージンを設定することで、特徴ベクトル間距離が完全にゼロになり、同一クラスが一点に集中するよう学習されるのを防ぎ、ある程度集合すれば誤差0となるようにする。これにより過適合を低減。このように一点に集中することは避ける、という概念は色々な手法で見られる。

勉強すること
  - seq2seq
  - Attention（soft, hard）
  - q-learning
  - Variational Auto Encoder
  - GANの応用
  - FaceNets
    - 入力画像を128次元の特徴ベクトルに変換、このベクトル間のユークリッド距離をそのまま類似度と定義
    - triplet lossを使用
      - negative : ランダムに選定したサンプルの中でhard negativeを使用（一番距離が近いnegativeのみを見て、それが離れるように学習）
      - positive : 全て使用
      - 解説記事（https://qiita.com/tancoro/items/35d0925de74f21bfff14 ）
        - Verification方法は、学習時の入力サンプル組でSotmaxLossタイプ(1サンプル)、Siamese Network(2枚1組)、TripletLoss(3枚1組)、Quadruplet Loss(4枚1組)などに分類できる
  - ResNet(3,40層などの少数層でどれだけ性能が向上するか検証)
  - batch normalization(inference)
  - 異常検知で間違い方によりロスの重みづけ、という方法があるか？ないならどうやってnegative Falseを避けているのか調べる
  - siamese and gan (https://aws.amazon.com/jp/blogs/news/combining-deep-learning-networks-gan-and-siamese-to-generate-high-quality-life-like-images/ )
    - GANのGeneratoerとDiscremenaterをSiameseNetにして、2つの出力[(True or False), (True or False)]を得る。この出力差をSiamameseで学習
    - siameseは一般的なものを使用
  - Discriminative Learning of Deep Convolutional Feature Point Descriptors
    - siamese networkには識別困難なペアから学習すべきという、カリキュラムラーニングと反対の考え方
  - GANの異常検知まとめ(http://habakan6.hatenablog.com/entry/2018/04/29/013200 )
  - 強化学習（https://qiita.com/sugulu/items/3c7d6cbe600d455e853b ）
    - q-learning（実際に実装して性能検証してみる）
      - dqn (https://lp-tech.net/articles/DYD3x )
  - Graph NN(https://qiita.com/shionhonda/items/d27b8f13f7e9232a4ae5 )
  - segmentation U-Net(https://qiita.com/tktktks10/items/0f551aea27d2f62ef708 )
  - sci-kit learnのsegmentation方法(skimage.segmentation...)を調べること(2019/4/5)
  - RBF
    - 訓練データをランダムに抽出しセントロイドとする（訓練データの代わりに"バイアス"として学習してもいい）
    - 下位層からの"重み"をスケールとする合成ガウス分布を算出
    - 入力がはこの合成ガウス分布上にプロットされる
      - セントロイドが多いほど任意の関数を実現できるため、表現力が上がる
    
# 環境構築
  - Windows10でのtensorflow-gpuの使用するために（https://ossyaritoori.hatenablog.com/entry/2018/03/27/Keras_%26_Tensorflow_%28GPU%E6%9C%89%29%E3%81%AE%E7%92%B0%E5%A2%83%E6%A7%8B%E7%AF%89_on_Windows_with_Anaconda ）  
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
  - Graph Convolution まとめ2019(https://arxiv.org/pdf/1901.00596.pdf )
  - 2画像を2グラフに変換→和→GCN→各ピクセルごとのクラスラベル(セグメンテーション)→(https://www.ee.iitb.ac.in/course/~avik/ICVGIP18_arXiv.pdf )
      - 2グラフの和を取る(pairwiseCNNなど)よりseamese network形式の方が一般的にいいため、ここでも和でなくsiamse形式に変更してみるといい可能性
  - 条件付き確率場をどのタイミングで使うのか？連続値なのか（勾配計算は可能か）を調べること
  
# 深く勉強すべき
- Siamese Network (https://qiita.com/TatsuyaMizuguchi17/items/f6ef9d7884b4cf4b364e )
- Siamese triplet loss (https://github.com/adambielski/siamese-triplet/blob/master/README.md )
- AnoGAN (https://qiita.com/NakaokaRei/items/231ec4efe42dfe79d1ff )
- GANで異常検知（http://ni4muraano.hatenablog.com/entry/2018/08/14/174901 ）
- 画像を周波数分解してCNNに入力、チャネル数を増やしたことでどのような結果になるか？
- FaceNet
- pairwise cnn
# 便利メモ
  - tqdmを使うとデータの処理状況(itarationできるもの)がわかる
  - numpyのconcatenateではaxisを指定することで、任意の要素を連結可能
  - 疎行列の使用方法(http://nktmemoja.github.io/ml/2016/12/29/tensorflow-sparse.html )
    - tf.sparse.add(A, B) などを使うには、引数A,Bが両方ともsparseでなくてはならない
      - 片方がdenseなら、sparse行列をdenseにしてからtf.add()などを使用する (sparseに統一してもいいが、indice,value,shapeを指定する必要)
  - tensorflowにおいて任意の活性化関数の使用方法(https://stackoverflow.com/questions/39921607/how-to-make-a-custom-activation-function-with-only-python-in-tensorflow )
    - 活性化関数act()と勾配d_act()を設定
      - act()はnumpy関数に変換する
      - act()とd_act()をtensorflow側(tf.py_funcをオーバーライドした関数py_func(活性化関数, 入力仮引数, 出力型([tf.float32など]), stateful=stateful, 名前, 勾配関数)を使用)に登録→これをやって初めて計算グラフに乗る
  - GANの学習がうまくいかないとき（https://gangango.com/2018/11/16/post-322/ ）
    - 学習回数比率を「生成器 : 識別器 = 2 : 1」などに偏らせる
  - kerasで複数の出力を取得し、それらを同時に最適化する方法（https://codeday.me/jp/qa/20190409/586386.html ） 
  - kerasで複数の出力に、別々の最適化法を適用する方法（https://www.st-hakky-blog.com/entry/2017/12/07/173928 ）
# いますること
  - Graph ConvolutionでMNISTの分類、実世界画像でセグメンテーション
    - segmenationではsiamese networkを参考に2画像同時入力でより高精度化を期待
    - 局所的な繋がり（隣接ピクセル）を考慮できるためsegmentationに適しているかもしれない
      - CRF(Conditional Random Field)(条件付き確率場後処理)を実装して結果観察
    - Segmentationタスクは分野的に大きな発展が困難な可能性があり、別領域（データセット、タスク）での応用も考えること
      — データセットを自作するようなタスク
        - 知識グラフの「A is B」関係の推論（「A=B」関係のみで繋がるようクラスタリングすれば上位概念のクラスタが抽出可能かもしれない）
          - 知識グラフをGraph Convolutionする手法 (https://medium.com/programming-soda/graph-convolution%E3%82%92%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E3%81%AB%E5%BF%9C%E7%94%A8%E3%81%99%E3%82%8B-part1-b792d53c4c18)
  - GANで手書き文字を自動生成、その際の潜在変数により書き手の類似度を出力する
    - Generator部分では、生成画像がストロークの集合になっているかの制約も加える
      - ストロークを20つくらいに仮定し、拡大縮小、回転、位置を変えることで文字生成（20画以下の場合は不要分を画面外に位置が移るようにすることで対処）
      - 回転行列や拡大行列、移動行列を定義しそれと行列積を取ればBP可能（回転行列などでは回転角θを共有するため、ある種の重み共有かもしれない）
    - 手書き文字でなく、絵画の集合を用いれば画家（イラストレータの）の類似度計算もできる
  - RBFを用いつつ汎化性を上げられれば、学習が高速という利点がある
    - 汎化性を上げるために、最初はRBFにより近似するが、徐々にReLUに近づけていくといいかもしれない
    - RBF解説記事(https://towardsdatascience.com/radial-basis-functions-neural-networks-all-we-need-to-know-9a88cc053448 )
      - 通常のNNと分離平面が異なる（RBFは分離平面を用いない？→要確認20190424）
      - RBFがとてつもなく遅い理由を調査すべき
        - そもそも速い理由がExtreme Learning Machineと同様に3層しかないからと考えられる
      - 現在のNNはReLUを使うため線形性が高すぎてAdversarial Exampleに弱いため、RBFで非線形性を学習するといい可能性
  - Graph Convolutionのsegmentation結果の可視化
    - 疎行列の3次元以上の表現方法を調査
    - tensorの要素指定方法（tfarray[:10][:10]のようにする方法）を調査
      - これによりCNNのように局所グラフを畳み込む
    - GANによる蒸留
      - 蒸留の性能検証のためには、表現力が足らないモデルで実験する（過学習を解決すべき問題としない）
  - 知識抽出（多層NNを用いて）
    - 1990年代のアイデアを調査(20190521)
    - 知識抽出が難しいのは、すべての場合を網羅するための計算量が膨大になるから（MNISTでもO(2^784)）
      - まず簡単な人口データセットを作成しルール抽出を行う
        - 実データではきれいに知識抽出されない（知識が複雑すぎる）のは当たり前なので、きれいに抽出できるもので確かめる
      - 入力画像内ピクセルの取捨選択にはkeras.layers.Multiply()([input, filter])を使う
        - filter = dense(100)([1])などとすればいい、ダメならInput()で常に1を受け取る
    -GANのスタイル変換を用いつつ分類精度を保つには、MSEとCrossEntropyの更新比率を色々試してみるしかない
      - もしくはそれぞれのロスでの更新回数の比率を偏らせる（GANの生成器と識別器の更新回数比率を2:1にするような）
    - NN内の各ノード活性率をメモリネットワークに保存
      - メモリネットワーク(http://deeplearning.hatenablog.com/entry/memory_networks )
        - まず基本的なモデルを理解・実装し、中間層の出力（1つのベクトル）をメモリに格納する手法を試してみる
          - 全ノードの状態（スカラー）をノードごとにメモリに格納するのか、層ごとに1つのベクトルとして格納するのか
          - メモリに格納するentenceをどうするのか（通常の学習時の中間層出力 or 1-hotのスタイル画像）
    - メモリネットワークもGANも使わずに分析から入るべき
      - 中間層出力と重みの可視化
      - プルーニングの実装
        - https://github.com/BenWhetton/keras-surgeon
      - 量子化の実装
      - 条件ごとに重み保存、一括読み込み
        - Model.save(パス)はパスで指定したファイルに重みを保存するだけ
            - つまり重みを共有していても、Model.load(パス)するまで、共有重みは更新されない
            - 重要重みのみ保存して読み込み、Heの法則を正しく適用
    - NN×木構造
      - 1ノードづつ層としてkerasで記述後、concatenateすれば良さそう
        - ノード数1の出力が認められるのか調査
