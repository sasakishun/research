from ..segmentation import segmentation_adjacency
from ..graph import coarsen_adj, perm_features


def preprocess_pipeline(image,
                        segmentation_algorithm,
                        feature_extraction_algorithm,
                        levels,
                        connectivity=4,
                        scale_invariance=False,
                        stddev=1):

    segmentation = segmentation_algorithm(image)
    adj, points, mass = segmentation_adjacency(segmentation, connectivity)
    # print("adj:{}".format(adj))
    # print("adj.shape:{}".format(adj.shape))
    # 各バッチごとにSparseTensorを生成、命名規則は
    # adj:SparseTensor(indices=Tensor("adj_dist_4_64/indices:0", shape=(?, ?), dtype=int64),
    #  values=Tensor("adj_dist_4_64/values:0", shape=(?,), dtype=float32),
    #  dense_shape=Tensor("adj_dist_4_64/shape:0", shape=(?,), dtype=int64))

    features = feature_extraction_algorithm(segmentation, image)

    adjs_dist, adjs_rad, perm = coarsen_adj(adj, points, mass, levels,
                                            scale_invariance, stddev)

    features = perm_features(features, perm)

    return features, adjs_dist, adjs_rad


def preprocess_pipeline_fixed(segmentation_algorithm,
                              feature_extraction_algorithm,
                              levels,
                              connectivity=4,
                              scale_invariance=False,
                              stddev=1):
    def _preprocess(image):
        return preprocess_pipeline(image, segmentation_algorithm,
                                   feature_extraction_algorithm, levels,
                                   connectivity, scale_invariance, stddev)

    return _preprocess
