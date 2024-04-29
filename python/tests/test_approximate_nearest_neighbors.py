from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
from _pytest.logging import LogCaptureFixture
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import Row
from sklearn.datasets import make_blobs

from spark_rapids_ml.core import alias
from spark_rapids_ml.knn import (
    ApproximateNearestNeighbors,
    ApproximateNearestNeighborsModel,
)
from spark_rapids_ml.utils import _concat_and_free

from .sparksession import CleanSparkSession
from .test_nearest_neighbors import (
    NNEstimator,
    NNModel,
    func_test_example_no_id,
    func_test_example_with_id,
    reconstruct_knn_df,
)
from .utils import (
    array_equal,
    create_pyspark_dataframe,
    get_default_cuml_parameters,
    idfn,
    pyspark_supported_feature_types,
)


def test_default_cuml_params() -> None:
    from cuml import NearestNeighbors as CumlNearestNeighbors

    cuml_params = get_default_cuml_parameters(
        [CumlNearestNeighbors],
        [
            "handle",
            "metric",
            "p",
            "algo_params",
            "metric_expanded",
            "metric_params",
            "output_type",
        ],
    )

    spark_params = ApproximateNearestNeighbors()._get_cuml_params_default()
    cuml_params["algorithm"] = "ivfflat"  # change cuml default 'auto' to 'ivfflat'
    assert cuml_params == spark_params


@pytest.mark.parametrize(
    "algo_and_params",
    [("ivfflat", {"nlist": 1, "nprobe": 2})],
)
@pytest.mark.parametrize(
    "func_test",
    [func_test_example_no_id, func_test_example_with_id],
)
def test_example(
    algo_and_params: Tuple[str, Optional[dict[str, Any]]],
    func_test: Callable[[NNEstimator, str], Tuple[NNEstimator, NNModel]],
    gpu_number: int,
    tmp_path: str,
) -> None:
    algorithm = algo_and_params[0]
    algo_params = algo_and_params[1]

    gpu_knn = ApproximateNearestNeighbors(algorithm=algorithm, algo_params=algo_params)
    gpu_knn, gpu_model = func_test(tmp_path, gpu_knn)  # type: ignore

    for obj in [gpu_knn, gpu_model]:
        assert obj._cuml_params["algorithm"] == algorithm
        assert obj._cuml_params["algo_params"] == algo_params


@pytest.mark.parametrize(
    "combo",
    [
        ("array", 10000, None),
        ("vector", 2000, {"nlist": 10, "nprobe": 2}),
        ("multi_cols", 5000, {"nlist": 20, "nprobe": 4}),
    ],
)  # vector feature type will be converted to float32 to be compatible with cuml single-GPU NearestNeighbors Class
@pytest.mark.parametrize("data_shape", [(10000, 50)], ids=idfn)
@pytest.mark.parametrize("data_type", [np.float32])
def test_ivfflat(
    combo: Tuple[str, int, Optional[Dict[str, Any]]],
    data_shape: Tuple[int, int],
    data_type: np.dtype,
) -> None:

    feature_type = combo[0]
    max_record_batch = combo[1]
    algo_params = combo[2]
    n_neighbors = 50
    n_clusters = 10
    tolerance = 1e-4

    X, _ = make_blobs(
        n_samples=data_shape[0],
        n_features=data_shape[1],
        centers=n_clusters,
        random_state=0,
    )  # make_blobs creates a random dataset of isotropic gaussian blobs.

    # set average norm sq to be 1 to allow comparisons with default error thresholds
    # below
    root_ave_norm_sq = np.sqrt(np.average(np.linalg.norm(X, ord=2, axis=1) ** 2))
    X = X / root_ave_norm_sq

    # obtain exact knn cuml results
    from cuml import NearestNeighbors as cuNN

    cuml_knn = cuNN(algorithm="brute", n_neighbors=n_neighbors, output_type="numpy")
    cuml_knn.fit(X)
    distances_exact, indices_exact = cuml_knn.kneighbors(X)

    def cal_avg_recall(indices_ann: np.ndarray) -> float:
        assert indices_ann.shape == indices_exact.shape
        assert indices_ann.shape == (len(X), n_neighbors)
        retrievals = [np.intersect1d(a, b) for a, b in zip(indices_ann, indices_exact)]
        recalls = np.array([len(nns) / n_neighbors for nns in retrievals])
        return recalls.mean()

    def cal_avg_dist_gap(distances_ann: np.ndarray) -> float:
        assert distances_ann.shape == distances_exact.shape
        assert distances_ann.shape == (len(X), n_neighbors)
        gaps = np.abs(distances_ann - distances_exact)
        return gaps.mean()

    y = np.arange(len(X))  # use label column as id column

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    with CleanSparkSession(conf) as spark:
        data_df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X, y
        )
        assert label_col is not None
        data_df = data_df.withColumn(label_col, col(label_col).cast("long"))
        id_col = label_col

        knn_est = (
            ApproximateNearestNeighbors(
                algorithm="ivfflat", algo_params=algo_params, k=n_neighbors
            )
            .setInputCol(features_col)
            .setIdCol(id_col)
        )

        # test kneighbors: obtain spark results
        knn_model = knn_est.fit(data_df)
        query_df = data_df
        (item_df_withid, query_df_withid, knn_df) = knn_model.kneighbors(query_df)

        knn_df = knn_df.sort(f"query_{id_col}")
        knn_df_collect = knn_df.collect()

        # test kneighbors: collect spark results for comparison with cuml results
        distances = np.array([r["distances"] for r in knn_df_collect])
        indices = np.array([r["indices"] for r in knn_df_collect])

        # test kneighbors: compare top-1 nn indices(self) and distances(self)

        self_index = [knn[0] for knn in indices]
        assert np.all(self_index == y)

        self_distance = [dist[0] for dist in distances]
        assert self_distance == [0.0] * len(X)

        # test kneighbors: compare with cuml ANN on avg_recall and dist
        cuml_ivfflat = cuNN(
            algorithm="ivfflat", algo_params=algo_params, n_neighbors=n_neighbors
        )
        cuml_ivfflat.fit(X)
        distances_cumlann, indices_cumlann = cuml_ivfflat.kneighbors(X)
        distances_cumlann **= 2  # square up to get l2 distances

        avg_recall_cumlann = cal_avg_recall(indices_cumlann)
        avg_recall = cal_avg_recall(indices)
        assert abs(avg_recall - avg_recall_cumlann) < tolerance

        avg_dist_gap_cumlann = cal_avg_dist_gap(distances_cumlann)
        avg_dist_gap = cal_avg_dist_gap(distances)
        assert abs(avg_dist_gap - avg_dist_gap_cumlann) < tolerance

        # test exactNearestNeighborsJoin
        knnjoin_df = knn_model.approxSimilarityJoin(query_df_withid)
        reconstructed_knn_df = reconstruct_knn_df(
            knnjoin_df, row_identifier_col=knn_model.getIdCol()
        )
        reconstructed_collect = reconstructed_knn_df.collect()

        def get_sorted_indices(row: Row) -> list[int]:
            row_dists = row["distances"]
            row_indices = row["indices"]
            idx_sorted = sorted(
                range(len(row_indices)), key=lambda i: (row_dists[i], row_indices[i])
            )
            indices_sorted = [row["indices"][idx] for idx in idx_sorted]
            return indices_sorted

        def assert_row_equal(r1: Row, r2: Row) -> None:
            assert r1[f"query_{id_col}"] == r2[f"query_{id_col}"]
            r1_distances = r1["distances"]
            r2_distances = r2["distances"]
            assert r1_distances == r2_distances

            # sort indices in case two neighbors having same distance to the query
            r1_indices_sorted = get_sorted_indices(r1)
            r2_indices_sorted = get_sorted_indices(r2)
            assert r1_indices_sorted == r2_indices_sorted

        assert len(reconstructed_collect) == len(knn_df_collect)
        for i in range(len(reconstructed_collect)):
            r1 = reconstructed_collect[i]
            r2 = knn_df_collect[i]
            assert_row_equal(r1, r2)


@pytest.mark.parametrize("data_shape", [(1_000_000, 3)], ids=idfn)
def test_large_broadcast(
    gpu_number,
    data_shape: Tuple[int, int],
) -> None:
    n_samples = data_shape[0]
    n_features = data_shape[1]

    conf = {
        "spark.master": f"local[{gpu_number}]",
        "spark.python.worker.reuse": "false",
        "spark.driver.host": "127.0.0.1",
        "spark.driver.memory": "100g",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.driver.maxResultSize": "1m",
    }

    from .conftest import _get_spark, _spark

    _spark.stop()

    from pyspark.sql import SparkSession

    builder = SparkSession.builder
    for key, value in conf.items():
        builder.config(key, value)
    spark = builder.getOrCreate()

    def py_func(id):
        return np.random.random(n_features).tolist()

    pyspark_func = udf(lambda x: py_func(x), "array<float>")

    id_col = "id"
    features_col = "features"

    data_df = spark.range(n_samples).withColumn("features", pyspark_func("id"))

    print(f"debug numPartitions: {data_df.rdd.getNumPartitions()}")

    # begin
    def count_rows_in_partition(index, iterator):
        count = sum(1 for _ in iterator)  # summing the number of rows in the iterator
        return [(index, count)]

    # Apply the function to the DataFrame's RDD
    partition_counts = data_df.rdd.mapPartitionsWithIndex(
        count_rows_in_partition
    ).collect()

    # Print the number of rows in each partition
    for partition_id, count in partition_counts:
        print(f"Partition {partition_id} has {count} rows")
    # end

    knn_est = (
        ApproximateNearestNeighbors(algorithm="ivfflat")
        .setInputCol(features_col)
        .setIdCol(id_col)
    )

    knn_model = knn_est.fit(data_df)
    bcast_qids, bcast_qfeatures = knn_model._broadcast_as_nparray(
        data_df, BROADCAST_LIMIT=2 << 30
    )

    qid_all = _concat_and_free([chunk.value for chunk in bcast_qids])
    qfeatures_all = _concat_and_free([chunk.value for chunk in bcast_qfeatures])

    assert len(qid_all) == n_samples
    assert qfeatures_all.shape == (n_samples, n_features)

    _spark = _get_spark()
