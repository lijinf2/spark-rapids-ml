#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Any, Callable, Dict, List, Union

import cudf
import pandas as pd
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    Row,
    StructField,
    StructType,
)

from sparkcuml.core import (
    INIT_PARAMETERS_NAME,
    _CumlEstimator,
    _CumlModel,
    _set_pyspark_cuml_cls_param_attrs,
)


class SparkCumlKMeans(_CumlEstimator):
    """
    KMeans algorithm partitions data points into a fixed number (denoted as k) of clusters.
    The algorithm initializes a set of k random centers then runs in iterations.
    In each iteration, KMeans assigns every point to its nearest center,
    then calculates a new set of k centers.

    Examples
    --------
    >>> from sparkcuml.cluster import SparkCumlKMeans
    TODO
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.set_params(**kwargs)

    def setK(self, value: int) -> "SparkCumlKMeans":
        """
        Sets the value of `n_clusters`.
        """
        self.set_params(n_clusters=value)
        return self

    def setFeaturesCol(self, value: str) -> "SparkCumlKMeans":
        """
        Sets the value of `inputCol`.
        """
        self.set_params(inputCol=value)
        return self

    def setPredictionCol(self, value: str) -> "SparkCumlKMeans":
        """
        Sets the value of `outputCol`.
        """
        self.set_params(outputCol=value)
        return self

    def setMaxIter(self, value: int) -> "SparkCumlKMeans":
        """
        Sets the value of `max_iter`.
        """
        self.set_params(max_iter=value)
        return self

    def _get_cuml_fit_func(
        self, dataset: DataFrame
    ) -> Callable[[List[cudf.DataFrame], Dict[str, Any]], Dict[str, Any]]:
        def _cuml_fit(
            df: List[cudf.DataFrame], params: Dict[str, Any]
        ) -> Dict[str, Any]:
            from cuml.cluster.kmeans_mg import KMeansMG as CumlKMeansMG

            kmeans_object = CumlKMeansMG(
                handle=params["handle"],
                output_type="cudf",
                **params[INIT_PARAMETERS_NAME],
            )

            concated = cudf.concat(df)
            kmeans_object.fit(
                concated,
                sample_weight=None,
            )

            return {
                "cluster_centers_": [
                    kmeans_object.cluster_centers_.to_numpy().tolist()
                ],
            }

        return _cuml_fit

    def _out_schema(self) -> Union[StructType, str]:
        return StructType(
            [
                StructField(
                    "cluster_centers_", ArrayType(ArrayType(DoubleType()), False), False
                ),
            ]
        )

    def _create_pyspark_model(self, result: Row) -> "SparkCumlKMeansModel":
        return SparkCumlKMeansModel.from_row(result)

    @classmethod
    def _cuml_cls(cls) -> type:
        from cuml import KMeans

        return KMeans

    @classmethod
    def _not_supported_param(cls) -> List[str]:
        """
        For some reason, spark cuml may not support all the parameters.
        In that case, we need to explicitly exclude them.
        """
        return [
            "handle",
            "output_type",
        ]


class SparkCumlKMeansModel(_CumlModel):
    def __init__(
        self,
        cluster_centers_: List[List[float]],
    ):
        super().__init__()

        self.cluster_centers_ = cluster_centers_

        cumlParams = SparkCumlKMeans._get_cuml_params_default()
        self.set_params(**cumlParams)

    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        ret_schema = StructType(
            [StructField(self.getOutputCol(), IntegerType(), False)]
        )
        return ret_schema

    def _get_cuml_transform_func(
        self, dataset: DataFrame
    ) -> Callable[[cudf.DataFrame], pd.DataFrame]:

        cuml_alg_params = {}
        for k in SparkCumlKMeans._get_cuml_params_default():
            cuml_alg_params[k] = self.getOrDefault(k)

        cluster_centers_ = self.cluster_centers_
        output_col = self.getOutputCol()

        def _transform_internal(df: cudf.DataFrame) -> pd.DataFrame:
            from cuml.cluster.kmeans_mg import KMeansMG as CumlKMeansMG

            kmeans_object = CumlKMeansMG(**cuml_alg_params)
            kmeans_object.dtype = df.dtypes[0]
            kmeans_object.n_cols = len(df.columns)

            from sparkcuml.utils import cudf_to_cuml_array

            kmeans_object.cluster_centers_ = cudf_to_cuml_array(
                cudf.DataFrame(cluster_centers_), order="C"
            )

            res = list(kmeans_object.predict(df, normalize_weights=False).to_numpy())
            return pd.DataFrame({output_col: res})

        return _transform_internal

    def setFeaturesCol(self, value: str) -> "SparkCumlKMeansModel":
        """
        Sets the value of `inputCol`.
        """
        self.set_params(inputCol=value)
        return self

    def setPredictionCol(self, value: str) -> "SparkCumlKMeansModel":
        """
        Sets the value of `outputCol`.
        """
        self.set_params(outputCol=value)
        return self

    def getFeaturesCol(self) -> str:
        """
        Gets the value of `inputCol` or its default value.
        """
        return self.getOrDefault(self.inputCol)

    def getPredictionCol(self) -> str:
        """
        Gets the value of `outputCol` or its default value.
        """
        return self.getOrDefault(self.outputCol)


_set_pyspark_cuml_cls_param_attrs(SparkCumlKMeans, SparkCumlKMeansModel)
