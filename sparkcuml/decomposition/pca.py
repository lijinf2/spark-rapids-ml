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

from typing import Any, Union

import cudf
from cuml.decomposition.pca_mg import PCAMG as CumlPCAMG
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import ArrayType, DoubleType, Row, StructField, StructType

from sparkcuml.core import (
    INIT_PARAMETERS_NAME,
    _CumlEstimator,
    _CumlModel,
    _set_pyspark_cuml_cls_param_attrs,
)


class SparkCumlPCAModel(_CumlModel):
    def __init__(
        self, mean: list[float], pc: list[list[float]], explained_variance: list[float]
    ):
        super().__init__()
        self.mean = mean
        self.pc = pc
        self.explained_variance = explained_variance

    def _transform(self, df: DataFrame) -> DataFrame:
        pass


class SparkCumlPCA(_CumlEstimator):
    """
    PCA algorithm projects high-dimensional vectors into low-dimensional vectors
    while preserving the similarity of the vectors. This class provides GPU accleration for pyspark mllib PCA.

    Examples
    --------
    >>> from sparkcuml.decomposition import SparkCumlPCA
    >>> data = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    >>> topk = 1
    >>> gpu_pca = SparkCumlPCA().setInputCol("features").setK(topk)
    >>> df = spark.SparkContext.parallelize(data).map(lambda row: (row,)).toDF(["features"])
    >>> gpu_model = gpu_pca.fit(df)
    >>> print(gpu_model.mean)
    [2.0, 2.0]
    >>> print(gpu_model.pc)
    [[0.7071067811865475, 0.7071067811865475]]
    >>> print(gpu_model.explained_variance)
    [1.9999999999999998]
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.set_params(**kwargs)

    def setK(self, value: int) -> "SparkCumlPCA":
        """
        Sets the value of `k`.
        """
        self.set_params(n_components=value)
        return self

    def setInputCol(self, value: str) -> "SparkCumlPCA":
        """
        Sets the value of `inputCol`.
        """
        self.set_params(inputCol=value)
        return self

    def setOutputCol(self, value: str) -> "SparkCumlPCA":
        """
        Sets the value of `outputCol`.
        """
        self.set_params(outputCol=value)
        return self

    def _fit_internal(self, df: list[cudf.DataFrame], **kwargs: Any) -> dict[str, Any]:
        pca_object = CumlPCAMG(
            handle=kwargs["handle"],
            output_type="cudf",
            **kwargs[INIT_PARAMETERS_NAME],
        )

        pca_object.fit(
            df,
            kwargs["numVec"],
            kwargs["dimension"],
            kwargs["partsToRanks"],
            kwargs["rank"],
            _transform=False,
        )
        cpu_mean = pca_object.mean_.to_arrow().to_pylist()
        cpu_pc = pca_object.components_.to_numpy().tolist()
        cpu_explained_variance = pca_object.explained_variance_.to_numpy().tolist()

        ret_dict = {
            "mean": [cpu_mean],
            "pc": [cpu_pc],
            "explained_variance": [cpu_explained_variance],
        }

        return ret_dict

    def _out_schema(self) -> Union[StructType, str]:
        return StructType(
            [
                StructField("mean", ArrayType(DoubleType(), False), False),
                StructField("pc", ArrayType(ArrayType(DoubleType()), False), False),
                StructField(
                    "explained_variance", ArrayType(DoubleType(), False), False
                ),
            ]
        )

    def _create_pyspark_model(self, result: Row) -> SparkCumlPCAModel:
        return SparkCumlPCAModel(
            result["mean"],
            result["pc"],
            result["explained_variance"],
        )

    @classmethod
    def _cuml_cls(cls) -> type:
        from cuml import PCA

        return PCA

    @classmethod
    def _not_supported_param(cls) -> list[str]:
        """
        For some reason, spark cuml may not support all the parameters.
        In that case, we need to explicitly exclude them.
        """
        return ["handle", "copy"]


_set_pyspark_cuml_cls_param_attrs(SparkCumlPCA, SparkCumlPCAModel)
