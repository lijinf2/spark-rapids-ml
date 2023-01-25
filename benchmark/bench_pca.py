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

import argparse
import time
from typing import Tuple

import numpy as np
import pandas as pd
from pyspark.ml.feature import PCA
from pyspark.ml.functions import array_to_vector
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

from benchmark.utils import WithSparkSession
from sparkcuml.decomposition import SparkCumlPCA


def test_pca_bench(
    spark: SparkSession,
    n_components: int,
    num_gpus: int,
    num_cpus: int,
    no_cache: bool,
    parquet_path: str,
) -> Tuple[float, float, float]:

    fit_time = None
    transform_time = None
    total_time = None
    func_start_time = time.time()

    df = spark.read.parquet(parquet_path)
    first_col = df.dtypes[0][0]
    first_col_type = df.dtypes[0][1]
    is_single_col = True if 'array' in first_col_type else False
    if is_single_col == False:
        input_cols = [c for c in df.schema.names]

    if num_gpus > 0:
        assert num_cpus <= 0
        start_time = time.time()
        if not no_cache:
            df = df.repartition(num_gpus).cache()
            df.count()
            print(f"prepare session and dataset took: {time.time() - start_time} sec")

        start_time = time.time()
        gpu_pca = (
            SparkCumlPCA(num_workers=num_gpus)
            .setK(n_components)
        )

        if is_single_col:
            gpu_pca = gpu_pca.setInputCol(first_col).setOutputCol("pca_features")
        else:
            output_cols = ["o" + str(i) for i in range(n_components)]
            gpu_pca = gpu_pca.setInputCol(input_cols).setOutputCol(output_cols)

        gpu_model = gpu_pca.fit(df)
        fit_time = time.time() - start_time
        print(f"gpu fit took: {fit_time} sec")

        start_time = time.time()
        gpu_model.transform(df).count()
        transform_time = time.time() - start_time
        print(f"gpu transform took: {transform_time} sec")

        total_time = time.time() - func_start_time
        print(f"gpu total took: {total_time} sec")

    if num_cpus > 0:
        assert num_gpus <= 0
        start_time = time.time()
        if is_single_col:
            vector_df = df.select(array_to_vector(df[first_col]).alias(first_col))
        else:
            vector_assembler = VectorAssembler(outputCol="features").setInputCols(input_cols)
            vector_df = vector_assembler.transform(df)
            first_col = "features"


        if not no_cache:
            vector_df = vector_df.cache()
            vector_df.count()
            print(f"prepare session and dataset: {time.time() - start_time} sec")

        start_time = time.time()
        cpu_pca = PCA().setK(n_components)
        cpu_pca = cpu_pca.setInputCol(first_col).setOutputCol("pca_features")

        cpu_model = cpu_pca.fit(vector_df)
        fit_time = time.time() - start_time
        print(f"cpu fit took: {fit_time} sec")

        start_time = time.time()
        cpu_model.transform(vector_df).count()
        transform_time = time.time() - start_time
        print(f"cpu transform took: {transform_time} sec")

        total_time = time.time() - func_start_time
        print(f"cpu total took: {total_time} sec")

    return (fit_time, transform_time, total_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_components", type=int, default=3)
    parser.add_argument("--num_gpus", type=int, default=1, help='number of available GPUs. If num_gpus > 0, sparkcuml will run with the number of dataset partitions equal to num_gpus.')
    parser.add_argument("--num_cpus", type=int, default=6, help='number of available CPUs. If num_cpus > 0, spark will run and with the number of dataset partitions to num_cpus.')
    parser.add_argument("--no_cache", action='store_true', default=False, help='whether to enable dataframe repartition, cache and cout outside sparkcuml fit')
    parser.add_argument("--num_runs", type=int, default=2, help='set the number of repetitions for cold/warm runs')
    parser.add_argument("--report_path", type=str, default="")
    parser.add_argument("--parquet_path", type=str, default="")
    parser.add_argument("--spark_confs", action="append", default=[])
    args = parser.parse_args()

    report_pd = pd.DataFrame()

    with WithSparkSession(args.spark_confs) as spark:
        for run_id in range(args.num_runs):
            (fit_time, transform_time, total_time) = test_pca_bench(
                spark,
                args.n_components,
                args.num_gpus,
                args.num_cpus,
                args.no_cache,
                args.parquet_path,
            )

            report_dict = {
                "run_id": run_id, 
                "fit": fit_time,
                "transform": transform_time,
                "total": total_time,
                "n_components": args.n_components,
                "num_gpus": args.num_gpus,
                "num_cpus": args.num_cpus,
                "no_cache": args.no_cache,
                "parquet_path": args.parquet_path,
            }

            for sconf in args.spark_confs:
                key, value = sconf.split("=")
                report_dict[key] = value

            alg_name = 'sparkcuml_pca' if args.num_gpus > 0 else 'spark_pca'
            pdf = pd.DataFrame(
                data = {k : [v] for k, v in report_dict.items()},
                index = [alg_name]
            )
            print(pdf)
            report_pd = pd.concat([report_pd, pdf])

    print(f"\nsummary of the total {args.num_runs} runs:\n")
    print(report_pd)
    if args.report_path != "":
        report_pd.to_csv(args.report_path, mode="a")
        report_pd.to_csv(args.report_path, mode="a")
