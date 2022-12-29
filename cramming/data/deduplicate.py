"""This is glue code to connect to the rust-based deduplication of https://github.com/google-research/deduplicate-text-datasets
there is probably a smart way to implement deduplication for huggingface datasets directly,
but this is just a dumb dump-everything-into-tmp-files solution.

Code based on branch https://github.com/google-research/deduplicate-text-datasets/tree/dev-v1
See original license below.
"""

"""Installation how-to:
cargo install --target-dir ../cramming/dedup
Make sure to make sure that path_to_rust_code is set to the correct value if installing differently
"""

# ORIGINAL LICENSE:

# Copyright 2021 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datasets

import os
import numpy as np
from tqdm import tqdm

import time
import tempfile

import torch


def deduplicate_huggingface_dataset(dataset, threshold=100, original_cwd="."):
    """ "Seamlessly" run exact deduplication as in Lee et al."""
    path_to_rust_code = os.path.join(original_cwd, "dedup", "release")
    with tempfile.TemporaryDirectory() as tmpdir:
        text_file = _write_tmp_file(dataset, dirname=tmpdir)
        _make_suffix_array(text_file, tmpdir, path_to_rust_code)

        # Run other rust code directly
        options = f"--length-threshold {threshold} --cache-dir {tmpdir}/cache/"

        print("Finding self-similar parts...")
        os.popen(
            f"{path_to_rust_code}/dedup_dataset self-similar --data-file {text_file} " f"{options} --num-threads {torch.get_num_threads()}"
        ).read()
        print("Collect self-similar from all parts...")
        os.popen(f"{path_to_rust_code}/dedup_dataset collect --data-file {text_file} " f"{options}> {tmpdir}/drop_tokens_file").read()
        dataset = _finish_and_return_to_hf_dataset(text_file, f"{tmpdir}/drop_tokens_file")
    return dataset


def _write_tmp_file(dataset, dirname):
    text_file = os.path.join(dirname, "tmp_full_dataset_as_text")

    with open(text_file, "wb") as fout:
        for example in tqdm(dataset, desc="Writing dataset to tmp files."):  # not batched...
            fout.write((example["text"] + "<EOT>").encode("utf-8"))
    return text_file


def _make_suffix_array(text_file, tmpdir, path_to_rust_code):
    data_size = os.path.getsize(text_file)
    HACK = 100000

    started = []

    if data_size > 10e9:
        total_jobs = 100
        jobs_at_once = 20
    elif data_size > 1e9:
        total_jobs = 96
        jobs_at_once = 96
    elif data_size > 10e6:
        total_jobs = 4
        jobs_at_once = 4
    else:
        total_jobs = 4
        jobs_at_once = 1

    S = data_size // total_jobs
    print("Partition into parts and create suffix arrays...")
    for jobstart in range(0, total_jobs, jobs_at_once):
        wait = []
        for i in range(jobstart, jobstart + jobs_at_once):
            s, e = i * S, min((i + 1) * S + HACK, data_size)
            cmd = f"{path_to_rust_code}/dedup_dataset make-part --data-file {text_file} --start-byte {s} --end-byte {e}"
            started.append((s, e))
            # print(cmd)
            wait.append(os.popen(cmd))

            if e == data_size:
                break

        print("Waiting for jobs to finish")
        [x.read() for x in wait]

    print("Checking all wrote correctly")

    while True:
        files = [f"{text_file}.part.{s}-{e}" for s, e in started]

        wait = []
        for x, (s, e) in zip(files, started):
            size_data = os.path.getsize(x)
            FACT = np.ceil(np.log(size_data) / np.log(2) / 8)
            # print("FACT", FACT)
            size_table = os.path.getsize(x + ".table.bin")
            if not os.path.exists(x) or not os.path.exists(x + ".table.bin") or size_table == 0 or size_data * FACT != size_table:
                cmd = f"{path_to_rust_code}/dedup_dataset make-part --data-file {text_file} --start-byte {s} --end-byte {e}"
                # print(cmd)
                wait.append(os.popen(cmd))
        print("Rerunning", len(wait), "jobs because they failed.")
        [x.read() for x in wait]
        time.sleep(1)
        if len(wait) == 0:
            break

    print("Merging suffix trees")

    torun = " --suffix-path ".join(files)
    options = f"--output-file {tmpdir}/out.table.bin --suffix-path {torun} --num-threads {torch.get_num_threads()}"
    print(f"{path_to_rust_code}/dedup_dataset merge {options}")
    os.popen(f"{path_to_rust_code}/dedup_dataset merge {options}").read()
    # exit(0)
    print("Now merging individual tables")
    os.popen(f"cat {tmpdir}/out.table.bin.* > {tmpdir}/out.table.bin").read()
    print("Cleaning up")
    os.popen(f"mv {tmpdir}/out.table.bin {text_file}.table.bin").read()


def _finish_and_return_to_hf_dataset(original_text_file, remove_file_cache):
    """For simplicity the entire new dataset has to fit into memory..."""
    remove = []
    with open(remove_file_cache) as fin:
        for line in fin:
            if "out" in line:
                break
        for line in fin:
            remove.append(list(map(int, line.split())))
        remove = remove[::-1]

    print(f"Number of removal tuples is {len(remove)}")

    with open(original_text_file, "rb") as original_dataset:
        deduped_dataset = dict(text=[])
        start = 0
        buffer = ""
        for _ in tqdm(range(len(remove)), desc="Writing deduplicated data back to hf dataset"):
            a, b = remove.pop()
            buffer += original_dataset.read(a - start).decode("utf-8", errors="ignore")  # Is the error ignore here a terrible idea??
            original_dataset.seek(b)
            start = b

            buf_split = buffer.split("<EOT>")
            if len(buf_split) > 1:
                deduped_dataset["text"] += buf_split[:-1]
                buffer = buf_split[-1]
        deduped_dataset["text"] += (buffer + original_dataset.read().decode("utf-8")).split("<EOT>")[:-1]

    dataset = datasets.Dataset.from_dict(deduped_dataset)
    return dataset
