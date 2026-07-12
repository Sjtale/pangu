#!/usr/bin/env python3
"""Losslessly gzip a checkpoint and verify byte-exact decompression."""

import argparse
import gzip
import hashlib
import os


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compress_checkpoint(source, output, level=9):
    source = os.path.abspath(source)
    output = os.path.abspath(output)
    if source == output:
        raise ValueError("Source and output paths must differ")
    if os.path.exists(output):
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp"
    source_hash = sha256_file(source)
    restored_hash = hashlib.sha256()
    try:
        with open(source, "rb") as source_stream, gzip.open(
            temporary, "wb", compresslevel=level
        ) as output_stream:
            for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                output_stream.write(chunk)
        with gzip.open(temporary, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                restored_hash.update(chunk)
        if restored_hash.hexdigest() != source_hash:
            raise RuntimeError("Decompressed SHA256 differs from source")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)

    source_size = os.path.getsize(source)
    output_size = os.path.getsize(output)
    print(f"Source: {source}")
    print(f"Output: {output}")
    print(f"SHA256: {source_hash}")
    print(f"Size: {source_size / 2**20:.2f} -> {output_size / 2**20:.2f} MiB")
    print(f"Reduction: {(1.0 - output_size / source_size) * 100.0:.2f}%")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default="data/checkpoints/model_fp16_alias_compact.pth"
    )
    parser.add_argument(
        "--output", default="data/checkpoints/model_fp16_alias_compact.pth.gz"
    )
    parser.add_argument("--level", type=int, choices=range(1, 10), default=9)
    args = parser.parse_args()
    compress_checkpoint(args.source, args.output, args.level)


if __name__ == "__main__":
    main()
