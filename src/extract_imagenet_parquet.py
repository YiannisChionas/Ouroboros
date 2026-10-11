#!/usr/bin/env python3
"""Extract HuggingFace ImageNet-1k parquet files to an ImageFolder tree.

    {dst}/{split}/{label:04d}/{split}_{file_idx:03d}_{row:05d}.JPEG

Image bytes are written as-is (no re-encoding). Class folders are zero-padded so
that ImageFolder's sorted class order equals the original label index.

Resumable: a marker {dst}/.done/{parquet_name} is written after each parquet file
is fully extracted; marked files are skipped on re-run.

Usage:
  python extract_imagenet_parquet.py --src /opt/storage/datasets/imagenet1k/data \
      --dst /var/local/storage/$USER/imagenet1k --workers 4
"""

import argparse
import glob
import os
import re
from multiprocessing import Pool

import pyarrow.parquet as pq

SPLITS = {'train': 'train', 'validation': 'val'}
EXPECTED = {'train': 1281167, 'val': 50000}


def extract_file(job):
    path, dst = job
    name = os.path.basename(path)
    marker = os.path.join(dst, '.done', name)
    if os.path.exists(marker):
        return name, 0, True

    m = re.match(r'(train|validation)-(\d+)-of-\d+\.parquet$', name)
    split, file_idx = SPLITS[m.group(1)], int(m.group(2))

    n = 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=256, columns=['image', 'label']):
        images = batch.column('image').to_pylist()
        labels = batch.column('label').to_pylist()
        for img, label in zip(images, labels):
            out_dir = os.path.join(dst, split, f'{label:04d}')
            out = os.path.join(out_dir, f'{split}_{file_idx:03d}_{n:05d}.JPEG')
            with open(out, 'wb') as f:
                f.write(img['bytes'])
            n += 1

    with open(marker, 'w') as f:
        f.write(f'{n}\n')
    return name, n, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',     default='/opt/storage/datasets/imagenet1k/data')
    parser.add_argument('--dst',     required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, 'train-*.parquet')) +
                   glob.glob(os.path.join(args.src, 'validation-*.parquet')))
    print(f'Found {len(files)} parquet files in {args.src}', flush=True)

    os.makedirs(os.path.join(args.dst, '.done'), exist_ok=True)
    for split in SPLITS.values():
        for label in range(1000):
            os.makedirs(os.path.join(args.dst, split, f'{label:04d}'), exist_ok=True)

    done = 0
    with Pool(args.workers) as pool:
        for name, n, skipped in pool.imap_unordered(extract_file, [(f, args.dst) for f in files]):
            done += 1
            status = 'skipped (already done)' if skipped else f'{n} images'
            print(f'[{done}/{len(files)}] {name}: {status}', flush=True)

    # Final check: per-split image counts vs the official ImageNet-1k sizes
    ok = True
    for split, expected in EXPECTED.items():
        count = sum(len(os.listdir(os.path.join(args.dst, split, d)))
                    for d in os.listdir(os.path.join(args.dst, split)))
        print(f'{split}: {count} images (expected {expected})', flush=True)
        ok &= count == expected
    if ok:
        open(os.path.join(args.dst, 'EXTRACTION_COMPLETE'), 'w').close()
    print('EXTRACTION_COMPLETE' if ok else 'EXTRACTION_INCOMPLETE', flush=True)


if __name__ == '__main__':
    main()
