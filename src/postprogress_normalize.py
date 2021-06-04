# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         postprogress_normalize
# Description:  此文件用于 对最终生成的submit文件进行最终的收敛
# Copy from    https://www.kaggle.com/nofreewill/normalize-your-predictions
# 2021年6月1日 V1. 如果我们有两份不同的normalize文件 我们可以通过观察两份文件中的 合法项来合并一份最终的文件。 一种不带beam search版的 redikt validate
# Author:       Administrator
# Date:         2021/5/21
# -------------------------------------------------------------------------------
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger

import pandas as pd
import edlib
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
from pathlib import Path


def normalize_inchi(inchi):
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            return inchi
        else:
            try:
                return Chem.MolToInchi(mol)
            except:
                return inchi

    except: return inchi


# Segfault in rdkit taken care of, run it with:
# while [ 1 ]; do python normalize_inchis.py && break; done
if __name__ == '__main__':
    # Input & Output
    orig_path = Path('submit.csv')
    norm_path = orig_path.with_name(orig_path.stem + '_norm.csv')

    # Do the job
    N = norm_path.read_text().count('\n') if norm_path.exists() else 0
    print(N, 'number of predictions already normalized')

    r = open(str(orig_path), 'r')
    w = open(str(norm_path), 'a', buffering=1)

    for _ in range(N):
        r.readline()
    line = r.readline()  # this line is the header or is where it segfaulted last time
    w.write(line)

    for line in tqdm(r):
        splits = line[:-1].split(',')
        image_id = splits[0]
        inchi = ','.join(splits[1:]).replace('"', '')
        inchi_norm = normalize_inchi(inchi)
        w.write(f'{image_id},"{inchi_norm}"\n')

    r.close()
    w.close()


    sub_df = pd.read_csv('submit.csv')
    sub_norm_df = pd.read_csv('submit_norm.csv')

    lev = 0
    N = len(sub_df)
    for i in tqdm(range(N)):
        inchi, inchi_norm = sub_df.iloc[i,1], sub_norm_df.iloc[i,1]
        lev += edlib.align(inchi, inchi_norm)['editDistance']

    print(lev/N)