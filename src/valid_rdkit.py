# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         valid_rdkit
# Description:  此文件用于融合两个指定的csv或者多个指定的csv文件 选取其中的合法rdkit 作为提交
# Author:       Administrator
# Date:         2021/6/1
# -------------------------------------------------------------------------------
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger

import pandas as pd
import edlib
RDLogger.DisableLog('rdApp.*')
from pathlib import Path


### 2文件版本
Name1 = 'submit_norm_1.84.csv'
Name2 = 'submit_norm_1.88.csv'
Name3 = 'submit_norm_1.84_norm.csv'


def normalize_inchi(inchi1,inchi2):
    try:
        mol1 = Chem.MolFromInchi(inchi1)
        mol2 = Chem.MolFromInchi(inchi2)
        if mol1 is None and mol2 is None:
            return inchi1
        elif mol1 is None:
            return Chem.MolToInchi(mol2)
        elif mol2 is None:
            return Chem.MolToInchi(mol1)
        else:
            return Chem.MolToInchi(mol1)
    except:
        return inchi1


# Segfault in rdkit taken care of, run it with:
# while [ 1 ]; do python valid_rdkit.py && break; done
if __name__ == '__main__':
    # Input & Output
    # orig_path = Path('submission.csv')
    orig_path1 = Path(Name1)
    orig_path2 = Path(Name2)
    # 占位

    norm_path = orig_path1.with_name(orig_path1.stem + '_norm.csv')

    # Do the job
    N = norm_path.read_text().count('\n') if norm_path.exists() else 0
    print(N, 'number of predictions already normalized')

    r1 = open(str(orig_path1), 'r')
    r2 = open(str(orig_path2), 'r')
    # 占位

    w = open(str(norm_path), 'a', buffering=1)

    for _ in range(N):
        r1.readline()
        r2.readline()
        # 占位

    line1 = r1.readline()  # this line is the header or is where it segfaulted last time
    line2 = r2.readline()  # this line is the header or is where it segfaulted last time
    # 占位

    w.write(line1)  # 为了防止segment 预先把表现最好的csv文件写进去 然后再进行测试

    for id,(line1,line2) in enumerate(zip(tqdm(r1),tqdm(r2))):
        # if id >= 30:
        #     break
        splits1 = line1[:-1].split(',')
        splits2 = line2[:-1].split(',')
        # 占位

        image_id = splits1[0]
        assert(image_id == splits2[0]  )
        inchi1 = ','.join(splits1[1:]).replace('"', '')
        inchi2 = ','.join(splits2[1:]).replace('"', '')
        # 占位

        inchi_norm = normalize_inchi(inchi1,inchi2)
        w.write(f'{image_id},"{inchi_norm}"\n')


    r1.close()
    r2.close()
    w.close()
    #
    sub_df = pd.read_csv(Name1)
    sub_norm_df = pd.read_csv(Name3)

    lev = 0
    N = len(sub_df)
    for i in tqdm(range(N)):
        inchi, inchi_norm = sub_df.iloc[i,1], sub_norm_df.iloc[i,1]
        lev += edlib.align(inchi, inchi_norm)['editDistance']

    print(lev/N)



#### 三文件交叉验证版本
# Name1 = 'submit_norm_2.06.csv'
# Name2 = 'submit_norm_2.10.csv'
# Name3 = 'submit_norm_2.13.csv'
# Name_sub = 'submit_norm_2.06_norm.csv'
#
#
# def normalize_inchi(inchi1,inchi2,inchi3):
#     try:
#         mol1 = Chem.MolFromInchi(inchi1)
#         if mol1 is not None:
#             return Chem.MolToInchi(mol1)
#
#         mol2 = Chem.MolFromInchi(inchi2)
#         if mol2 is not None:
#             return Chem.MolToInchi(mol2)
#
#         mol3 = Chem.MolFromInchi(inchi3)
#         if mol3 is not None:
#             return Chem.MolToInchi(mol3)
#
#         return inchi1
#     except:
#         return inchi1
#
#
# # Segfault in rdkit taken care of, run it with:
# # while [ 1 ]; do python valid_rdkit.py && break; done
# if __name__ == '__main__':
#     # Input & Output
#     # orig_path = Path('submission.csv')
#     orig_path1 = Path(Name1)
#     orig_path2 = Path(Name2)
#     orig_path3 = Path(Name3)
#     # 占位
#
#     norm_path = orig_path1.with_name(orig_path1.stem + '_norm.csv')
#
#     # Do the job
#     N = norm_path.read_text().count('\n') if norm_path.exists() else 0
#     print(N, 'number of predictions already normalized')
#
#     r1 = open(str(orig_path1), 'r')
#     r2 = open(str(orig_path2), 'r')
#     r3 = open(str(orig_path3), 'r')
#     # 占位
#
#     w = open(str(norm_path), 'a', buffering=1)
#
#     for _ in range(N):
#         r1.readline()
#         r2.readline()
#         r3.readline()
#         # 占位
#
#     line1 = r1.readline()  # this line is the header or is where it segfaulted last time
#     line2 = r2.readline()  # this line is the header or is where it segfaulted last time
#     line3 = r3.readline()  # this line is the header or is where it segfaulted last time
#     # 占位
#
#     w.write(line1)  # 为了防止segment 预先把表现最好的csv文件写进去 然后再进行测试
#
#     for line1,line2,line3 in zip( tqdm(r1),tqdm(r2),tqdm(r3) ):
#         # if id >= 30:
#         #     break
#         splits1 = line1[:-1].split(',')
#         splits2 = line2[:-1].split(',')
#         splits3 = line3[:-1].split(',')
#         # 占位
#         # print(splits3,splits2,splits1)
#
#         image_id = splits1[0]
#         assert(image_id == splits2[0]  and image_id == splits3[0])
#         inchi1 = ','.join(splits1[1:]).replace('"', '')
#         inchi2 = ','.join(splits2[1:]).replace('"', '')
#         inchi3 = ','.join(splits3[1:]).replace('"', '')
#         # 占位
#
#         inchi_norm = normalize_inchi(inchi1,inchi2,inchi3)
#         w.write(f'{image_id},"{inchi_norm}"\n')
#
#
#     r1.close()
#     r2.close()
#     r3.close()
#     w.close()
#
#     sub_df = pd.read_csv(Name1)
#     sub_norm_df = pd.read_csv(Name_sub)
#
#     lev = 0
#     N = len(sub_df)
#     for i in tqdm(range(N)):
#         inchi, inchi_norm = sub_df.iloc[i,1], sub_norm_df.iloc[i,1]
#         lev += edlib.align(inchi, inchi_norm)['editDistance']
#
#     print(lev/N)