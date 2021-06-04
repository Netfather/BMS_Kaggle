# ReadMe

## 1. 数据下载

请访问比赛官网  

[Bristol-Myers Squibb - Molecular Translation](https://www.kaggle.com/c/bms-molecular-translation/overview)

然后将下载的数据解压后放在./data/bms_dataset 文件夹下

## 2. 数据预处理

我们需要将数据从分开的格式对应为数据-化学式对应的格式。

请运行 ./src/pdata_new.py 文件。

运行完成后，将会在 ./data 文件夹下生成若干个pickle文件和对应的train,test的csv文件

## 3. 测试数据去噪

由于test中的数据噪声比远比train中噪声严重，特别是黑白噪点，因此使用去噪手段将test中所有图片去噪，然后再做推断。

请运行 ./src/utils/den_deb_box/DenoDeblur.py 文件，完成后，将会在./data/bms_dataset文件夹下生成denoise_test文件。