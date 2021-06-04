# ReadMe

## 1. 训练

运行./src/train_xxx.py文件，一共包含两种模型，以Eb5为encoder，transformer为decoder的模型

以Eb6为encoder，transformer为decoder的模型

## 2. 提交

运行./src/submit_xxx.py文件。

如果要使用模型集成，运行后缀为ensamble的文件

## 3. 后处理

这个竞赛后处理尤为重要，我集成之后最好的结果为0.97，但是通过后处理可以到达0.77，提升了0.2个点。后处理有两个部分构成.

1. 使用 postprogress_normalize.py 文件，将结果消除空格等，无意义填充，确保所有的结果都是合法的化学式。
2. 使用 valid_rdkit.py 文件，交叉比对 2个或3个 提交csv文件，尽量确保每一项的化学式都是合法的，例如，如果sub1中有一项不合法，而sub2中合法，那就使用sub2中。