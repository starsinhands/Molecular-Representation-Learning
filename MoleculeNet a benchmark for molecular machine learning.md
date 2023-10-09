# [MoleculeNet: a benchmark for molecular machine learning](https://pubs.rsc.org/en/content/articlelanding/2018/SC/C7SC02664A)

**摘要：**MoleculeNet包含多个公共数据集，建立了评估度量，并提供多个先前提出的分子特征和学习算法的高质量开源实现（DeepChem库中）。此外，MoleculeNet benchmarks表明，可学习表示是分子机器学习的强大工具，并广泛提供最好的性能。然而，这个结果是需要注意的。在数据稀缺和高度不平衡的分类条件下，可学习的表示仍然难以处理复杂的任务。对于[量子力学](https://so.csdn.net/so/search?q=量子力学&spm=1001.2101.3001.7020)和生物物理数据集，使用物理感知的特征化(physics-aware featurizations)可能比选择特定的学习算法更重要。

机器学习对分子性质研究的意义重大，但目前提出的大多算法均在不同的数据集上进行测试，因此很难比较这些算法的性能。并且分子数据库通常比较小，化学研究的宽度又很广，因此，分子机器学习需要能预测这些宽范围的性质是一个很有挑战性的任务。输入的分子形态大小不一，连通性和构象之间的差距很大，因此还需将分子转化成适合机器学习的形式，这边需要从分子中提取有用的相关信息进行特征化。总结：**分子机器学习的困难：数据量的限制，预测范围广，输入分子结构的异质性和学习算法的选择**。因此，这项工作旨在通过管理一些数据集集合，创建一套这样的软件：实现许多已知的分子特征，并提供高质量的算法实现，从而促进分子[机器学习方法](https://so.csdn.net/so/search?q=机器学习方法&spm=1001.2101.3001.7020)的发展。就像WordNet 和 ImageNet一样。

MoleculeNet 包含超过70万种化合物的性质数据，所有的数据都被收录进开源的**DeepChem**包

DeepChem的用户可以通过提供的库调用轻松地加载所有这些数据。MoleculeNet中也包含了一些常见的（生物）化学特征化方法，以及一些机器学习算法的实现，这些实现依赖于Scikit-Learn和Tensorflow。此外，值得注意的是，在模型评估时，**机器学习中常见的Random splitting方法并不适用于化学数据**。MoleculeNet为DeepChem提供了一个数据分割机制库，并通过多种数据分割方法来评估所有算法。

现有的一些数据集：PubChem、PubChem BioAssasy、ChEMBL、PubChem、ChemSpider、Crystallography Open Database、Cambridge Structural Database、protein data bank。

但上述数据库都不是针对机器学习的，这些数据集没有定义衡量算法有效性的**指标**，也没有将**数据分割**为训练/验证/测试集。然而，不同组之间的评价指标和子集的选择差异很大，导致使用统一数据库的两篇文章可能无法进行比较。![](C:\Users\蒋仔\Desktop\夏令营4429964990 230626\fig1.png)

MoleculeNet包含大量公共数据库（17个）用于测试不同性质，这些性质主要分为四大类：量子力学，物理化学，生物物理学和生理学。

**具体的数据库内容推荐的数据分割方法及评价指标如下表：**![](C:\Users\蒋仔\Desktop\夏令营4429964990 230626\table1.png)

**以及这些数据库的性能总结：**![](C:\Users\蒋仔\Desktop\夏令营4429964990 230626\table3.png)

### 不同的数据分割方法

​    Random splitting：随机分割

​    Scaffffold splitting: 根据样本的二维结构框架对样本进行分割**<span style="color:red">（在RDKit中实现）</span>**

​    Stratified random sampling:该方法按增加标签值的顺序对数据点进行排序

​    Time splitting:包含时间信息

### 特征化方法

​    Smiles：有局限（大多数分子机器学习方法需要进一步的信息，从有限的数据中学习分子的复杂电子或拓扑特征），但是有潜力（一些研究已经证明）。

​    MoleculeNet提供了6种表示方法：ECFP、Coulomb matrix、Grid featurizer、Symmetry function、Graph convolutions和Weave。

### 模型

传统模型：

​    Logistic regression、Support vector classification、Kernel ridge regression、Random forests、Gradient boosting、Multitask/singletask network、Bypass multitask networks、Influence relevance voting

图模型：

​    Graph convolutional models、Weave models、Directed acyclic graph models、Deep tensor neural networks、ANI-1、Message passing neural networks

###  结果

​    基于图的模型(图卷积模型、编织模型和DTNN)在大多数数据集上的优势优于其他方法，但在数据稀缺的情况下，基于图的方法对复杂任务不够稳健；在严重不平衡的类数据集上，传统的方法如核SVM在阳性召回率方面优于可学习的特征。并且对不同的任务使用特殊性质是很有必要的，使用距离信息的DTNN和MPNN在QM数据集上比简单的图卷积表现得更好。未来，数据驱动的算法可能优于物理算法，代替手工算法。



-----------------



-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

deepchem是一个Python库，用于在分子和量子数据集上进行机器学习和深度学习。它基于PyTorch和其他流行的机器学习框架构建。它的目的是让用户能够轻松地将机器学习应用到新的领域，并构建和评估新的模型。它还旨在让用户能够轻松地在生产环境中使用机器学习，通过提供易于使用的接口和工具

>**数据：**提供了常用的分子和量子数据集，以及加载、处理、划分、增强等数据操作的函数。
>**特征：**提供了常用的特征提取方法，如Featurizer对象，以及自定义特征函数的接口。
>**模型：**提供了常用的机器学习和深度学习模型，如多层感知器（MLP）、图神经网络（GNN）、变分自编码器（VAE）等，以及自定义模型函数的接口。
>**评估：**提供了一些常用的评估指标和方法，如均方误差（MSE）、准确率（Accuracy）、受试者工作特征曲线（ROC）等，以及自定义评估函数的接口。
>**应用：**提供了一些常见的化学和物理应用场景和示例代码，如药物发现、量子化学等。

**1：**先通过下面的代码安装好deepchem：

```python
import sys
import os
import requests
import subprocess
import shutil
from logging import getLogger, StreamHandler, INFO


logger = getLogger(__name__)
logger.addHandler(StreamHandler())
logger.setLevel(INFO)


def install(
        chunk_size=4096,
        file_name="Miniconda3-latest-Linux-x86_64.sh",
        url_base="https://repo.continuum.io/miniconda/",
        conda_path=os.path.expanduser(os.path.join("~", "miniconda")),
        rdkit_version=None,
        add_python_path=True,
        force=False):
    """install rdkit from miniconda
    ```
    import rdkit_installer
    rdkit_installer.install()
    ```
    """

    python_path = os.path.join(
        conda_path,
        "lib",
        "python{0}.{1}".format(*sys.version_info),
        "site-packages",
    )

    if add_python_path and python_path not in sys.path:
        logger.info("add {} to PYTHONPATH".format(python_path))
        sys.path.append(python_path)

    if os.path.isdir(os.path.join(python_path, "rdkit")):
        logger.info("rdkit is already installed")
        if not force:
            return

        logger.info("force re-install")

    url = url_base + file_name
    python_version = "{0}.{1}.{2}".format(*sys.version_info)

    logger.info("python version: {}".format(python_version))

    if os.path.isdir(conda_path):
        logger.warning("remove current miniconda")
        shutil.rmtree(conda_path)
    elif os.path.isfile(conda_path):
        logger.warning("remove {}".format(conda_path))
        os.remove(conda_path)

    logger.info('fetching installer from {}'.format(url))
    res = requests.get(url, stream=True)
    res.raise_for_status()
    with open(file_name, 'wb') as f:
        for chunk in res.iter_content(chunk_size):
            f.write(chunk)
    logger.info('done')

    logger.info('installing miniconda to {}'.format(conda_path))
    subprocess.check_call(["bash", file_name, "-b", "-p", conda_path])
    logger.info('done')

    logger.info("installing rdkit")
    subprocess.check_call([
        os.path.join(conda_path, "bin", "conda"),
        "install",
        "--yes",
        "-c", "rdkit",
        "python=={}".format(python_version),
        "rdkit" if rdkit_version is None else "rdkit=={}".format(rdkit_version)])
    logger.info("done")

    import rdkit
    logger.info("rdkit-{} installation finished!".format(rdkit.__version__))


if __name__ == "__main__":
    install()
```

安装：

>conda install -c [conda-forge](https://so.csdn.net/so/search?q=conda-forge&spm=1001.2101.3001.7020) deepchem
>
>或者：
>
>conda install -c deepchem deepchem

**2：**使用TensorGraph API训练一个基于**图卷积**的模型来预测Tox21数据集中的化合物毒性

```python
'''Script that trains low data models on Tox21 dataset.'''
 
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
 
import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc
 
# Load Tox21 dataset
n_features = 1024
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21()
train_dataset, valid_dataset, test_dataset = tox21_datasets
 
# Fit models
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
 
n_layers = 1 # shallow model since so little data
nb_epoch = 10 # very few epochs needed and more tend to overfit
 
model_1 = dc.models.TensorGraphMultiTaskClassifier(
    len(tox21_tasks), n_features,
    layer_sizes=[1000]*n_layers,
    dropouts=[.25]*n_layers,
    weight_init_stddevs=[.02]*n_layers,
    bias_init_consts=[1.]*n_layers,
    learning_rate=0.0003)
 
model_2 = dc.models.TensorGraphMultiTaskClassifier(
    len(tox21_tasks), n_features,
    layer_sizes=[1000]*n_layers,
    dropouts=[.25]*n_layers,
    weight_init_stddevs=[.02]*n_layers,
    bias_init_consts=[1.]*n_layers,
    learning_rate=0.0003)
 
model_3 = dc.models.TensorGraphMultiTaskClassifier(
    len(tox21_tasks), n_features,
    layer_sizes=[1000]*n_layers,
    dropouts=[.25]*n_layers,
    weight_init_stddevs=[.02]*n_layers,
    bias_init_consts=[1.]*n_layers,
    learning_rate=0.0003)
 
def reshape_y_pred(y_true, y_pred):
  """
  TensorFlow is very picky about shapes.
  """
  shape = y_pred.get_shape().as_list()
  start = len(shape) - len(y_true.get_shape().as_list())
  if start < 0:
      raise ValueError("Wrong number of dimensions")
  for i in range(start):
      y_true = tf.expand_dims(y_true, axis=i)
  return y_true
 
def add_training_loss(model):
  label_placeholder = model.label_placeholders[0]
  weights_placeholder = model.weights_placeholders[0]
  
  logits = model.outputs[0]
  
  weighted_loss_function_dnn_multitask_classification(logits=logits,label_placeholder=label_placeholder,num_classes=2,num_tasks=len(tox21_tasks),weights_placeholder=weights_placeholder)
  
def weighted_loss_function_dnn_multitask_classification(logits,label_placeholder,num_classes,num_tasks,weights_placeholder):
  
   # Reshape into batch_size x num_task x num_classes tensor for softmax_cross_entropy_with_logits
  
   logits_reshaped=tf.reshape(logits,(tf.shape(logits)[0],num_tasks,num_classes))
   labels_reshaped=tf.reshape(label_placeholder,(tf.shape(label_placeholder)[0],num_tasks))
   weights_reshaped=tf.reshape(weights_placeholder,(tf.shape(weights_placeholder)[0],num_tasks))
   
   # Convert labels into one-hot encoding
   
   labels_one_hot=tf.one_hot(tf.cast(labels_reshaped,dtype=tf.int32),depth=num_classes,axis=-1)
   
   # Compute loss
   
   loss_per_task_per_datapoint=tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot,logits=logits_reshaped,dim=-1)
   
   # Weight by datapoint weights
   
   weighted_loss_per_task_per_datapoint=tf.multiply(loss_per_task_per_datapoint,weights_reshaped)
   
   # Average over datapoints
   
   loss_per_task=tf.reduce_mean(weighted_loss_per_task_per_datapoint,axis=0)
   
   # Add losses together
   
   total_loss=tf.reduce_sum(loss_per_task,axis=-1) 
   
   return total_loss
 
 
model_1.add_output(model_1.outputs[-1])
model_2.add_output(model_2.outputs[-1])
model_3.add_output(model_3.outputs[-1])
 
add_training_loss(model_1)
add_training_loss(model_2)
```

**3：**用作和deepchem对比

```
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv', splitter='random')
```

**4：**

```python
import numpy as np
import deepchem as dc
# 载入一个毒性数据集——特征化过程为将包含分子信息的数据集转换为矩阵和向量
tox21_tasks,tox21_datasets,transformers = dc.molnet.load_tox21()
# 查看一下数据
print(f'tox21_tasks:{tox21_tasks};tox21_datasetsL:{tox21_datasets}')
# 分割数据集
train_dataset,valid_dataset,test_dataset = tox21_datasets
# 查看Transformer
print(f'transformers:{transformers}')
# [<deepchem.trans.transformers.BalancingTransformer at XXXXXXXXX>]
# 建立全连接网络
model = dc.models.MultitaskClassifier(n_tasks=12,n_features=1024,layer_sizes=[1000])
model.fit(train_dataset,nb_epoch=100)
metric = dc.metrics.Metric(dc.metrics.roc_auc_score,np.mean)
# 模型评估
train_scores = model.evaluate(train_dataset,[metric],transformers)
test_scores = model.evaluate(test_dataset,[metric],transformers)
print(train_scores)
print(test_scores)
```

>## **Error：**
>
>2023-08-03 15:49:29.524489: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
>2023-08-03 15:49:29.524610: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.

**Step5：**

**Step6：**

**Step7：**

**Step8：**

**Step9：**

**Step10：**

## solved（命令

```python
python setup.py build
sudo python setup.py install
python setup.py install
conda install -c rdkit rdkit
https://blog.csdn.net/baidu_39389949/article/details/126172675
```

