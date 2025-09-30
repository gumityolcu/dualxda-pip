# **dualxda**: A Library for Efficient, Effective and Explainable Data Attributions

<p style="font-size: 30px;"><b>DualXDA</b> delivers reliable, and extremely efficient data attribution, using orders-of-magnitude less time and memory while leading across diverse evaluation metrics. By combining data and feature attribution, DualXDA can explain the reasons for the attribution values it produces.</p>

:newspaper: Check out our [paper](https://arxiv.org/abs/2402.12118)!

:pencil2: Please cite us according to the bib entry below if you use DualXDA in your work:

```
@article{yolcu2025dualxda,
      title={DualXDA: Towards Sparse, Efficient and Explainable Data Attribution in Large AI Models}, 
      author={Galip Ümit Yolcu and Moritz Weckbecker and Thomas Wiegand and Wojciech Samek and Sebastian Lapuschkin},
      year={2025},
      eprint={2402.12118},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.12118}, 
}
```

Here is an example explanation you can generate using this package, using the image of a wolf from the AwA2 dataset as the test image:

<img src="./img/fig1.png" width="1300">

In this figure, we see
- **1-** A feature level explanation on the model decision (wolf). Blue color indicates negative evidence, whereas red color is used for regions that encourage the classification being explained. Surprisingly, the distinctive facial features are working against this classification.
- **2-** Positively relevant training images, showing the evidence from the training dataset that corroborates the classification being explained.
- **3 and 4-** Heatmaps showing the interaction between test and train features in determining the model output. We see that the model can match the snouts for the third and fourth images, and first two images contribute through fur features.
- **5-** Negatively relevant training samples. Notice that they are all foxes.
- **6 and 7-** We can now see that the LRP heatmap shows the snouts as a negatively relevant region, because the model detects similarity to the fox class through snout features. This reveals that for our model, snout features are detected on multiple canines and do not provide a robust basis for classification.   


In this repository, we present our Python package that can be used to produce data attributions and feature level explanations of these attribution values.
DualXDA is a methodology consisting of two steps:
1. **DualDA** replaces the final linear layer of a neural network with a surrogate kernel machine. This naturally produces a model where the output is determined as a sum of contributions from each training point, which constitutes a data attribution strategy. DualDA provides state-of-the-art attribution quality while showcasing 11,000x or bigger improvements in the required computation time.

2. **XDA**, on the other hand, relies on Layerwise Relevance Propagation to reveal the reasons for highly influential training data. For any given pair of train-test datapoints, XDA produces heatmaps concurrently on the two inputs, which indicate their interplay and the reasons for the computed attribution value for the training datapoint.

For further details, we refer the reader to our publication [DualXDA: Towards Sparse, Efficient and Explainable Data Attribution in Large AI Models](https://arxiv.org/abs/2402.12118). Currently, only classification models can be attributed.


## Installation

In order to install dualxda, use pip:

``pip install dualxda``

If you are using another package manager, please first install our modification of the sklearn from [here](https://github.com/gumityolcu/scikit-learn-dual). Note that this will install a package with name, sklearn_dual and not clash with any other installation of sklearn.

## Usage

### DualDA: Efficient Data Attribution using SVM surrogates

Here is a list of components you need to start attributing model outputs to training data:

- `model: torch.nn.Module` torch module that performs classification
- `train_dataset: torch.utils.data.Dataset` Dataset object to attribute against, including the input transformations needed for model inference
- `cache_path: str` Path of cache directory to save surrogate model parameters
- `layer_name: str` The name of the final classification layer of the model
- `device` A device parameter that torch functions will accept

These components are enough to create a `DualDA` object:

```python
from dualxda import DualDA

da = DualDA(model, train_dataset, layer_name, device, cache_path)
```

This will check the `cache_path` for existing caches, and use it if there is any. If not, it will train a surrogate SVM on the inputs of the supplied `layer_name`. It will then save the minimal required assets for using DualDA in the future, in the cache directory.

Afterwards, we need a `test_dataset` to get test samples and attribute the model output on those test samples:

```python
ldr = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

x, y = next(iter(ldr))
preds = model(x.to("cuda")).argmax(dim=-1)
xpl = da.attribute(x=x, xpl_targets=preds)
```
In the final line, `xpl_targets` selects the outputs to be attributed using the training dataset. In practice, this is either the ground truth labels of the test data. Finally, the return object `xpl` is a `torch.Tensor` of shape (BATCH SIZE, TRAINING DATA SIZE) which holds the data attributions for the given batch.

The `DualDA` object uses a hook to collect intermediate representations from the model, when you are done with your attributions, you might want to remove the hook:

```python
da.remove_hook()
```

Alternatively, you can use Python contexts which will handle removing the hooks when the context ends:

``` python
with DualDA(
        model,
        train_dataset,
        layer_name,
        device,
        cache_path,
    ) as da:
        x, y = next(iter(ldr))
        preds = model(x.to("cuda")).argmax(dim=-1)
        xpl = da.attribute(x=x, xpl_targets=preds)
```
## Handling Sparsity
DualDA provides effective explanations using a small percentage of the training dataset, using the datapoints which are support vectors in the surrogate model. The level of sparsity can be adjusted with the C parameter while constructing the `DualDA` object, which is set to 1.0 by default. A higher C value will result in less sparsity:

```python

da = DualDA(model, train_dataset, layer_name, device, cache_path, C=2.0)
```

The attributions generated by `DualDA.attribute` include all the training datapoints by default. Alternatively, you can just get the nonzero attributions on support vectors by setting `drop_zero_columns=True` when you call the `attribute` method. You can also get the indices of the support vectors in the training dataset using `DualDA.active_indices`.

## XDA

To get a feature level explanation of the computed attribution scores, you can call `da.xda_heatmap` with the id of the training point you want to attribute against. You can set `mode="train"` or `mode="test"` to get feature level explanations on the train or the test datapoint. The two feature heatmaps enlighten the interaction of the two datapoints, showing how different parts of the train and test image determine the model output.

```python
relevance_map = da.xda_heatmap(
                test_sample,
                train_sample_id,
                attribution_value,
                mode="train",
                composite=composite,
            )
```

### zennit Composites and Canonizers 
This will use the [Layerwise Relevance Propagation (LRP)](https://iphome.hhi.de/samek/pdf/MonXAI19.pdf), a feature attribution method from Explainable AI, to generate feature level explanations. In terms of software, this is done using [zennit](https://zennit.readthedocs.io/). zennit implements different LRP rules using the `Composite` class. The `composite` variable in the above code can be a string, containing the name of most commonly used composites, like "EpsilonPlusFlat" or "EpsilonAlpha2Beta1". If you want to other or custom composites, you can also directly pass a `zennit.Composite` object.

For models that include batch normalization and skip connections, best practices in LRP includes a [canonization](https://arxiv.org/abs/2211.17174) step. This is handled through the `canonizer` argument. You can pass custom `zennit.Canonizer` objects according to your model's architecture. For further details, please read into [zennit's documentation](https://zennit.readthedocs.io/).

## Producing Figures
You can use the dualxda library to generate figures from your data attributions and XDA attributions.

```python
with DualDA(
    model,
    train_ds,
    layer_name,
    device,
    cache_dir=cache_dir,
) as da:
    x, y = next(iter(ldr))
    preds = model(x.to("cuda")).argmax(dim=-1)
    xpl = da.attribute(x=x, xpl_targets=preds)
    da.da_figure(
        test_sample=x[0],
        inv_transform=train_ds.inverse_transform,
        class_names=range(10),
        attr=xpl[0],
        save_path="./xda_figures",
        fname=f"da_figure",
    )
    da.xda(
        test_sample=x[0],
        inv_transform=train_ds.inverse_transform,
        class_names=range(10),
        attr=xpl[0],
        attr_target=preds[0],
        composite="EpsilonPlus",
        canonizer="SequentialMergeBatchNorm",
        save_path="./xda_figures",
        fname=f"xda_figure",
    )
```