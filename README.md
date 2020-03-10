Ray's Machine Learning Tools (remake from KaggleWidget.TF_Utils)
---

A light-weighted and easy-to-use tools for general deep learning based on tensorflow 2.0

* Components:
    *  Database
    *  Model
    *  Trainers
    *  Scenes
---

**Database**

    It is a re-implementation of [tensorflow_dataset](https://www.tensorflow.org/datasets),
    while simplify its APIs and remove some functionalities for easy use.

    Example usage:
    '''python

    import tensorflow as tf
    from MLBOX.Database import DBLoader

    loader = DBLoader()
    loder.load_built_in("mnist")
    train, test = loader.train, loader.test

    # now train && test are Database.core.database.Dataset object
    # which can be sliced/splitted

    assert train.count == 60000

    train, validation = train.split(ratio=0.8)
    assert train.count == 48000
    assert validation.count == 12000

    # tensorflow.data.Dataset can be get via .to_tfdataset
    ds = train.to_tfdataset(epoch=1, batch=1)
    assert isinstance(ds, tf.data.Dataset)
    '''

**Model**

    (On-Going) Supports both classfical and SOTA deep learning models

**Trainers**

    Provide out-of-box trainer for Model on Database.

---

RoadMap/TODO:
    # checkout fast.ai (https://docs.fast.ai/index.html) to get some useful features
    # fully upgrade to tensorflow 2.0
    # rewrite dataset to support more database building options
        - like store with structure (currently always random storing)
        - more custom parsing options (like parsing data in pairs)

    # add more CV SOTA models
        -- Optimizer: lookahead, Adabound, ...
        -- VAE: VAE, VQ-VAE, ...
	-- Classification: ResNeXt, SENet, SE-ResNet, ...
	-- ObjectDetection: YoloV3, ...
        -- Bags of tricks: cosine decay, label smoothing, mixup, distill, ...

    # intergrate time series and tabular data
    # support pytroch
        -- how to deal with tfrecord and pytorch?

Progress:
    [Updated] ModelLogger re-written Dec. 2019
    better split machnimsm
