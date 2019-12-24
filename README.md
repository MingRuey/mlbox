Ray's Machine Learning Tools (remake from KaggleWidget.TF_Utils)

---

Frameworks: Tensorflow 2.0

Components:
    --  Database
    --  Model
    --  Trainers
    --  Scenes

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
