# ECGrecover

A Deep Learning Approach for Electrocardiogram Signal Completion.


## Current package structure
```
ECGrecover
├── compute_metrics
│    ├── compute_metrics.py
│    ├── compute_peaks.py
│    └── __init__.py
├── data
│    ├── test.npy
│    ├── train.npy
│   └── validation.npy
├── learn
│    ├── __init__.py
│    └── Training.py
├── model
├── Results
├── tools
│ 	├── CompleteData.py
│ 	├── LoadModel.py
│ 	├── LossFunction.py
│ 	├── PreProcessing.py
│ 	└── SoftDTW_function.py
├── main.py
├── readme.md
└── enviroment.yml
```



## Usage

To execute ECGrecover:

- Create and install the enviroment:
```console
foo@bar:~$ conda env create -f environment.yml
```

- Activate the enviroment:
```console
foo@bar:~$ source activate ECGrecover_env
```

- Launch the test from CLI:

<code>-batch_size</code>: default 256</code><br/>
<code>-data_path</code>: data folder<br/>
<code>-device</code>: GPU on which the model will be loaded<br/>
<code>-epoch</code>: default 10  <br/>
<code>-save_path</code>: model folder<br/>
<code>-save_results</code>: tables/plots folder when testing the model<br/>
<code>-seed</code>: seed to shuffle the data<br/>
<code>- --Train</code>: to train the model<br/>




- To train the model:
```console
(ECGrecover_env) foo@bar:~$ python main.py data/ model/ 0 1 --Train empty 100 256
```

- To test the model:
```console
(ECGrecover_env) foo@bar:~$ python main.py data/test.npy model/ 0 1 Results/
```

#### Enviroment
We run each experiment on a machine equipped with Nvidia A100 80GB GPU, with the total training duration approximating 1 hour and 30 minutes.

### Please note
The Generepol dataset is not public.

