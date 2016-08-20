## End-To-End Memory Networks for Question Answering
This is an implementation of MemN2N model in Python for the [bAbI question-answering tasks](http://fb.ai/babi) 
as shown in the Section 4 of the paper "[End-To-End Memory Networks](http://arxiv.org/abs/1503.08895)". It is based on 
Facebook's [Matlab code](https://github.com/facebook/MemNN/tree/master/MemN2N-babi-matlab).

![Web-based Demo](http://i.imgur.com/mKtZ7kB.gif)

## Requirements
* Python 2.7
* Numpy, Flask (only for web-based demo) can be installed via pip:
```
$ sudo pip install -r requirements.txt
```
* [bAbI dataset](http://fb.ai/babi) should be downloaded to `data/tasks_1-20_v1-2`: 
```
$ wget -qO- http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz | tar xvz -C data
```

## Usage
* To run on a single task, use `babi_runner.py` with `-t` followed by task's id. For example,   
```
python babi_runner.py -t 1
```
The output will look like:
```
Using data from data/tasks_1-20_v1-2/en
Train and test for task 1 ...
1 | train error: 0.876116 | val error: 0.75
|===================================               | 71% 0.5s
```
* To run on 20 tasks:
```
python babi_runner.py -a
```
* To train using all training data from 20 tasks, use the joint mode:
```
python babi_runner.py -j
```

## Question Answering Demo
* In order to run the Web-based demo using the pretrained model `memn2n_model.pklz` in `trained_model/`, run:
```
python -m demo.qa
```

* Alternatively, you can try the console-based demo:
```
python -m demo.qa -console
```

* The pretrained model `memn2n_model.pklz` can be created by running:
```
python -m demo.qa -train
```

* To show all options, run `python -m demo.qa -h`

## Benchmarks
See the results [here](https://github.com/vinhkhuc/MemN2N-babi-python/tree/master/bechmarks).

### Author
Vinh Khuc

### Future Plans
* Port to TensorFlow/Keras
* Support Python 3

### References
* Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus, 
  "[End-To-End Memory Networks](http://arxiv.org/abs/1503.08895)",
  *arXiv:1503.08895 [cs.NE]*.