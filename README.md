# LUT-Net
An implementation of binary decision tree with fringe-features extraction.

## Introduction
This repository contains one of the methods we used in the [IWLS 2020](https://iwls20.cade.utah.edu/) Programming Contest.
We train a binary binary decision tree with fringe-features extraction proposed in [[1]](#ref1)[[2]](#ref2), and convert it into a Boolean logic circuit.
You may also refer to the [contest description](https://github.com/iwls2020-lsml-contest/iwls2020-lsml-contest/blob/main/contest_description.pdf), our DATE'19 paper [[3]](#ref3) (written with other contestants) and the [project page](https://po-chun-chien.github.io/projects/5.ML+LS/) for more details.

## Usage
Before running the program, please make sure you have the required pip packages installed.
```
pip3 install -r requirements.txt
```

The training and validation data should be in [PLA](https://ultraespresso.di.univr.it/assets/data/espresso/espresso5.pdf) format.
You can download the contest benchmarks via this [link](https://github.com/iwls2020-lsml-contest/iwls2020-lsml-contest).
To save the learned binary decision tree model, you can write it into a pickle file with the `--save_model` flag, and/or dump it into a logic circuit in [BLIF](https://people.eecs.berkeley.edu/~alanmi/publications/other/blif.pdf) format with the `--dump_blif` flag.
Below is an example to run our program.
You can use the `--help` flag to see a more detailed usage.
```
python3 main.py --train_data benchmarks/train/ex00.train.pla --valid_data benchmarks/validation/ex00.valid.pla --save_model ex00_model.pk --dump_blif ex00_model.blif
```

## References
<a class="anchor" id="ref1">[1]</a> G. Pagallo and D. Haussler, “Boolean Feature Discovery in Empirical Learning,” Machine Learning, vol. 5, no. 1, p. 71–99, 1990.

<a class="anchor" id="ref2">[2]</a> A. L. Oliveira and A. Sangiovanni-Vincentelli, “Learning Complex Boolean Functions: Algorithms and Applications,” in Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS), p. 911–918, 1993.

<a class="anchor" id="ref3">[3]</a> S. Rai et al., "Logic Synthesis Meets Machine Learning: Trading Exactness for Generalization," in Proceedings of the Design, Automation and Test in Europe Conference (DATE), 2021.

# Suggestions, Questions, Bugs, etc
You are welcome to create an [issue](https://github.com/Po-Chun-Chien/LUT-Net/issues) to make suggestions, ask questions, or report bugs, etc.
