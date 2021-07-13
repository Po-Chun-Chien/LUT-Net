# LUT-Net
An implementation of LUT-Net (a network of look-up-tables) learning procedure described in [[1]](#ref1).

## Introduction
This repository contains one of the methods we used in the [IWLS 2020](https://iwls20.cade.utah.edu/) Programming Contest.
The program supports the direct learning of a LUT-Net from training data, and the conversion from a sparse neural network into a LUT-net (not fully supported).
You may also refer to the [contest description](https://github.com/iwls2020-lsml-contest/iwls2020-lsml-contest/blob/main/contest_description.pdf), our DATE'19 paper [[3]](#ref3) (written with other contestants) and the [project page](https://po-chun-chien.github.io/projects/5.ML+LS/) for more details.

## Usage
Before running the program, please make sure you have the required pip packages installed.
```
pip3 install -r requirements.txt
```

### Learning Directly from Data
The training and validation data should be in [PLA](https://ultraespresso.di.univr.it/assets/data/espresso/espresso5.pdf) format.
You can download the contest benchmarks via this [link](https://github.com/iwls2020-lsml-contest/iwls2020-lsml-contest).
The number of inputs of each LUT can be specified with the `--lut_k` flag, and the shape of the netowrk can be specified with the `--hidden_layers` flag.
To save the learned binary decision tree model, you can write it into a pickle file with the `--save_model` flag, and/or dump it into a logic circuit in [BLIF](https://people.eecs.berkeley.edu/~alanmi/publications/other/blif.pdf) format with the `--dump_blif` flag.
Below is an example to run our program.
You can use the `--help` flag to see a more detailed usage.
```
python3 lutNet.py --train_data benchmarks/train/ex00.train.pla --valid_data benchmarks/validation/ex00.valid.pla --lut_k 6 --hidden_layers 1000,500,100 --save_model ex00_model.pk --dump_blif ex00_model.blif --verbose
```

### Conversion from a Pre-trained Sparse Neural Network
The LUT-Net we used in the contest were all converted from a pre-trained sparse neural network (NN).
Each neuron in the NN is converted into a LUT by enumerating all the possible input assignments and rounding/quantizing its activation (to 0/1).
Here we listed some guidelines for NN model training and conversion.
* The activation function of each neuron should be sigmoid function, which restricts the output to a value between 0 and 1. 
* The NN is expected to be sparse (not fully connected), since an enumeration step (exponential complexity) is required during the conversion. In practice, we apply the connection pruning technique proposed in [[3]](#ref3) so that the number of inputs of each neuron is at a reasonable quantity.

After conversion, the number of inputs of each LUT is the same as the number of inputs of each neuron.
Inevitably, there is also a minor accuracy degradation caused by the rounding errors of the sigmoid activations.
(This can be possibly solved by adding a loss function that favors the binarization of neuron outputs to during NN training.)

Unfortunately, an easy-to-use interface is currently not supported.
You may need to trace the source code and integrate your model into the flow yourself.
Below is an example to run our program.
```
python3 lutNet.py --train_data benchmarks/train/ex00.train.pla --valid_data benchmarks/validation/ex00.valid.pla --nn_model nn/pruned10_12_model/ex00_best.pth.tar --save_model ex00_model.pk --dump_blif ex00_model.blif --verbose
```

## References
<a class="anchor" id="ref1">[1]</a> S. Chatterjee, “Learning and Memorization,” in Proceedings of the International Conference on Machine Learning (PMLR), pp. 755–763, 2018.

<a class="anchor" id="ref2">[2]</a> S. Rai et al., "Logic Synthesis Meets Machine Learning: Trading Exactness for Generalization," in Proceedings of the Design, Automation and Test in Europe Conference (DATE), 2021.

<a class="anchor" id="ref3">[3]</a> S. Han, J. Pool, J. Tran, and W. J. Dally, “Learning both weights and connections for efficient neural networks,” in Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS), p. 1135–1143, 2015.

# Suggestions, Questions, Bugs, etc
You are welcome to create an [issue](https://github.com/Po-Chun-Chien/LUT-Net/issues) to make suggestions, ask questions, or report bugs, etc.
