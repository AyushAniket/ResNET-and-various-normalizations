## Assignment 1 @ COL870 IIT Delhi
# ResNET-and-various-normalizations
Implementation of Residual Networks as in paper https://arxiv.org/abs/1512.03385 for Image Classification of CIFAR 10 dataset with various normalization schemes.

For training the ResNET model with 'n' layers and desired normalization scheme, run the following command:
~~~
python3 train_cifar.py --normalization [ bn | in | bin | ln | gn | nn | torch_bn] --data_dir <directory_containing_data> --output_file <path to the trained model> --n [1 |  2 | 3 ] 
~~~

For testing the trained model :
~~~
python3 test_cifar.py -model_file <path to the trained model> --normalization [ bn | in | bin | ln | gn | nn | inbuilt ] --n [ 1 |  2 | 3  ] --test_data_file <path to a csv with each line representing an image> --output_file <file containing the prediction in the same order as in the input csv>
~~~
