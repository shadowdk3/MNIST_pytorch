# MNIST 

- pixel: 28*28

## train
```python main.py --batch-size 128 --epochs 10 --save-model --model-name mnist_linear_b128_e40.pt```

## eval
```python predict.py --eval /media/slam/2BAA4C7433C20D90/data/MNIST/eval/ --model models/mnist_linear_b128_e20.pt```

## result 
- b: batch size
- e: epoch
  
### linear

- LinearNet
  
| model      | Eval Accuracy | Train Accuracy |
| ----------- | ----------- | ----------- |
| mnist_linear_b128_e10.pt   | 80%       | 97% |
| mnist_linear_b128_e20.pt   | 80%       | 98% |
| mnist_linear_b128_e40.pt   | 90%       | 98% |
| mnist_linear_b128_e50.pt   | 90%       | 98% |
| mnist_linear_b128_e100.pt  | 90%       | 98% |

### CNN

- CNNNet
  
| model      | Eval Accuracy | Train Accuracy |
| ----------- | ----------- | ----------- |
| mnist_cnn_b128_e10.pt   | 100%       | 98% |