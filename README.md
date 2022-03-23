# Active Learning for Image Classification Toolkit

# Requirements

- pytorch
- torchvision
- tensorboard
- tqdm
- scikit-learn
- matplotlib
- seaborn

# Running an experiment

```
python run.py \
            --model resnet \
            --optimizer sgd \
            --scheduler cosine \
            --lr 0.01 \
            --nStart 1000 \
            --nQuery 3000 \
            --nEnd 25000 \
            --batch_size 20 \
            --data CIFAR10 \
            --alg rand
```
