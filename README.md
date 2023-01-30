# Demystifying Self-supervised Trojan Attacks



This is the code implementation (pytorch) for our paper:  
[Demystifying Self-supervised Trojan Attacks](https://arxiv.org/abs/2210.07346)

We design and evaluate CTRL,  an extremely simple self-supervised trojan attack. By polluting
a tiny fraction of training data (≤ 1%) with indistinguishable
poisoning samples, CTRL causes any trigger-embedded input
to be misclassified to the adversary’s desired class with a high
probability (≥ 99%) at inference. More importantly, through
the lens of CTRL, we study the mechanisms underlying self-supervised trojan attacks. With both empirical and analytical
evidence, we reveal that the representation invariance property
of SSL, which benefits adversarial robustness, may also be the
very reason making SSL highly vulnerable to trojan attacks.



## Screenshot
![screenshot](https://github.com/CCCjiang/CTRL/blob/master/imgs/training.jpg)





## Quick Start


1. Train a clean model:  
    e.g. `SimCLR` with `ResNet18` on `CIFAR10` 
    ```python3
    python main_train.py --dataset cifar10 --mode normal --method simclr --threat_model our --channel 1 2 --trigger_position 15 31 --poison_ratio 0.01 --lr 0.06 --wd 0.0005 --magnitude 100.0 --poisoning --epochs 800 --gpu 0 --window_size 32 --trial clean
    ```

2. Test backdoor attack:  
    e.g. `SimCLR` with `ResNet18` on `CIFAR10`
    ```python3
    python main_train.py --dataset cifar10 --mode frequency --method simclr --threat_model our --channel 1 2 --trigger_position 15 31 --poison_ratio 0.01 --lr 0.06 --wd 0.0005 --magnitude 100.0 --poisoning --epochs 800 --gpu 0 --window_size 32 --trial test 
    ```


## Todo List
1. SimSiam
2. Defense
3. Linear Evaluation
4. Test and find bugs

## License
This code has a GPL-style license.

## Cite our paper
```
@article{li2022demystifying,
  title={Demystifying Self-supervised Trojan Attacks},
  author={Li, Changjiang and Pang, Ren and Xi, Zhaohan and Du, Tianyu and Ji, Shouling and Yao, Yuan and Wang, Ting},
  journal={arXiv preprint arXiv:2210.07346},
  year={2022}
}
```