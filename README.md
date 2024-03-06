# 1. Person Reidentification

- [1. Person Reidentification](#1-person-reidentification)
- [2. Environment command](#2-environment-command)
  - [2.1. Nvidia](#21-nvidia)
  - [2.2. Jupyter](#22-jupyter)
  - [2.3. screen](#23-screen)
- [3. Experiments](#3-experiments)
  - [3.1. PCB](#31-pcb)
  - [3.2. APNet](#32-apnet)

# 2. Environment command

## 2.1. Nvidia

```
nvidia-smi
```

## 2.2. Jupyter

```
screen -S jupyter
conda activate py396
cd /home/hy/project/
jupyter lab --allow-root
```

## 2.3. screen

```
screen -ls
screen -S training
screen -S training -X quit
screen -S 21700 -X quit
```

```
screen -D -r training
conda activate py396
cd /home/hy/project/
```

# 3. Experiments

## 3.1. [PCB](version/pcb) 

## 3.2. [APNet](version/apnet) 