#############################
# screen 
#############################
screen -ls
screen -S training
screen -D -r training
screen -S training -X quit
screen -S 21700 -X quit

#############################
# path 
#############################
nvidia-smi


#############################
# shell 
#############################
screen -S jupyter
conda activate py396
cd /home/hy/project/
jupyter lab --allow-root

