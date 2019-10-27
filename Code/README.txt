Readme:

Run the file dqn_snake.py as described in the report. 

To train a network
python dqn_snake.py --add False --folder 7 --lr 0.0001 --model dqn


To test it f.e.

dqn_snake.py --add True --folder 19 --lr 0.0001 --model tiny --no_load_latest --mode play --checkpoint dqn_checkpoints_19_True_tiny_0.0001/chkpoint_tiny_8.pth.tar --record
