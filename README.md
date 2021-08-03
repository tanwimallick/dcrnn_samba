# Setup Environment
username@sm-01:~$ export PATH=$PATH:/opt/sambaflow/bin;export OMP_NUM_THREADS=1;source /opt/sambaflow/venv/bin/activate

# Commands
python dcrnn_supervisor.py compile -b=1 --pef-name=‘dcrnn.pef’ --output-folder=./pef
