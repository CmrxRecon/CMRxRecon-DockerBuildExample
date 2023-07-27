#/bin/bash
# check the enviroment and get the sumbmitter information,
# DO NOT EDIT THIS LINE
set -e
nvidia-smi
cd /output_dir
python /before_run.py

# Run your inference code and output the result to /output_dir
python /app/inference.py /input_dir /output_dir

# DO NOT EDIT THIS LINE
python /after_run.py
