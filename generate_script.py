import os

combined_dir = 'data_supervised/single'
jobscript_dir = 'jobscripts_single'
os.makedirs(jobscript_dir, exist_ok=True)

# Template for the SLURM script
slurm_template = """#!/bin/bash
#SBATCH --job-name=supervised_bert_{name}
#SBATCH --output=outputs/job.{name}.%j.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --nodelist=cn[3,4,5,6,7,9,10,12,13,18,19]
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL

hostname
nvidia-smi
module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate NLP
cd ~/Few_Nerds
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python run_supervised.py --data_dir data_supervised/combined/{name} --model_type bert --model_name_or_path bert-base-uncased --output_dir supervised_output --output_file_name {name} --do_train --do_eval --evaluate_during_training --logging_steps 1250 --labels labels.txt --do_lower_case --overwrite_cache --overwrite_output_dir
"""

# Loop through the combined folder
for folder_name in os.listdir(combined_dir):
    full_path = os.path.join(combined_dir, folder_name)
    if os.path.isdir(full_path):
        job_script = slurm_template.format(name=folder_name)
        script_filename = os.path.join(jobscript_dir, f"{folder_name}.sbatch")
        with open(script_filename, "w") as f:
            f.write(job_script)

print(f"Job scripts created in '{jobscript_dir}'")
