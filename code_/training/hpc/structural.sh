#!/bin/bash
output_dir=/share/ddomlab/sdehgha2/working-space/main/P1_pls-dataset/pls-dataset-space/PLS-Dataset/results
# Define arrays for regressor types, targets, and models
regressors=("RF" "DT" "XGBR" "NGB")
models=("mordred" "ecfp")


# Loop through each combination of regressor, target, and model
for regressor in "${regressors[@]}"; do
    for model in "${models[@]}"; do
              bsub <<EOT
          
#BSUB -n 6
#BSUB -W 10:01
#BSUB -R span[ptile=2]
#BSUB -R "rusage[mem=16GB]"
#BSUB -J "${model}_${regressor}_"  
#BSUB -o "${output_dir}/mordred_${regressor}_.out"
#BSUB -e "${output_dir}/mordred_${regressor}_.err"

source ~/.bashrc
conda activate /usr/local/usrapps/ddomlab/sdehgha2/pls-dataset-env
python ../train_structure_only.py $model \
             --regressor_type $regressor \

EOT
  done
done