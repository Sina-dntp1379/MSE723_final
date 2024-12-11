#!/bin/bash
output_dir=/share/ddomlab/sdehgha2/working-space/main/informatics/MSE723_final/HpcOut

# Correctly define models and numerical features
models_to_run=("RF" "ElasticNet" "DT" "XGBR" "NGB")
# numerical_feat_list=("selected features")

        for model in "${models_to_run[@]}"; do
            # for feats in "${numerical_feat_list[@]}"; do
                bsub <<EOT

#BSUB -n 6
#BSUB -W 15:01
#BSUB -R span[ptile=2]
##BSUB -x
#BSUB -R "rusage[mem=16GB]"
#BSUB -J "${model}_with_${feats}"
#BSUB -o "${output_dir}/${feats}_with_${model}_.out"
#BSUB -e "${output_dir}/${feats}_with_${model}_.err"

source ~/.bashrc
conda activate /usr/local/usrapps/ddomlab/sdehgha2/pls-dataset-env

python ../train_numerical_only.py --regressor_type "${model}" \
                                #   --numerical_feats "${feats}" \

conda deactivate

EOT
    done
done


