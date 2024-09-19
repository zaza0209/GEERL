#!/bin/bash
cd bash_files
# 1: seed, 2: num_teams; 3: team_size; 4: num_weeks; 5: method; 6: sd_cluster_ex_noise
# 7: sd_white; 8: num_weeks_eva; 9: folds_eva; 10: basis; 11: cv_criterion 
# 12: setting; 13: ctn_state_sd; 14: horizon; 15:gamma_1; 16: gamma_2; 17:nthread; 18: sharpness
# 19: individual_action_effect; 20: within_team_effect;  21: only_cv; 22:num_teams_eva; 
# 23: corr_type; 24: num_batches; 25: cv_loss; 26:accelerate_method
# mem: if cv and nthreads=5, assign 30g, if fit and eva nthread=5, assign 15
write_slurm() {
    # Save the original IFS
    local old_ifs="$IFS"
    # Set IFS to underscore for concatenation
    IFS='_'
    
    # Construct the job name from the function arguments
    local job_name="valuecompare_$*"
    
    # Restore the original IFS
    IFS="$old_ifs"
    
    # Ensure the job name is not empty
    if [ -z "$job_name" ]; then
        echo "Error: Job name is empty"
        return
    fi
    
    # Construct the slurm script filename
    local slurm_file="job_${job_name}.slurm"
    
    # Check for any potential errors in filename
    if [ -z "$slurm_file" ]; then
        echo "Error: Slurm file name is empty"
        return
    fi

# mem cv: 40 else 20
    echo "#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --partition=medium
#SBATCH --mem=${mem}g
#SBATCH --cpus-per-task=${nthread}
#SBATCH --array=0-49
#SBATCH -o ./reports/%x_%A_%a.out 

cd ..
python3 value_comparison.py \$SLURM_ARRAY_TASK_ID $@
" > "$slurm_file"

    # Submit the job if running is enabled
    if $run; then
       sbatch "$slurm_file"
    fi
    
}
run=true
folds_eva=100
sharpness=100
individual_action_effect=0
within_team_effect=0
only_cv=0
only_states=1
history_length=1
transition_state_type='weekly'
delete_week_end=0
include_weekend_indicator=0
refit=1
nthread=5
for setting in "ctn0"; do #setting used in the simulation section: "ctn0" "r_autoex_s_exsubject"
    sd_cluster_ex_noise=0
    sd_white=0
    ctn_state_sd=0.5
    state_ex_noise=0
    for gamma_1 in "0"; do
        for gamma_2 in "0"; do  
            num_teams_eva=1  
            horizon_eva=1000
            for num_weeks_eva in 1; do #  20
                product=$((num_weeks_eva * horizon_eva))
                if [ "$product" -gt 3000 ]; then
                    mem=40
                else
                    mem=5
                fi
                for Q_sd in 1 2; do
                    for include_Mit in 1; do
                        write_slurm $sd_cluster_ex_noise $sd_white $num_weeks_eva $folds_eva $setting $ctn_state_sd $gamma_1 $gamma_2 $nthread $sharpness $individual_action_effect $within_team_effect $num_teams_eva $include_Mit $only_states $transition_state_type $delete_week_end $horizon_eva $include_weekend_indicator $state_ex_noise $Q_sd $refit 
                    done
                done
            done
        done
    done
done
