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
    local job_name="indir_$*"
    
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
#SBATCH --cpus-per-task=${16}
#SBATCH --array=0-2
#SBATCH -o ./reports/%x_%A_%a.out 

cd ..
python3 R_autoex.py \$SLURM_ARRAY_TASK_ID $@
" > "$slurm_file"

    # Submit the job if running is enabled
    if $run; then
       sbatch "$slurm_file"
    fi
    
}
run=true
folds_eva=100

cv_criterion=2 #"min"2
sharpness=100
individual_action_effect=0
within_team_effect=0
only_cv=0
new_cov=1
new_GEE=1
new_uti=1
cv_seed=None
only_states=1
state_combine_friends=0
history_length=1
transition_state_type='weekly'
delete_week_end=0
include_weekend_indicator=0
cv_in_training=0


optimal_GEE=0
combine_actions=1
refit=0
for setting in "ctn0"; do #setting used in the simulation section: "ctn0" "r_autoex_s_exsubject"
    for accelerate_method in "batch_processing"; do # "split_clusters"
        for corr_type in "r_autoex_s_exsubject"; do  #'s_exsubject'"r_autoex_s_exsubject""r_autoex_s_exsubject""r_autoex_s_exsubject" 'uncorrelated'   "rs_autoexsubject" # #
            if [[ $setting == ctn* ]]; then
                basis_type_list="polynomial"
            elif [[ $setting == tab* ]]; then
                basis_type_list="one_hot" #rbf
            fi
            if [ "$only_cv" -eq 1 ]; then
                nthread=5
            else
                nthread=5
            fi
            if [ "$corr_type" == 'r_ex' ]; then
                sd_cluster_ex_noise=0.5
                sd_white=0
                ctn_state_sd=0.5
                state_ex_noise=0
                method_list="independent random" #exchangeable 
            elif [ "$corr_type" == 'r_autoex' ]; then
                sd_cluster_ex_noise=2
                sd_white=0.5
                ctn_state_sd=0.5
                state_ex_noise=0
                method_list="autoex" #autoex_exsubject independent"
            elif [ "$corr_type" == 'r_autoexsubject' ]; then
                sd_cluster_ex_noise=2
                sd_white=0.8
                ctn_state_sd=0.5
                state_ex_noise=0
                method_list="autoexsubject independent"
            elif [ "$corr_type" == 'rs_autoexsubject' ]; then
                sd_cluster_ex_noise=0.5
                sd_white=0.5
                ctn_state_sd=0.5
                state_ex_noise=0
                method_list="autoexsubject independent"
            elif [ "$corr_type" == "r_autoex_s_exsubject" ]; then
                sd_cluster_ex_noise=2
                sd_white=0.5
                ctn_state_sd=0
                state_ex_noise=0.5
                method_list="independent" # random  autoex_exsubject  action=1 action=0    
            elif [ "$corr_type" == 's_exsubject' ]; then
                sd_cluster_ex_noise=6
                sd_white=0.5
                ctn_state_sd=0.5
                state_ex_noise=0
                method_list="independent exchangeable_subjects" #   random 
            elif [ "$corr_type" == 'uncorrelated' ]; then
                sd_cluster_ex_noise=0
                sd_white=0
                ctn_state_sd=0.5
                state_ex_noise=0
                method_list="independent" #random
            fi
            for gamma_1 in "0"; do
                for gamma_2 in "0"; do # "0" "0.3"
                    for basis in $basis_type_list; do
                        for num_teams in 5; do  #10 20
                            num_teams_eva=1 #${num_teams}
                            for num_weeks in 1; do #  1000
                                for horizon in 5; do #25 50
                                
                                    horizon_eva=1000
                                    
                                    for num_weeks_eva in 1; do #  20
                                        for team_size in 7; do #50
                                            for method in $method_list; do    
                                                if [ "$method" == "independent" ]; then
                                                    num_batches=1
                                                else
                                                    num_batches=1
                                                fi
                                                product=$((num_weeks_eva * horizon))
                                                if [ "$product" -gt 3000 ]; then
                                                    mem=40
                                                else
                                                    mem=5
                                                fi
                                                for cv_loss in "tdssq"; do #"init_Q" "kerneldist" 
                                                    for include_Mit in 1; do
                                                        write_slurm ${num_teams} ${team_size} ${num_weeks} ${method} ${sd_cluster_ex_noise} ${sd_white} ${num_weeks_eva} ${folds_eva} ${basis} ${cv_criterion} ${setting} ${ctn_state_sd} ${horizon} ${gamma_1} ${gamma_2} ${nthread} ${sharpness} ${individual_action_effect} ${within_team_effect} ${only_cv} ${num_teams_eva} ${corr_type} ${num_batches} ${cv_loss} ${accelerate_method} ${new_cov} ${new_GEE} ${new_uti} $cv_seed $include_Mit $only_states $state_combine_friends $history_length $transition_state_type $delete_week_end $horizon_eva $include_weekend_indicator $cv_in_training $state_ex_noise $optimal_GEE $combine_actions $refit
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
