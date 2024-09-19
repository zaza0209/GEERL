#!/bin/bash
# cd /home/liyuanhu/ClusterRL/joint_learning/individual_level/bash_files
cd bash_files
# 1: seed, 2: num_teams; 3: team_size; 4: sd_cluster_ex_noise
# 5: sd_white; 6: num_weeks_eva; 7: folds_eva; 
# 8: setting; 9: ctn_state_sd; 10: horizon; 11:gamma_1; 12: gamma_2; 13:nthread; 14: sharpness
# 15: individual_action_effect; 16: within_team_effect; 17: num_teams_eva
# 18: corr_type; 19 corr_eva; 20: num_weeks; 21: only_training; 22:include_Mit; 23:use_replay_buffer
# 24:update_target_every; 25: early_stopping_patience
write_slurm() {
    # Save the original IFS
    local old_ifs="$IFS"
    # Set IFS to underscore for concatenation
    IFS='_'
    
    # Construct the job name from the function arguments
    local job_name="online_$*"
    
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
#SBATCH --partition=xlargecpu
#SBATCH --mem=5g
#SBATCH --cpus-per-task=${12}
#SBATCH --array=0
#SBATCH -o ./reports/%x_%A_%a.out 

# cd /home/liyuanhu/ClusterRL/joint_learning/individual_level
cd ..
python3 Qonline_single.py \$SLURM_ARRAY_TASK_ID $@
" > "$slurm_file"

    # Submit the job if running is enabled
    if $run; then
       sbatch "$slurm_file"
    fi
    
}

run=true

num_weeks_eva=1
folds_eva=100
sharpness=100
individual_action_effect=0
within_team_effect=0
corr_eva=0
only_training=0
if [ "$only_training" -eq 1 ]; then
    nthread=1
else
    nthread=5
fi
update_target_every=100
early_stopping_patience=1000

state_combine_friends=0
history_length=1
transition_state_type="weekly"
delete_week_end=0
include_weekend_indicator=0
train_corr_eva=0
reward_buffer_len=0
early_stop=0

for setting in "ctn0"; do #"tab2"  "ctn1" "tab3" 
    for corr_type in  "r_autoex_s_exsubject" ; do  #'r_autoex'"r_ex""r_autoex""rs_autoexsubject" # #
        if [ "$corr_type" == 'r_ex' ]; then
            sd_cluster_ex_noise=0.5
            sd_white=0
            ctn_state_sd=0.5
            state_ex_noise=0
            method_list="independent random" #exchangeable 
        elif [ "$corr_type" == 'r_autoex' ]; then
            sd_cluster_ex_noise=2
            sd_white=0.5
            ctn_state_sd=0.2
            state_ex_noise=0
            method_list="random autoex independent" #
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
            ctn_state_sd=0.2
            state_ex_noise=0.2
            method_list="autoex_exsubject independent"
        elif [ "$corr_type" == 's_exsubject' ]; then
            sd_cluster_ex_noise=2
            sd_white=0.5
            ctn_state_sd=0.2
            state_ex_noise=0
            method_list="random exchangeable_subjects independent" #  
        elif [ "$corr_type" == 'uncorrelated' ]; then
            sd_cluster_ex_noise=0
            sd_white=0
            ctn_state_sd=0.5
            state_ex_noise=0
            method_list="independent" #random
        fi
        for gamma_1 in "0"; do
            for gamma_2 in "0.3"; do # "0" "0.3"
                for num_teams in 1; do
                    num_teams_eva=1 #${num_teams}
                    for team_size in 1; do
                        for num_weeks in 1; do
                            horizon=100
                            horizon_eva=1000
                            for include_Mit in 0; do
                                for use_replay_buffer in 1; do
                                    for hidden_nodes_str in "64"; do #
                                        hidden_nodes_array=($hidden_nodes_str)
                                        hidden_nodes_lens=${#hidden_nodes_array[@]}
                                        for early_stopping_criterion in "reward"; do #"loss"
                                            write_slurm ${num_teams} ${team_size} ${sd_cluster_ex_noise} ${sd_white} ${num_weeks_eva} ${folds_eva} ${setting} ${ctn_state_sd} ${horizon} ${gamma_1} ${gamma_2} ${nthread} ${sharpness} ${individual_action_effect} ${within_team_effect} ${num_teams_eva} ${corr_type} ${corr_eva} ${num_weeks} $only_training ${include_Mit} $use_replay_buffer $early_stopping_criterion $update_target_every $early_stopping_patience $state_combine_friends $history_length $transition_state_type $delete_week_end $horizon_eva $include_weekend_indicator $train_corr_eva $reward_buffer_len $early_stop $state_ex_noise $hidden_nodes_lens "${hidden_nodes_array[@]}"
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
