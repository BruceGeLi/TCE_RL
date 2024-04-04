# Author: Ge Li, ge.li@kit.edu
# Description: Alias for Horeka HPC experiments, I used these alias to run experiments quickly on Horeka HPC.
# Usage: source horeka_alias.sh
# Note: You can add and adapt your own alias / path here, and source this file in your .bashrc file to use these alias commands.

# Horeka: https://www.scc.kit.edu/en/services/horeka.php

alias CD='cd'
alias LS='ls'
alias GIT='git'

# Slurm watch alias
alias wa='watch -n 20 squeue'
alias was='watch -n 20 squeue --start'
alias keep='watch -n 600 squeue'
alias sfree='sinfo_t_idle'
alias sgpu='salloc -p accelerated -n 1 -t 120 --mem=20000 --gres=gpu:1'
alias scpu='salloc -p cpuonly -n 1 -t 120 --mem=20000'


# cd alias
alias cdresult='cd ~/projects/tce/MPRL/mprl/result'
alias cdconfig='cd ~/projects/tce/MPRL/mprl/config'
alias cdmprl='cd ~/projects/tce/MPRL/mprl'
alias cds='cd ~/projects/tce/mprl_exp_result/slurmlog'

# Git alias
alias gp='cd ~/projects/tce/MPRL && git pull'
alias gpl='cd ~/projects/tce/MPRL && git pull && git log'
alias gpf='cd ~/projects/tce/fancy_gymnasium && git pull'
alias grc='cdmprl && python check_git_repos.py'

# Env alias
alias vb='cd ~/ && vim .bashrc'
alias ss='cd ~/ && source .bashrc && conda activate tce'

# Exp
alias runexp='cdmprl && python mp_exp.py'

## TCE BOX RANDOM INIT
alias box_random_tce_entire='runexp ./config/box_push_random_init/tce/entire/horeka.yaml   -o -s'

## BBRL BOX RANDOM INIT
alias box_random_bbrl_entire='runexp ./config/box_push_random_init/bbrl/entire/horeka.yaml   -o -s'

## BBRL Metaworld
alias meta_bbrl_prodmp_entire='runexp ./config/metaworld/bbrl/entire/horeka.yaml   -o -s'

## TCE Metaworld
alias meta_tce_entire='runexp ./config/metaworld/tce/entire/horeka.yaml   -o -s'

## BBRL Table tennis 4d ProDMP
alias tt4d_bbrl_prodmp='runexp ./config/table_tennis_4d/bbrl/entire/horeka.yaml   -o -s'

## TCE Table tennis 4d ProDMP
alias tt4d_tce_prodmp='runexp ./config/table_tennis_4d/tce/entire/horeka.yaml   -o -s'

## TCE Hopper Jump ProDMP
alias hopper_tce='runexp ./config/hopper_jump/tce/entire/horeka.yaml   -o -s'