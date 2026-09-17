#!/bin/sh
# Re-run every experiment in the corrected pen, in dependency order.
#
# Stage one trains the three founding policies. Everything after it needs those policies,
# so nothing else can start until all three are done. Two jobs run at a time, because that
# is how many cores there are, with one thread each so they do not fight over them.
set -e
cd "$(dirname "$0")"

export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p logs results

echo "=== stage 1: the three founding policies"
printf '0\n1\n2\n' | xargs -P 2 -I{} sh -c \
  'python experiments/pretrain_pen.py 120 {} > logs/pretrain_{}.log 2>&1 && echo "  pretrain {} done"'

echo "=== stage 2: are the three policies comparable"
python experiments/compare_policies.py 0,1,2 2500 > logs/compare.log 2>&1
echo "  done"

echo "=== stage 3: the colony runs"
cat > logs/jobs.txt <<'EOF'
python experiments/campaign_shared.py 0 6000 > logs/shared_0.log 2>&1
python experiments/campaign_shared.py 1 6000 > logs/shared_1.log 2>&1
python experiments/campaign_shared.py 2 6000 > logs/shared_2.log 2>&1
python experiments/campaign_individual.py 0 6000 > logs/individual_0.log 2>&1
python experiments/campaign_individual.py 1 6000 > logs/individual_1.log 2>&1
python experiments/campaign_individual.py 2 6000 > logs/individual_2.log 2>&1
python experiments/breakdown.py blocked 6000 0 0 > logs/blocked_0.log 2>&1
python experiments/breakdown.py blocked 6000 1 1 > logs/blocked_1.log 2>&1
python experiments/breakdown.py blocked 6000 2 2 > logs/blocked_2.log 2>&1
python experiments/breakdown.py intrusion 6000 0 0 > logs/intrusion_0.log 2>&1
python experiments/breakdown.py intrusion 6000 1 1 > logs/intrusion_1.log 2>&1
python experiments/breakdown.py intrusion 6000 2 2 > logs/intrusion_2.log 2>&1
python experiments/breakdown.py damage 6000 0 0 > logs/damage_0.log 2>&1
python experiments/breakdown.py damage 6000 1 1 > logs/damage_1.log 2>&1
python experiments/breakdown.py damage 6000 2 2 > logs/damage_2.log 2>&1
EOF
xargs -P 2 -I{} sh -c '{} && echo "  done: {}"' < logs/jobs.txt

echo "=== stage 4: the reward comparison"
# Serial, because both modes append to the same curves file and would overwrite each other.
python experiments/sweep_rl.py offspring 0,1,2,3,4 > logs/sweep_offspring.log 2>&1
python experiments/sweep_rl.py survival 0,1,2,3,4 > logs/sweep_survival.log 2>&1
python experiments/plot_rl.py > logs/plot.log 2>&1

echo "=== all done"
