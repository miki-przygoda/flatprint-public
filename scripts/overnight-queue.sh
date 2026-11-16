#!/usr/bin/env bash
# Run several generation configs one after another (one model on the GPU at a time), each through overnight.sh,
# from a detached tmux session so the queue outlives the terminal.
#
#   scripts/overnight-queue.sh [--hours H | --until HH:MM] CONFIG [CONFIG ...]
#
#   --hours H   a budget for the whole queue: each config gets what is left of it (overnight.sh --hours), and
#               no new image starts once it is spent (the one in progress finishes)
#   --until HH:MM  the same, ending at the next HH:MM
#   A config whose run is already going (its fp-<config> session exists) is waited for, not started again.
#
#   tmux attach -t fp-queue      watch the queue (each run also has its own fp-<config> session and data/runs/ dir)
#   cat data/runs/latest/status  the run in progress
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--job" ]]; then   # inside tmux
    shift
    end=$1; shift                      # epoch seconds, 0 = no budget
    for config in "$@"; do
        session="fp-$(basename "$config" .toml)"
        budget=()
        if [[ $end -gt 0 ]]; then
            left=$(( end - $(date +%s) ))
            if [[ $left -le 60 ]]; then echo "$(date '+%H:%M:%S') budget spent, skipping $config"; continue; fi
            ms=$(( left * 1000 / 3600 ))       # hours in thousandths, plain bash arithmetic
            budget=(--hours "$(( ms / 1000 )).$(printf '%03d' $(( ms % 1000 )))")
        fi
        if tmux has-session -t "$session" 2>/dev/null; then
            echo "$(date '+%H:%M:%S') $config is already running: waiting for it"
        else
            echo "$(date '+%H:%M:%S') starting $config ${budget[*]+${budget[*]}}"
            scripts/overnight.sh "$config" ${budget[@]+"${budget[@]}"}
        fi
        while tmux has-session -t "$session" 2>/dev/null; do sleep 30; done
        echo "$(date '+%H:%M:%S') finished $config: $(grep -E '^(state|summary)=' data/runs/latest/status | tr '\n' ' ')"
    done
    echo "$(date '+%H:%M:%S') queue done"
    exit 0
fi

end=0
if [[ "${1:-}" == "--until" ]]; then
    end=$(date -j -f "%H:%M" "$2" +%s); [[ $end -le $(date +%s) ]] && end=$(( end + 86400 )); shift 2
elif [[ "${1:-}" == "--hours" ]]; then
    whole=${2%%.*}; frac=${2#*.}; [[ "$frac" == "$2" ]] && frac=0
    end=$(( $(date +%s) + whole * 3600 + 10#${frac:0:1} * 360 )); shift 2   # H or H.d hours
fi
[[ $# -ge 1 ]] || { sed -n '2,13p' "$0"; exit 2; }
for config in "$@"; do [[ -f "$config" ]] || { echo "no such config: $config" >&2; exit 2; }; done
tmux has-session -t fp-queue 2>/dev/null && { echo "a queue is already running: tmux attach -t fp-queue" >&2; exit 1; }
tmux new-session -d -s fp-queue "$0 --job $end $* ; echo; echo '(queue finished; this window stays open)'; read"
echo "queued $#: $*$([[ $end -gt 0 ]] && echo " (until $(date -r "$end" '+%H:%M'))")"
echo "  watch: tmux attach -t fp-queue"
