#!/usr/bin/env bash
# Run a generation config in a detached tmux session, so it outlives the terminal or SSH session.
#
#   scripts/overnight.sh CONFIG [--images N] [--hours H] [--per-prompt N] [--host NAME]
#
#   --images N      stop after N new images
#   --hours H       stop starting new images after H hours (the image in progress finishes)
#   --per-prompt N  override the config's seeds.per_prompt (raise it so the budget, not the config, is the limit)
#
# Each run gets data/runs/<stamp>-<config>/ with run.log and a `status` file (state=running|done|failed).
# data/runs/latest points at the newest run. Afterwards it merges, exports the set to data/<set>/ for preview,
# and posts a macOS notification. FLATPRINT_DATA picks another data dir (default data/). Re-running the same config resumes: finished images are skipped.
#
#   tmux attach -t fp-<config>      watch it (detach with Ctrl-b d)
#   cat data/runs/latest/status     check it without attaching
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--job" ]]; then   # inside tmux
    shift
    data=$1 run_dir=$2 set_name=$3; shift 3
    export FLATPRINT_DATA="$data"
    status() { printf '%s\n' "$@" >> "$run_dir/status"; }
    export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/flatprint-cache/hf}"
    keep_awake=()
    command -v caffeinate >/dev/null && keep_awake=(caffeinate -ims)   # macOS: no idle/system sleep (on mains power)
    set +e
    ${keep_awake[@]+"${keep_awake[@]}"} uv run flatprint generate "$@" 2>&1 | tee "$run_dir/run.log"
    code=${PIPESTATUS[0]}
    { uv run flatprint merge && uv run flatprint export --set "$set_name" --to "$data/$set_name"; } >> "$run_dir/run.log" 2>&1
    post=$?
    set -e
    summary=$(tr '\r' '\n' < "$run_dir/run.log" | grep -a 'done=' | tail -1)
    # generate exits 1 when some images failed but the run itself completed; that's still "done"
    if [[ $code -le 1 && $post -eq 0 && -n "$summary" ]]; then state=done; else state=failed; fi
    status "state=$state" "ended=$(date '+%Y-%m-%d %H:%M:%S')" "exit=$code" "summary=$summary"
    command -v osascript >/dev/null && osascript -e "display notification \"$summary\" with title \"flatprint run $state\"" || true
    exit 0
fi

[[ $# -ge 1 && -f "$1" ]] || { sed -n '2,17p' "$0"; exit 2; }
config=$1; shift
args=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --images) args+=(--limit "$2"); shift 2 ;;
        --hours|--per-prompt|--host) args+=("$1" "$2"); shift 2 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

data="${FLATPRINT_DATA:-data}"
name=$(basename "$config" .toml)
session="fp-$name"
if tmux has-session -t "$session" 2>/dev/null; then
    echo "already running: tmux attach -t $session" >&2; exit 1
fi
set_name=$(uv run python -c "import sys; from flatprint.config import load_config; print(load_config(sys.argv[1]).set)" "$config")
stamp=$(date +%Y%m%d-%H%M)
run_dir="$data/runs/$stamp-$name"
mkdir -p "$run_dir"
printf '%s\n' "state=running" "started=$(date '+%Y-%m-%d %H:%M:%S')" "config=$config" "set=$set_name" \
    "args=${args[*]+${args[*]}}" "session=$session" > "$run_dir/status"
ln -sfn "$stamp-$name" "$data/runs/latest"

tmux new-session -d -s "$session" "$0" --job "$data" "$run_dir" "$set_name" "$config" ${args[@]+"${args[@]}"}
echo "started $session -> $run_dir"
echo "  watch:  tmux attach -t $session"
echo "  status: cat $run_dir/status"
