#!/usr/bin/env bash
# Phase A queue: runs the remaining AMP/DAP steps on one build, ONE AT A
# TIME, after every job recorded in PIDS.txt has exited.
#
#   nohup bash scripts/run_phaseA_queue.sh > /dev/null 2>&1 &
#
# Order (each step starts only when the previous one has ended):
#   1. wait until every pid=<n> listed in $PIDS_FILE has exited (the
#      nested-CV job A14) and no run_a1*.py driver of this code tree is
#      running; the queue never runs concurrently with them
#   2. A13 single LOO, classifier seeds 1, 2, 3, 4
#   3. A15 tables (reads the A13 / A14 result files; no model)
#   4. A16 ablations
#   5. A17 classic meta-learning
#   6. A18 bagging
# A failed step is logged and the queue goes on with the next step (the
# steps do not need each other's output, except A15, which simply leaves
# out the tables whose inputs are missing). The exit code of the queue is
# the number of failed steps. Every driver checks the build with the rule
# gate guard itself and can be started again: it resumes from its
# checkpoints.
#
# NOTE on step 6: with the default `--undefined-camin stop`, A18 exits
# with code 3 without fitting anything when a sampled sub-group has no
# CAmin (plan OQ7 / amendment AM12 is open). Set A18_ARGS, e.g.
#   A18_ARGS="--undefined-camin drop_instance"
# only after the rule has been decided.
#
# Environment (defaults in brackets):
#   BUILD [waiver_main]   N_JOBS [4]   PYTHON [python]
#   RUNS_ROOT [<repo>/runs]
#   PIDS_FILE [$RUNS_ROOT/PIDS.txt]   QUEUE_LOG [$RUNS_ROOT/queue.log]
#   DATA_ROOT [$CDM_DATA_ROOT, else <repo>/data]  (exported as CDM_DATA_ROOT)
#   POLL_SECONDS [60]     A18_ARGS []   EXTRA_ARGS [] (added to every driver)
#   QUEUE_DRY_RUN [0]     1 = log the commands, run nothing (still waits)
#   WAIT_FOR_DRIVERS [1]  0 = wait for the recorded pids only (for the tests)
set -u

BUILD="${BUILD:-waiver_main}"
N_JOBS="${N_JOBS:-4}"
PYTHON="${PYTHON:-python}"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-$CODE_DIR/runs}"
PIDS_FILE="${PIDS_FILE:-$RUNS_ROOT/PIDS.txt}"
QUEUE_LOG="${QUEUE_LOG:-$RUNS_ROOT/queue.log}"
DATA_ROOT="${DATA_ROOT:-${CDM_DATA_ROOT:-$CODE_DIR/data}}"
POLL_SECONDS="${POLL_SECONDS:-60}"
A18_ARGS="${A18_ARGS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
QUEUE_DRY_RUN="${QUEUE_DRY_RUN:-0}"
WAIT_FOR_DRIVERS="${WAIT_FOR_DRIVERS:-1}"

mkdir -p "$RUNS_ROOT" "$(dirname "$QUEUE_LOG")"
export CDM_DATA_ROOT="$DATA_ROOT"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$QUEUE_LOG"; }

# One queue at a time.
exec 9> "$RUNS_ROOT/queue.lock"
if ! flock -n 9; then
    log "QUEUE NOT STARTED: another queue holds $RUNS_ROOT/queue.lock"
    exit 99
fi

log "QUEUE START pid=$$ build=$BUILD n_jobs=$N_JOBS code=$CODE_DIR dry_run=$QUEUE_DRY_RUN a18_args='$A18_ARGS'"

# ---- 1. wait for the recorded jobs -----------------------------------------------
recorded_pids() {
    # every pid=<n> in the PIDS file, except this shell
    [ -f "$PIDS_FILE" ] || return 0
    grep -o 'pid=[0-9][0-9]*' "$PIDS_FILE" | cut -d= -f2 | sort -u | grep -v -x "$$" || true
}

alive_pids() {
    local p state out=""
    for p in $(recorded_pids); do
        # alive = the process exists and is not a zombie (a finished job whose
        # parent has not collected it yet counts as exited)
        state="$(ps -o stat= -p "$p" 2>/dev/null | tr -d ' ')"
        if [ -n "$state" ] && [ "${state#Z}" = "$state" ]; then out="$out $p"; fi
    done
    echo "$out"
}

other_drivers() {
    # driver scripts of this code tree that are running (not started by this queue)
    [ "$WAIT_FOR_DRIVERS" = "1" ] || return 0
    pgrep -f 'run_a1[0-9]_[a-z0-9_]*\.py' 2>/dev/null \
        | grep -v -x "$$" | tr '\n' ' ' || true
}

last_note=""
while :; do
    alive="$(alive_pids)"
    others="$(other_drivers)"
    if [ -z "${alive// /}" ] && [ -z "${others// /}" ]; then
        # look again after a short pause: a job may be between two processes
        sleep 2
        alive="$(alive_pids)"
        others="$(other_drivers)"
        if [ -z "${alive// /}" ] && [ -z "${others// /}" ]; then break; fi
    fi
    note="waiting: recorded pids alive [${alive# }], other drivers running [${others% }]"
    if [ "$note" != "$last_note" ]; then log "$note"; last_note="$note"; fi
    sleep "$POLL_SECONDS"
done
log "no recorded job is running; starting the steps"

# ---- 2.-6. the steps, one at a time ---------------------------------------------------
failed=0
run_step() {
    local name="$1"; shift
    log "STEP START $name: $*"
    if [ "$QUEUE_DRY_RUN" = "1" ]; then
        log "STEP DRY-RUN $name (not executed)"
        return 0
    fi
    local t0 rc
    t0=$(date +%s)
    ( cd "$CODE_DIR" && "$@" ) >> "$RUNS_ROOT/queue_stdout.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        log "STEP DONE $name in $(( $(date +%s) - t0 )) s"
    elif [ $rc -eq 3 ] && [ "$name" = "A18 bagging" ]; then
        failed=$((failed + 1))
        log "STEP BLOCKED $name: exit code 3 = sub-groups without CAmin and no rule chosen (plan OQ7 / AM12 open); nothing was fitted; see a18_blocked.json in the step's folder"
    else
        failed=$((failed + 1))
        log "STEP FAILED $name: exit code $rc after $(( $(date +%s) - t0 )) s (see $RUNS_ROOT/queue_stdout.log and the step's run.log)"
    fi
    return 0
}

for seed in 1 2 3 4; do
    # shellcheck disable=SC2086
    run_step "A13 LOO seed $seed" "$PYTHON" scripts/run_a13_loo.py --build "$BUILD" \
        --seed "$seed" --n-jobs "$N_JOBS" --runs-root "$RUNS_ROOT" $EXTRA_ARGS
done
# shellcheck disable=SC2086
run_step "A15 tables" "$PYTHON" scripts/run_a15_tables.py --build "$BUILD" \
    --runs-root "$RUNS_ROOT" $EXTRA_ARGS
# shellcheck disable=SC2086
run_step "A16 ablation" "$PYTHON" scripts/run_a16_ablation.py --build "$BUILD" \
    --n-jobs "$N_JOBS" --runs-root "$RUNS_ROOT" $EXTRA_ARGS
# shellcheck disable=SC2086
run_step "A17 meta-learning" "$PYTHON" scripts/run_a17_metalearn.py --build "$BUILD" \
    --runs-root "$RUNS_ROOT" $EXTRA_ARGS
# shellcheck disable=SC2086
run_step "A18 bagging" "$PYTHON" scripts/run_a18_bagging.py --build "$BUILD" \
    --n-jobs "$N_JOBS" --runs-root "$RUNS_ROOT" $A18_ARGS $EXTRA_ARGS

log "QUEUE END failed_steps=$failed"
exit "$failed"
