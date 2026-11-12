"""scripts/run_phaseA_queue.sh: order of the steps and the wait for the
recorded jobs. Dry run only: nothing is executed and no data is read."""
import os
import subprocess
import time

from cdm.config import CODE_ROOT

QUEUE = CODE_ROOT / "scripts" / "run_phaseA_queue.sh"


def run_queue(tmp_path, **env):
    full = dict(os.environ, RUNS_ROOT=str(tmp_path), POLL_SECONDS="1", QUEUE_DRY_RUN="1",
                WAIT_FOR_DRIVERS="0", **env)
    return subprocess.run(["bash", str(QUEUE)], env=full, capture_output=True, text=True)


def test_syntax():
    assert subprocess.run(["bash", "-n", str(QUEUE)]).returncode == 0


def test_waits_for_the_recorded_pid_then_runs_the_steps_in_order(tmp_path):
    job = subprocess.Popen(["sleep", "4"])
    (tmp_path / "PIDS.txt").write_text(
        f"2026-11-11T08:32:08Z A14 nested CV pid={job.pid} log=x\n"
        "2026-11-11T08:19:28Z A13 LOO seed 0 pid=999999999 log=y\n")
    t0 = time.time()
    p = run_queue(tmp_path)
    waited = time.time() - t0
    job.wait()
    assert p.returncode == 0, p.stderr
    assert waited >= 3.0                                  # not before the job ended
    log = (tmp_path / "queue.log").read_text().splitlines()
    assert f"recorded pids alive [{job.pid}]" in "\n".join(log)
    starts = [line.split("STEP START ")[1].split(":")[0] for line in log if "STEP START" in line]
    assert starts == ["A13 LOO seed 1", "A13 LOO seed 2", "A13 LOO seed 3", "A13 LOO seed 4",
                      "A15 tables", "A16 ablation", "A17 meta-learning", "A18 bagging"]
    i_wait = next(i for i, line in enumerate(log) if "no recorded job is running" in line)
    assert all(i > i_wait for i, line in enumerate(log) if "STEP START" in line)
    cmds = [line for line in log if "STEP START" in line]
    assert all("--build waiver_main" in c for c in cmds)
    assert "run_a13_loo.py --build waiver_main --seed 3 --n-jobs 4" in cmds[2]
    assert log[-1].endswith("QUEUE END failed_steps=0")


def test_only_one_queue_at_a_time(tmp_path):
    job = subprocess.Popen(["sleep", "3"])
    (tmp_path / "PIDS.txt").write_text(f"x pid={job.pid}\n")
    env = dict(os.environ, RUNS_ROOT=str(tmp_path), POLL_SECONDS="1", QUEUE_DRY_RUN="1",
               WAIT_FOR_DRIVERS="0")
    first = subprocess.Popen(["bash", str(QUEUE)], env=env)
    time.sleep(1.0)
    second = run_queue(tmp_path)
    assert second.returncode == 99
    assert first.wait() == 0
    job.wait()
    assert "QUEUE NOT STARTED" in (tmp_path / "queue.log").read_text()
