#!/usr/bin/env python3
"""Run a command template N times with up to P parallel workers.

Each job's stdout+stderr is saved to `outdir/job_<i>.log`.

Example:
  python scripts/run_scenes_parallel.py --cmd "./gen_scenes.sh {i}" --total 100 --parallel 8

The command template must include `{i}` which will be replaced by the job index (0..total-1).
"""
import argparse
import concurrent.futures
import os
import subprocess
import time
import sys


def run_job(cmd: str, idx: int, outdir: str) -> tuple[int, float]:
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, f"job_{idx}.log")
    start = time.time()
    # Use shell=True so users can pass complex commands; cmd is user-provided.
    with open(log_path, "wb") as f:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # Stream output to log file (binary) so encoding issues are avoided
        for chunk in iter(lambda: proc.stdout.read(4096), b""):
            f.write(chunk)
        proc.wait()
        rc = proc.returncode
    duration = time.time() - start
    return rc, duration


def main():
    p = argparse.ArgumentParser(description="Run a command template N times with P parallel workers")
    p.add_argument("--cmd", required=True, help="Command template, must contain '{i}' for job index")
    p.add_argument("--total", type=int, required=True, help="Total number of jobs to run")
    p.add_argument("--parallel", type=int, default=os.cpu_count(), help="Max concurrent jobs (default: CPU count)")
    p.add_argument("--outdir", default="parallel_runs", help="Directory to store per-job logs")
    p.add_argument("--stop-on-failure", action="store_true", help="Stop submitting new jobs if any job fails")
    args = p.parse_args()

    if "{i}" not in args.cmd:
        print("Error: --cmd must include the '{i}' placeholder", file=sys.stderr)
        sys.exit(2)

    total = args.total
    max_workers = max(1, args.parallel)

    print(f"Running {total} jobs with up to {max_workers} parallel workers")
    job_indices = range(total)

    future_to_idx = {}
    results = {}
    stopped = False

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        # submit all jobs; ThreadPoolExecutor will ensure only max_workers run concurrently
        for i in job_indices:
            if stopped:
                break
            cmd = args.cmd.format(i=i)
            fut = ex.submit(run_job, cmd, i, args.outdir)
            future_to_idx[fut] = i

        # iterate as jobs finish
        for fut in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                rc, dur = fut.result()
                results[idx] = (rc, dur)
                status = "OK" if rc == 0 else f"FAIL(code={rc})"
                print(f"Job {idx}: {status}  ({dur:.1f}s)  -> {os.path.join(args.outdir, f'job_{idx}.log')}")
                if rc != 0 and args.stop_on_failure:
                    print("Stopping submission of further jobs due to failure (stop-on-failure)")
                    stopped = True
            except Exception as e:
                print(f"Job {idx}: EXCEPTION {e}")
                results[idx] = (None, None)
                if args.stop_on_failure:
                    stopped = True

    # Summary
    succeeded = sum(1 for v in results.values() if v[0] == 0)
    failed = sum(1 for v in results.values() if v[0] not in (0, None))
    errored = sum(1 for v in results.values() if v[0] is None)
    print(f"Summary: {succeeded} succeeded, {failed} failed, {errored} errored (logs in {args.outdir})")


if __name__ == "__main__":
    main()
