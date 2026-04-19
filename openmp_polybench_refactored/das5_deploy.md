# DAS-5 Deployment Guide — OpenMP PolyBench

Target: `fs0.das5.cs.vu.nl` (VU site, partition `defq`, nodes `node[001-068]`)
Hardware: dual 8-core Xeon E5-2630-v3 (Haswell), 64 GB, 16 physical cores
Policy: max 15 min jobs daytime; long jobs off-hours (>18:00) and weekends

---

## 0. Pre-deployment audit (local)

Known verification failures from local LARGE CSV (20260418):

| Benchmark | Broken strategies | Action |
|---|---|---|
| cholesky | threads_static, tiled, tasks | quarantined in XL slurm |
| 3mm | tasks (>=4 threads) | quarantined in XL slurm |
| nussinov | tiled | quarantined in XL slurm |

`SKIP_BROKEN=1` (default in new XL slurm) filters these rows from the CSV.
To include them anyway: `SKIP_BROKEN=0 sbatch ...`

---

## 1. SSH setup and repository import

From your local machine:

```bash
# a) Ensure an ssh key exists; create one if missing
ls ~/.ssh/id_ed25519.pub 2>/dev/null || ssh-keygen -t ed25519 -C "das5-openmp"

# b) Push your public key to DAS-5 (replace USER with your account)
ssh-copy-id USER@fs0.das5.cs.vu.nl

# c) Optional: add a Host entry for convenience
cat >> ~/.ssh/config <<'EOF'
Host das5
    HostName fs0.das5.cs.vu.nl
    User USER
    ServerAliveInterval 60
    ForwardAgent yes
EOF
chmod 600 ~/.ssh/config

# d) Log in
ssh das5
```

Agent-forwarded auth (`ForwardAgent yes`) lets you `git clone` a private repo
on DAS-5 using the key that lives on your laptop. The key never lands on the
cluster disk.

### On DAS-5: clone the repo

```bash
# Working directory
cd $HOME

# Option A — git clone with your forwarded agent (private repo)
git clone git@github.com:TheSpawnal/Julia_versus_OpenMP.git
# OR public https:
# git clone https://github.com/TheSpawnal/Julia_versus_OpenMP.git

# The XL slurm script expects the project at $HOME/openmp_polybench_refactored
# so either:
ln -s Julia_versus_OpenMP/opmp_std_3/openmp_polybench_refactored \
      openmp_polybench_refactored

# OR move the subtree:
# mv Julia_versus_OpenMP/opmp_std_3/openmp_polybench_refactored ~/
```

If you prefer not to use GitHub, rsync from your laptop:

```bash
# from local machine
rsync -avz --exclude='results/' --exclude='obj/' --exclude='*.o' \
    /path/to/opmp_std_3/openmp_polybench_refactored/ \
    das5:~/openmp_polybench_refactored/
```

---

## 2. Environment and build on DAS-5

```bash
ssh das5
cd ~/openmp_polybench_refactored

# Make sure module env is active for this shell
source /etc/bashrc
source /etc/profile.d/lmod.sh

# Verify gcc version (DAS-5 uses Rocky Linux + OpenHPC; gcc from module)
module avail 2>&1 | grep -iE "gnu|gcc" | head
# Example: module load gnu12  (if needed; default gcc is usually fine)

gcc --version   # must be >= 7 for full OpenMP 4.5 support

# Build for Haswell (E5-2630-v3) — NOT native
# Building "native" on the headnode vs a compute node can differ.
make clean
make das5

# Verify all 6 binaries built
ls -lh benchmark_*
```

The Makefile `das5` target uses `-march=haswell -mavx2 -mfma`, which is stable
across all VU compute nodes and avoids the "Illegal instruction" footgun of
`-march=native` when the headnode CPU differs from compute nodes.

---

## 3. Preflight check

Copy the two new files to DAS-5:

```bash
# From local:
scp preflight_check.sh das5:~/openmp_polybench_refactored/
scp das5_extralarge.slurm das5:~/openmp_polybench_refactored/slurm/

# On DAS-5:
chmod 700 ~/openmp_polybench_refactored/preflight_check.sh
chmod 600 ~/openmp_polybench_refactored/slurm/das5_extralarge.slurm

cd ~/openmp_polybench_refactored
bash preflight_check.sh
```

Expect: toolchain OK, 4-thread OpenMP smoke test PASS, 6/6 binaries present,
MEDIUM sanity run with expected FAILs only on quarantined strategies.

---

## 4. Submit XL jobs (off-hours)

```bash
cd ~/openmp_polybench_refactored

# All six benchmarks — submit to start at 22:00 tonight
sbatch --begin=22:00 slurm/das5_extralarge.slurm all

# Or weekend start
sbatch --begin=saturday slurm/das5_extralarge.slurm all

# Monitor
squeue -u $USER
scontrol show job <JOBID>

# After completion
ls -lh results/*EXTRALARGE*.csv
tail -40 openmp_XL_*.out
```

Single-benchmark submissions if you want to stage:

```bash
sbatch --begin=22:00 slurm/das5_extralarge.slurm 2mm
sbatch --begin=22:30 slurm/das5_extralarge.slurm 3mm
sbatch --begin=23:00 slurm/das5_extralarge.slurm cholesky
sbatch --begin=23:30 slurm/das5_extralarge.slurm correlation
sbatch --begin=00:00 slurm/das5_extralarge.slurm heat3d
sbatch --begin=00:30 slurm/das5_extralarge.slurm nussinov
```

---

## 5. Retrieve results

```bash
# From local
rsync -avz das5:~/openmp_polybench_refactored/results/ \
          ./das5_openmp_results/
rsync -avz das5:~/openmp_polybench_refactored/openmp_XL_*.out \
          ./das5_openmp_logs/
```

---

## Security notes

- No credentials embedded in scripts.
- `ForwardAgent yes` keeps SSH keys on your laptop only.
- Slurm script `chmod 600` so other cluster users cannot read env vars.
- `--exclusive` on the XL job prevents other users from co-allocating on the
  same node, which is both correct for timing and avoids information leak
  through `/proc`.
- The cluster headnodes are multi-tenant — never run heavy work there, the
  preflight explicitly refuses to run on a node whose hostname matches `node*`.