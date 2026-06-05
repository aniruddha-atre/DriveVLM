# Running on RWTH HPC

Train/eval runs happen on the cluster (SLURM + Apptainer); serving/UI stay local.

## 0. Credentials — read first

**Do not put your HPC password in this repo or paste it into chat.** Authenticate with an
**SSH key**. The only HPC info that lives here is non-secret connection metadata in
`cluster.env` (gitignored) — host, username, account, partition.

> Whoever/whatever drives `sbatch` runs the commands *from your machine* over your SSH session.
> An assistant cannot and should not log in with your password; it writes the scripts, you run
> them (or run them in an authenticated terminal).

## 1. One-time SSH setup (on your laptop)

```bash
ssh-keygen -t ed25519 -C "drive-vlm"            # if you don't have a key
ssh-copy-id <user>@login23-1.hpc.itc.rwth-aachen.de
# add an alias in ~/.ssh/config:
#   Host rwth
#       HostName login23-1.hpc.itc.rwth-aachen.de
#       User <user>
ssh rwth                                         # should log in without a password
```

Then copy the connection template:
```bash
cp scripts/hpc/cluster.env.example scripts/hpc/cluster.env   # fill in, stays gitignored
```

## 2. Build the container (once, where you have root/fakeroot — laptop or CI)

```bash
apptainer build drive-vlm.sif scripts/hpc/Apptainer.def
scp drive-vlm.sif rwth:~/Drive-VLM/
```
On the cluster you typically can't build (no root) — you only `apptainer exec` the `.sif`.

## 3. Get code + data onto the cluster

```bash
rsync -av --exclude .venv --exclude data ./ rwth:~/Drive-VLM/   # code
# data: either `dvc pull` on the cluster (once a remote is configured), or rsync data/ over.
```

## 4. Submit the job

```bash
ssh rwth
cd ~/Drive-VLM && mkdir -p logs
sbatch --account=<your_account> scripts/hpc/eval_baseline.slurm
squeue --me            # watch it
tail -f logs/t2c-eval-*.out
```

## RWTH specifics to confirm once logged in

- **Partition/account:** `sinfo` lists partitions (GPU is commonly `c23g` on CLAIX-2023);
  your charge account comes from your project grant. Set both in `cluster.env` / the `--account`
  flag.
- **Apptainer module:** `module spider Apptainer` to find the exact module name.
- Placeholders in `cluster.env.example` (host, partition) are best-guesses — verify against
  your RWTH account page / `sinfo`.
