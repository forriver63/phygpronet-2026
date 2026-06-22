# 在任何机器上整理实验结果并上传 GitHub

> 适用场景：原 Mac 不可用，没有 Claude 协助，但你能 SSH 到 AutoDL 卡。

## 一键命令

```bash
# 在任何 Linux/Mac/WSL 上：
git clone https://github.com/forriver63/phygpronet-2026.git
cd phygpronet-2026
bash finalize/run_anywhere.sh
```

## 前置条件（一次性）

1. **SSH 能连到 3 张 AutoDL 卡**
   - 把 SSH 公钥添加到每张卡的 `~/.ssh/authorized_keys`
   - 或设置 `~/.ssh/config`，hostname 别名 `autodl-b`, `autodl-c`, `autodl-d`

   如果你的 ssh config 没有别名，直接用环境变量：
   ```bash
   export AUTODL_B="root@<ip> -p <port>"
   export AUTODL_C="root@<ip> -p <port>"
   export AUTODL_D="root@<ip> -p <port>"
   bash finalize/run_anywhere.sh
   ```

2. **python3 + git + ssh + tar** （Linux/Mac 默认都有）

3. **(可选) gh CLI**（用于 push）
   - 没装也行，脚本会跳过 push，你手动 `git push`

## 这个脚本会做什么

1. 从 3 张卡拉 `RESULTS.md`（每张卡训练完一个 config 自动追加一行）
2. 拉每个 config 的完整 `results.json`（per-class F1 等）
3. 跑 `aggregate_local.py`，按 config 分组算 mean ± std
4. 输出到 `results/ablations/SUMMARY.md`
5. `git commit + push` (可选)

## 输出文件位置

- **本地汇总**：`results/ablations/SUMMARY.md`（mean ± std 大表 + per-seed 明细）
- **每卡原始**：`results/ablations/per_card/<B|C|D>_RESULTS.md`
- **GitHub**：https://github.com/forriver63/phygpronet-2026

## 远端训练操作（在每张卡上 SSH 后）

| 操作 | 命令 |
|---|---|
| 实时进度 | `cat /root/autodl-tmp/runs/overnight/STATUS.txt` |
| 已完成结果 | `cat /root/autodl-tmp/runs/overnight/RESULTS.md` |
| 详细日志 | `tail -100 /root/autodl-tmp/runs/overnight/queue_runner.log` |
| 看跑了多少 | `ls /root/autodl-tmp/runs/overnight/exp*/results.json \| wc -l` |
| 看是否还在跑 | `ps -ef \| grep train_phyg \| grep -v grep` |

## 训练机器关机后如何恢复

Runner 是**幂等的**——已完成的 config 自动跳过。重启训练只需 SSH 进去执行：

```bash
# B 卡
nohup /root/autodl-tmp/queue_runner.sh /root/autodl-tmp/queue_B.txt > /tmp/q.out 2>&1 & disown

# C 卡
nohup /root/autodl-tmp/queue_runner.sh /root/autodl-tmp/queue_C.txt > /tmp/q.out 2>&1 & disown

# D 卡
nohup /root/autodl-tmp/queue_runner.sh /root/autodl-tmp/queue_D_s47.txt > /tmp/q.out 2>&1 & disown
```

如果远端的 `queue_runner.sh` 或队列文件丢失，本仓库已备份在 `finalize/queue_runner.sh` 和 `finalize/queues/`，scp 上去即可：

```bash
scp finalize/queue_runner.sh autodl-b:/root/autodl-tmp/
scp finalize/queues/queue_B.txt autodl-b:/root/autodl-tmp/queue_B.txt
# (autodl-c / autodl-d 同理)
```

## 故障排查

| 症状 | 解决 |
|---|---|
| `ssh: Connection refused` | AutoDL 卡可能关机；登录 https://www.autodl.com 启动 |
| `RESULTS.md` 为空 | 远端 runner 还没完成第一个 config，等 ~30 min |
| `git push` 失败 | `gh auth login`（用 forriver63 账号），或 `git remote set-url origin git@github.com:forriver63/phygpronet-2026.git` 用 SSH key |
| `python3 not found` | 装 Python 3.8+ |
| 训练报 numpy/torch 错误 | 在卡上：`pip install 'numpy>=2.0,<2.3'`（适配 torch 2.8 + scipy 1.17） |
