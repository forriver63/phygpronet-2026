# 新电脑同步指南（旧 Mac 用不了时）

> 适用场景：旧 Mac 不可用 / Claude 不可用，但 AutoDL 卡仍在跑

## ⭐ 最简单方式：**直接打开 GitHub 网页**

什么都不用装，浏览器直接看：

| 页面 | URL |
|---|---|
| **总览 + 关键数字** | https://github.com/forriver63/phygpronet-2026 |
| **mean ± std 大表** | https://github.com/forriver63/phygpronet-2026/blob/main/results/ablations/SUMMARY.md |
| **每卡实时状态** | https://github.com/forriver63/phygpronet-2026/tree/main/results/ablations/per_card |
| **PhyG winner 各 seed** | https://github.com/forriver63/phygpronet-2026/tree/main/results/phyg_winner |
| **fewshot 结果** | https://github.com/forriver63/phygpronet-2026/tree/main/results/fewshot |
| **可视化 figures** | https://github.com/forriver63/phygpronet-2026/tree/main/figs |

**数据来源 & 更新机制：**
- `autodl-b` 卡上有个 `auto_push_loop.sh`（已在跑）每 30 分钟自动 pull C/D 数据 → 算 mean±std → push GitHub
- 所以浏览器看到的最大延迟 ≤ 30 分钟
- **零操作 ✓**

---

## 备用方式：手动触发同步（如果 auto_push 挂了）

在任何 Linux/Mac/WSL：

```bash
git clone https://github.com/forriver63/phygpronet-2026.git
cd phygpronet-2026
bash finalize/run_anywhere.sh
```

需要：
- `python3 + git + ssh`
- SSH 公钥已加到 3 张卡的 `~/.ssh/authorized_keys` 或用环境变量：
  ```bash
  export AUTODL_B="root@<ip> -p <port>"
  export AUTODL_C="root@<ip> -p <port>"
  export AUTODL_D="root@<ip> -p <port>"
  ```
- (可选) `gh CLI` 已用 forriver63 登录（用于 push）

---

## 如果 auto_push 停了，怎么重启？

`auto_push_loop.sh` 跑在 `autodl-b`。如果 B 卡重启或进程被杀：

```bash
ssh autodl-b "nohup /root/autodl-tmp/auto_push_loop.sh > /tmp/auto_push.out 2>&1 & disown"
```

如果 B 卡上文件丢了，从 GitHub 重新部署：

```bash
ssh autodl-b "cd /root/autodl-tmp/repo && git pull && \
  cp finalize/auto_push.sh /root/autodl-tmp/ && \
  chmod +x /root/autodl-tmp/auto_push.sh && \
  echo '#!/bin/bash
while :; do /root/autodl-tmp/auto_push.sh; sleep 1800; done' > /root/autodl-tmp/auto_push_loop.sh && \
  chmod +x /root/autodl-tmp/auto_push_loop.sh && \
  nohup /root/autodl-tmp/auto_push_loop.sh > /tmp/auto_push.out 2>&1 & disown"
```

---

## 如果 ablation 训练崩了，怎么重启？

每张卡的 queue_runner.sh 是幂等的（按 `results.json` 是否存在跳过），重启只需 SSH 进去：

```bash
# B 卡
ssh autodl-b "nohup /root/autodl-tmp/queue_runner.sh /root/autodl-tmp/queue_B.txt > /tmp/q.out 2>&1 & disown"

# C 卡  
ssh autodl-c "nohup /root/autodl-tmp/queue_runner.sh /root/autodl-tmp/queue_C.txt > /tmp/q.out 2>&1 & disown"

# D 卡
ssh autodl-d "nohup /root/autodl-tmp/queue_runner.sh /root/autodl-tmp/queue_D_s47.txt > /tmp/q.out 2>&1 & disown"
```

如果远端的 `queue_runner.sh` 或队列文件丢了，从仓库复制：
```bash
git clone https://github.com/forriver63/phygpronet-2026.git
scp phygpronet-2026/finalize/queue_runner.sh autodl-b:/root/autodl-tmp/
scp phygpronet-2026/finalize/queues/queue_B.txt autodl-b:/root/autodl-tmp/queue_B.txt
# (C, D 同理)
```

---

## fewshot 链式触发（已配置，无需操作）

每张卡上 `chain_fewshot.sh` 已在后台等待，**ablation 一结束就自动启动 fewshot**：

| 卡 | Ablation 完成后自动跑 | Runs | 预计耗时 |
|---|---|---|---|
| B | MASS s42/45/47 + SHHS s42 | 36 | ~3h |
| C | SHHS s45 | 9 | ~45min |
| D | SHHS s47 | 9 | ~45min |

监控 fewshot 进度：
```bash
ssh autodl-b "tail /root/autodl-tmp/runs/overnight/chain_fewshot.log"
ssh autodl-b "ls /root/autodl-tmp/runs/overnight/{mass_fs,shhs_fs2}/*.json | wc -l"
```

如果 chain_fewshot 挂了，重启：
```bash
ssh autodl-X "nohup /root/autodl-tmp/chain_fewshot.sh /root/autodl-tmp/queue_fewshot.txt > /tmp/chain.out 2>&1 & disown"
```

---

## 关键 SSH 设置（新电脑上首次配置）

### 1. SSH 别名（推荐）
新电脑 `~/.ssh/config` 加：
```
Host autodl-b
    HostName connect.westb.seetacloud.com
    Port <B 的端口>
    User root
    IdentityFile ~/.ssh/id_ed25519

Host autodl-c
    HostName connect.westc.seetacloud.com
    Port <C 的端口>
    User root
    IdentityFile ~/.ssh/id_ed25519

Host autodl-d
    HostName connect.westb.seetacloud.com
    Port <D 的端口>
    User root
    IdentityFile ~/.ssh/id_ed25519
```

把你的公钥 (`~/.ssh/id_ed25519.pub`) 加到每张卡的 `~/.ssh/authorized_keys`。

### 2. GitHub
```bash
gh auth login  # 用 forriver63 账号
```

---

## 一句话总结

**正常情况：打开 https://github.com/forriver63/phygpronet-2026/blob/main/results/ablations/SUMMARY.md 即可，浏览器就行。**

每 30 分钟由 autodl-b 自动同步。SUMMARY.md 永远是 ≤30min 内最新数据。
