# 新电脑同步实验数据 — 一页备忘

> 适用：旧 Mac 不可用 / 没有 Claude 协助 / 想在新电脑看实验结果

---

## ⭐ 最简单（零安装）：浏览器直接看

| 想看的 | URL |
|---|---|
| **总览 + 关键数字** | https://github.com/forriver63/phygpronet-2026 |
| **mean ± std 大表** | https://github.com/forriver63/phygpronet-2026/blob/main/results/ablations/SUMMARY.md |
| **每张卡实时状态** | https://github.com/forriver63/phygpronet-2026/tree/main/results/ablations/per_card |
| **PhyG winner 各 seed** | https://github.com/forriver63/phygpronet-2026/tree/main/results/phyg_winner |
| **few-shot 结果** | https://github.com/forriver63/phygpronet-2026/tree/main/results/fewshot |
| **figures** | https://github.com/forriver63/phygpronet-2026/tree/main/figs |

**`autodl-b` 卡上每 30 分钟自动 push 一次，所以 GitHub 数据 ≤ 30min 内最新。零操作。**

---

## 如果想立刻看最新（手动触发）

新电脑装好 `git + python3 + ssh + gh CLI`：

```bash
# 1. clone repo
git clone https://github.com/forriver63/phygpronet-2026.git
cd phygpronet-2026

# 2. 配置 SSH 别名（一次性，编辑 ~/.ssh/config）
cat >> ~/.ssh/config <<'EOF'
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

Host autodl-e
    HostName connect.westc.seetacloud.com
    Port <E 的端口>
    User root
    IdentityFile ~/.ssh/id_ed25519
EOF

# 3. 把新电脑公钥加到 4 张卡的 ~/.ssh/authorized_keys（在 AutoDL 控制台或 ssh-copy-id）
ssh-copy-id autodl-b
ssh-copy-id autodl-c
ssh-copy-id autodl-d
ssh-copy-id autodl-e

# 4. GitHub 登录
gh auth login   # 用 forriver63 账号

# 5. 一键同步
bash finalize/run_anywhere.sh
```

`run_anywhere.sh` 会：
- 从 4 张卡拉 `RESULTS.md` + per-config `results.json` + fewshot 结果
- 算 mean ± std → `results/ablations/SUMMARY.md`
- `git push` 到 GitHub

---

## 训练 / 自动 push 万一挂了，怎么重启

### 1. Ablation 训练（4 张卡之一关机后）

每张卡的 `queue_runner.sh` 幂等（已完成 config 自动跳过）。重启：

```bash
ssh autodl-b "nohup /root/autodl-tmp/queue_runner.sh /root/autodl-tmp/queue_B.txt > /tmp/q.out 2>&1 & disown"
ssh autodl-c "nohup /root/autodl-tmp/queue_runner.sh /root/autodl-tmp/queue_C.txt > /tmp/q.out 2>&1 & disown"
ssh autodl-d "nohup /root/autodl-tmp/queue_runner.sh /root/autodl-tmp/queue_D_s47.txt > /tmp/q.out 2>&1 & disown"
```

### 2. Fewshot（ablation 跑完会自动启动；如果 chain 挂了）

```bash
ssh autodl-b "nohup /root/autodl-tmp/fewshot_runner.sh /root/autodl-tmp/queue_fewshot.txt > /tmp/fs.out 2>&1 & disown"
ssh autodl-c "nohup /root/autodl-tmp/fewshot_runner.sh /root/autodl-tmp/queue_fewshot.txt > /tmp/fs.out 2>&1 & disown"
ssh autodl-d "nohup /root/autodl-tmp/fewshot_runner.sh /root/autodl-tmp/queue_fewshot.txt > /tmp/fs.out 2>&1 & disown"
ssh autodl-e "nohup /root/autodl-tmp/fewshot_runner.sh /root/autodl-tmp/queue_fewshot.txt > /tmp/fs.out 2>&1 & disown"
```

### 3. Auto-push 循环（B 卡上）

```bash
ssh autodl-b "nohup /root/autodl-tmp/auto_push_loop.sh > /tmp/auto_push.out 2>&1 & disown"
```

### 4. 远端脚本丢失（卡被重置）

从 GitHub 直接同步回去：

```bash
git clone https://github.com/forriver63/phygpronet-2026.git
scp phygpronet-2026/finalize/queue_runner.sh   autodl-X:/root/autodl-tmp/
scp phygpronet-2026/finalize/fewshot_runner.sh autodl-X:/root/autodl-tmp/
scp phygpronet-2026/finalize/queues/queue_*    autodl-X:/root/autodl-tmp/
```

---

## 4 张卡分工（参考）

| 卡 | GPU | 主任务 |
|---|---|---|
| **B** | 4090 | s45 ablation → MASS s42/45/47 fewshot + SHHS s42 + auto-push master |
| **C** | 4090D | s46 ablation → SHHS s45 fewshot |
| **D** | 4090D | s47 ablation → SHHS s47 fewshot |
| **E** | 5090 | ISRUC s42/45/47 fewshot（27 runs） |

---

## 验证一切正常

```bash
# 4 张卡是否都在跑？
for h in autodl-b autodl-c autodl-d autodl-e; do
  echo "=$h="
  ssh $h "ps -ef | grep -E 'queue_runner|fewshot_runner|chain_fewshot' | grep -v grep | wc -l"
done
# 应每张返回 ≥ 1

# auto-push 是否在跑？
ssh autodl-b "ps -ef | grep auto_push_loop | grep -v grep | wc -l"
# 应返回 1

# 最新 GitHub commit 多久前？
gh api repos/forriver63/phygpronet-2026/commits/main --jq '.commit.committer.date'
```

---

## 一句话

**正常情况：浏览器打开 https://github.com/forriver63/phygpronet-2026/blob/main/results/ablations/SUMMARY.md 即可。**

如果手动需求紧急：`git clone … && bash finalize/run_anywhere.sh`（需要 SSH 到卡）。
