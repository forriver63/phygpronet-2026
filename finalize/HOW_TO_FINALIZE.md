# 如何在没有 Claude 协助的情况下整理仓库并上传

## 一键命令
```bash
bash /Users/max/dev/icml/finalize_repo.sh
```

## 这个脚本会做什么
1. 从 3 张卡（autodl-b / autodl-c / autodl-d）拉所有 `RESULTS.md` + `results.json`
2. 计算 mean ± std，生成 `phygpronet-2026/results/ablations/SUMMARY.md`
3. `git commit + push` 到 https://github.com/forriver63/phygpronet-2026

## 前置条件（已设置好，无需操作）
- `~/.ssh/config` 已配置 autodl-b/c/d 主机别名（私钥认证）
- `gh` CLI 已登录 forriver63 账号
- 本地 python3 可用

## 如果只想本地汇总，不上传 GitHub
跑前 3 步即可（直接 Ctrl-C 跳过 Step 4）。或单独跑：
```bash
python3 /tmp/aggregate_results.py
```

## 每张卡的关键文件路径（远程 SSH 后查看）
- 实时状态：`cat /root/autodl-tmp/runs/overnight/STATUS.txt`
- 已完成结果：`cat /root/autodl-tmp/runs/overnight/RESULTS.md`
- 详细日志：`tail /root/autodl-tmp/runs/overnight/queue_runner.log`

## 如果训练中断（机器关机/断电）
runner 是幂等的，重启训练只需：
```bash
ssh autodl-X "nohup /root/autodl-tmp/queue_runner.sh /root/autodl-tmp/queue_X.txt > /tmp/q.out 2>&1 & disown"
```
已完成的 config 会跳过（按 `results.json` 是否存在判断）。

## 仓库结构
```
phygpronet-2026/
├── docs/                              # docx 报告
├── results/
│   ├── final_results.json             # 汇总（baselines + winner + multi）
│   ├── baselines/<model>/seed*.json   # 6 baselines × 6 seeds
│   ├── phyg_winner/                   # winner exp3_hc20 × 5-6 seeds
│   ├── multi/                         # Multi 重训 × 3 seeds
│   ├── fewshot/{shhs,mass,isruc}/     # 27 runs each
│   └── ablations/
│       ├── per_card/                  # 每张卡的 RESULTS.md 副本
│       └── SUMMARY.md                 # mean ± std 聚合表
├── scripts/                           # 所有训练 + eval 脚本
├── figs/                              # 6 张论文 figure
└── README.md
```
