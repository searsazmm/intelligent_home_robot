#!/usr/bin/env bash
# 端到端联调自检：不开摄像头、不连模块 A，纯离线验证 A 的数据 → B → C 的链路。
#
#     bash tools/e2e_offline_check.sh
#
# 覆盖四件事：
#   1. B 能从 CSV 回放视觉数据并推出 normal/sad/tired/absent
#   2. 主动关怀在 --speed N 下**确实触发**（这是倍速缩放的验收点）
#   3. proactive 报文真的能被 8002 上的 C 端收到
#   4. 普通对话（C→B→C）也正常
#
# 不碰真实声卡、不开麦克风（--no-voice），任何机器上都能跑。
#
# ⚠️ 第 3 步用的是 watch_8002.py（**一句话都不发**的被动监听），
#    不是 mock_c_client.py。因为 mock_c_client 每发一句话都会刷新 B 的
#    user_cooldown（60 秒），主动关怀在此期间**按设计**不会触发 ——
#    用它验 proactive 必然失败，而不是功能坏了。
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE" || exit 1

TMP="${TMPDIR:-/tmp}"
LOG_B="$TMP/e2e_b.log"
LOG_WATCH="$TMP/e2e_watch.log"
LOG_CHAT="$TMP/e2e_chat.log"
HIST="$TMP/e2e_history.csv"

# ---------------------------------------------------------------------------
# 起跑前先确认端口是空的 —— 这一步能省掉一整类"看起来像灵异事件"的排查
# ---------------------------------------------------------------------------
# 踩过的坑：上一次实验留下的 B 没被 kill 掉，还占着 8001/8002。
# 而 Windows 的 SO_REUSEADDR 与 Linux **语义不同** —— 它允许第二个进程
# 绑定同一个 端口（等价于 Linux 的 SO_REUSEPORT），于是两个 B 同时"监听成功"，
# 新来的连接由操作系统分给其中**任意一个**。
#
# 结果是我这边看到：
#   · 新 B 的日志一路写着「对话通道 0 个客户端」
#   · 监听端却收到了状态报文，而且状态是 absent —— 来自那个**没有视觉输入**的旧 B
#   · 主动关怀照常触发，却一条都没送到监听端
# 看起来像广播坏了，实际上只是连到了另一个进程。
#
# Linux 上这种情况会直接抛 "Address already in use"，反而更好查。
for port in 8000 8001 8002; do
    if netstat -ano 2>/dev/null | grep -qE ":$port[[:space:]]+.*LISTENING"; then
        echo "✗ 端口 $port 已被占用 —— 大概上一个 B 还在跑。"
        echo
        echo "  先解决它（Windows + Git Bash）："
        echo "    netstat -ano | grep -E ':800[012].*LISTENING'   # 看 PID"
        echo "    taskkill //F //PID <PID>                        # 按 PID 杀"
        echo
        echo "  注意：模块 B 的 SO_REUSEADDR 在 Windows 上允许**两个进程同时**"
        echo "  绑定同一端口，所以第二个 B 不会报错而是静默抢走一部分连接。"
        exit 1
    fi
done

rm -f "$LOG_B" "$LOG_WATCH" "$LOG_CHAT" "$HIST"

# ⚠️ 这里的 ( ... ) & 是必须的：写成 `cd X && python -c ... &` 的话，
#    & 作用于整个 `cd && python` 列表，cd 只发生在子 shell 里，
#    后面的命令仍在旧目录 —— 表现为 "can't open file .../tools/xxx.py"。
#
# 第 1、2、3 步用**同一次** B 进程：离线回放只播一遍（119.9 数据秒，
# 5 倍速约 24 墙钟秒），重启一次就要从头再等一遍。
( python main.py --offline data/sample_vision.csv --speed 5 --demo \
      --no-voice --history "$HIST" > "$LOG_B" 2>&1 ) &
B_PID=$!

sleep 3
# 被动监听：听够整个回放时长，等 proactive 自己冒出来
( python tools/watch_8002.py --seconds 28 > "$LOG_WATCH" 2>&1 ) &
W_PID=$!

sleep 32
kill "$W_PID" "$B_PID" 2>/dev/null
wait "$W_PID" "$B_PID" 2>/dev/null

echo "===== 1) B 的状态判定 ====="
grep "状态变更" "$LOG_B" | sed 's/^/  /' || echo "  ✗ 一次状态都没变"

echo
echo "===== 2) 主动关怀是否触发（倍速缩放的验收点）====="
if grep -q "主动关怀 \[" "$LOG_B"; then
    grep "主动关怀 \[" "$LOG_B" | sed 's/^/  /'
else
    echo "  ✗ 没有触发！确认带了 --demo（不加的话默认阈值 20 数据秒，"
    echo "    在 5 倍速下只剩 4 墙钟秒，必然不触发）"
fi

echo
echo "===== 3) 8002 上收到了什么（C 端视角）====="
if grep -q "主动开口" "$LOG_WATCH"; then
    grep -A2 "主动开口" "$LOG_WATCH" | head -12 | sed 's/^/  /'
else
    echo "  ✗ C 端没看到 proactive —— 8002 广播这条线没通"
fi
echo "  --- 状态报文条数：$(grep -c '状态 ' "$LOG_WATCH") ---"

echo
echo "===== 4) 普通对话回环（C→B→C）====="
# 这一步要**另起一个 B**：上一步那个已经随回放结束退出了。
# 先起 B 再起 C —— 反过来 C 会先撞上连接失败、要等 1 秒退避重连。
#
# ⚠️ 下面 grep 的是**中文字面量**，所以这四个日志文件必须是 UTF-8。
#    这一点靠各入口的 `sys.stdout.reconfigure(encoding="utf-8")` 保证
#    （main.py / watch_8002.py / mock_c_client.py 都有）。
#    曾经 mock_c_client.py 漏了这行：输出重定向到文件时 Python 用 ANSI 码页（GBK），
#    于是中文写入是 GBK、本脚本是 UTF-8，grep 永远匹配不上 ——
#    脚本报「没收到回复」，而对话回环其实一条都没丢。
#    **别靠放宽断言来"修"它**，那会把真实故障一起放过去。
( python main.py --offline data/sample_vision.csv --speed 5 --no-voice \
      --history "$HIST" > "$TMP/e2e_b2.log" 2>&1 ) &
B2_PID=$!
sleep 3
( python tools/mock_c_client.py --auto --between 0.3 > "$LOG_CHAT" 2>&1 ) &
C_PID=$!
sleep 10
kill "$C_PID" "$B2_PID" 2>/dev/null
wait "$C_PID" "$B2_PID" 2>/dev/null
if grep -q "状态=" "$LOG_CHAT"; then
    echo "  ✓ 收到回复 $(grep -c '状态=' "$LOG_CHAT") 条"
    head -4 "$LOG_CHAT" | sed 's/^/  /'
else
    echo "  ✗ 没收到任何回复"
    echo "    先看 B 那边的日志：$(grep -c . "$TMP/e2e_b2.log") 行"
    tail -5 "$TMP/e2e_b2.log" | sed 's/^/    /'
fi

echo
echo "===== 5) 有没有异常 ====="
if grep -q "Traceback" "$LOG_B" "$LOG_WATCH" "$LOG_CHAT" "$TMP/e2e_b2.log" 2>/dev/null; then
    grep -n "Traceback" "$LOG_B" "$LOG_WATCH" "$LOG_CHAT" "$TMP/e2e_b2.log" | sed 's/^/  /'
else
    echo "  ✓ 无异常"
fi
