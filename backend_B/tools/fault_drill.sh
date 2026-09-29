#!/usr/bin/env bash
# 容错演练：故意把 A / B 弄掉，检查另外两个模块**不崩、能自己恢复**。
#
#     bash tools/fault_drill.sh
#
# 答辩时可以现场跑这个当"健壮性演示" —— 三个模块分别崩一次，链路自己长回来。
#
# 覆盖四件事：
#   1. 杀掉 A          → B 不崩，状态降级为 absent（不假装还看得见人）
#   2. A 回来          → B 自动重连，状态恢复
#   3. 杀掉 B 再重启   → C 不崩，自动重连
#   4. 重连之后        → C **立刻**拿到当前状态，而不是干等 15 秒心跳
#
# 第 4 条是本脚本存在的主要理由：那是 8002 上一个真实存在过的缺口
# （只连 8002 的前端开局有 15 秒空白），光看代码看不出来，必须掐表跑。
#
# C 用 offscreen 跑，不需要真实显示器/桌面 —— 服务器上也能验。
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE" || exit 1

TMP="${TMPDIR:-/tmp}"
LOG_A="$TMP/drill_a.log"
LOG_B="$TMP/drill_b.log"
LOG_B2="$TMP/drill_b2.log"
LOG_C="$TMP/drill_c.log"
HIST="$TMP/drill_history.csv"

FAILED=0
PIDS=()

cleanup() {
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null
    done
    sleep 0.5
    for pid in "${PIDS[@]:-}"; do
        kill -9 "$pid" 2>/dev/null
    done
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 端口预检：上次实验残留的进程占着 8000/8001/8002 会让结果完全没法看
# ---------------------------------------------------------------------------
# 特别提醒 Windows 的坑：SO_REUSEADDR 语义与 Linux 不同，**两个进程可以同时
# 绑定同一端口**（等价于 Linux 的 SO_REUSEPORT）。于是第二个 B 不会报错，
# 而是静默分走一部分连接 —— 表现为"日志正常但界面上啥也没有"。
for port in 8000 8001 8002; do
    if netstat -ano 2>/dev/null | grep -qE ":$port[[:space:]]+.*LISTENING"; then
        echo "✗ 端口 $port 已被占用。先清掉残留进程："
        echo "    netstat -ano | grep -E ':800[012].*LISTENING'"
        echo "    taskkill //F //PID <PID>"
        exit 1
    fi
done

rm -f "$LOG_A" "$LOG_B" "$LOG_B2" "$LOG_C" "$HIST"

# ⚠️ 注意这里**没有**用 ( ... ) & 包起来。
#    脚本开头已经 cd 到 backend_B，不需要子 shell 换目录；
#    不用子 shell 的话 $! 就是 python 自己的 PID，kill 一定打得中。
#    套了子 shell 时 $! 是那个 bash 的 PID，杀它未必能带走 python ——
#    表现是"明明 kill 了，端口还占着"。

echo "===== 0) 起 A（模拟模块 A，循环回放）====="
python tools/mock_a_server.py --loop > "$LOG_A" 2>&1 &
PIDS+=($!)
sleep 2

echo "===== 起 B ====="
python main.py --no-voice --history "$HIST" > "$LOG_B" 2>&1 &
B_PID=$!
PIDS+=($B_PID)
sleep 4

if grep -q "已连接模块 A" "$LOG_B"; then
    echo "  ✓ B 已连上 A"
else
    echo "  ✗ B 没连上 A —— 后面的演练都不成立，先看 $LOG_B"
    tail -10 "$LOG_B" | sed 's/^/    /'
    exit 1
fi

echo "===== 起 C（offscreen，不需要显示器）====="
# ⚠️ C 是**另一个模块**，必须到它自己的目录里跑 ——
#    本脚本开头 cd 到了 backend_B，直接写 `python main.py` 起的是**模块 B**，
#    而 B 不认 --log-file，会打一行 usage 然后退出（这个错误我犯过一次）。
#    `( cd X && exec python ... )`：exec 让子 shell 被 python **替换**掉，
#    于是 $! 仍然是 python 的 PID，kill 打得中（没有 exec 就只是子 shell 的 PID）。
( cd "$HERE/../frontend_C" && QT_QPA_PLATFORM=offscreen exec python main.py \
      --log-file "$LOG_C" > "$TMP/drill_c_console.log" 2>&1 ) &
C_PID=$!
PIDS+=($C_PID)
sleep 5

if grep -q "已连接模块 B 的对话通道" "$LOG_C"; then
    echo "  ✓ C 已连上 B 的 8002"
else
    echo "  ✗ C 没连上 B 的 8002"
    tail -10 "$LOG_C" 2>/dev/null | sed 's/^/    /'
    FAILED=$((FAILED + 1))
fi

# ---------------------------------------------------------------------------
# 演练 1：杀掉 B，再重启 —— C 应自动重连，并在重连后立刻拿到状态
# ---------------------------------------------------------------------------
echo
echo "===== 1) 杀掉 B → C 应自动重连 ====="
kill "$B_PID" 2>/dev/null
sleep 4

if grep -q "连接模块 B 的对话通道失败\|连接模块 B 的.*失败" "$LOG_C"; then
    echo "  ✓ C 记录了重连失败并继续退避重试（没崩）"
else
    echo "  ⚠ C 的日志里没看到重连失败记录（退避可能刚好没撞上，不一定是故障）"
fi

echo "  重启 B ……"
python main.py --no-voice --history "$HIST" > "$LOG_B2" 2>&1 &
B2_PID=$!
PIDS+=($B2_PID)
sleep 6

echo
echo "===== 2) 重连之后多久拿到状态（本脚本的主要理由）====="
# 期望：C 的日志里「已连接」之后**紧接着**就是一条「状态 →」。
# 中间隔 15 秒的话，说明 B 只在心跳时推状态 —— 那就是那个缺口又回来了。
if grep -q "已连接模块 B 的对话通道" "$LOG_C"; then
    echo "  ✓ C 已重新连上 B"
    echo "  --- C 日志里的连接与状态（按时间顺序）---"
    grep -nE "已连接模块 B|状态 →|表情切换" "$LOG_C" | tail -8 | sed 's/^/    /'

    # 掐表：连接那一行和它之后第一条状态行的秒数差
    python - "$LOG_C" <<'PY'
import re
import sys

# 输出重定向到管道时 Python 用 ANSI 码页（本机 GBK），
# 中文结论会变成乱码 —— 和 tools/ 里那几个入口是同一个坑。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 日志格式：  12:34:56 INFO    frontend_C | 已连接模块 B 的对话通道 ...
STAMP = re.compile(r"^(\d{2}):(\d{2}):(\d{2})")

def secs(line):
    m = STAMP.match(line)
    if not m:
        return None
    h, mnt, s = (int(g) for g in m.groups())
    return h * 3600 + mnt * 60 + s

lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
connect_at = None
for line in lines:
    t = secs(line)
    if t is None:
        continue
    if "已连接模块 B" in line:
        connect_at = t          # 记**最后一次**连接
        continue
    if connect_at is not None and "状态 →" in line:
        delta = (t - connect_at) % 86400
        verdict = "OK" if delta <= 2 else "TOO LATE"
        print(f"    >>> 重连后 {delta} 秒拿到状态  [{verdict}]")
        if delta > 2:
            print("    >>> 超过 2 秒说明连上时没有补发，只能等 15 秒心跳")
            sys.exit(1)
        sys.exit(0)
print("    >>> 重连后没看到状态行 —— 补发逻辑可能没生效")
sys.exit(1)
PY
    [ $? -ne 0 ] && FAILED=$((FAILED + 1))
else
    echo "  ✗ C 没能重新连上 B"
    FAILED=$((FAILED + 1))
fi

# ---------------------------------------------------------------------------
# 演练 3：杀掉 A —— B 不能崩，也不能假装还看得见人
# ---------------------------------------------------------------------------
echo
echo "===== 3) 杀掉 A → B 应降级为 absent 而不是硬撑 ====="
A_PID="${PIDS[0]}"
kill "$A_PID" 2>/dev/null
# VISION_STALE_SECONDS 默认 5 秒，多留一点余量
sleep 9

if grep -q "状态变更 → absent" "$LOG_B" || grep -q "状态变更 → absent" "$LOG_B2"; then
    echo "  ✓ B 把状态降级为 absent（没人在的时候不装有人）"
else
    echo "  ✗ B 没有降级为 absent —— 视觉数据过期没被处理"
    grep "状态变更" "$LOG_B" "$LOG_B2" | tail -4 | sed 's/^/    /'
    FAILED=$((FAILED + 1))
fi

if grep -q "准备重连\|次重连" "$LOG_B" "$LOG_B2"; then
    echo "  ✓ B 进入了退避重连"
else
    echo "  ✗ B 没有尝试重连"
    FAILED=$((FAILED + 1))
fi

echo
echo "===== 4) A 回来 → B 应自动恢复 ====="
python tools/mock_a_server.py --loop > "$TMP/drill_a2.log" 2>&1 &
PIDS+=($!)
sleep 8

if grep -q "已连接模块 A" "$LOG_B2"; then
    echo "  ✓ B 重新连上了 A"
else
    echo "✗ B 没恢复（退避上限 10 秒，8 秒可能还差点，可重跑一次确认）"
    tail -5 "$LOG_B2" | sed 's/^/    /'
fi

echo
echo "===== 5) 有没有异常 ====="
if grep -q "Traceback" "$LOG_A" "$LOG_B" "$LOG_B2" "$LOG_C" "$TMP/drill_c_console.log" 2>/dev/null; then
    grep -n "Traceback" "$LOG_A" "$LOG_B" "$LOG_B2" "$LOG_C" "$TMP/drill_c_console.log" | sed 's/^/  /'
    FAILED=$((FAILED + 1))
else
    echo "  ✓ 全程无 Traceback（A/B 被杀、C 重连、A 回来都没有崩）"
fi

echo
if [ "$FAILED" -eq 0 ]; then
    echo "===== 容错演练全部通过 ====="
else
    echo "===== 有 $FAILED 项没通过，见上面 ✗ ====="
fi
exit "$FAILED"
