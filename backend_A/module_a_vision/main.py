"""模块 A 入口。

    python -m module_a_vision.main                          # 合成源，正常剧本
    python -m module_a_vision.main --scenario drowsy        # 困倦剧本
    python -m module_a_vision.main --source camera          # 真实摄像头
    python -m module_a_vision.main --source video --video a.mp4
    python -m module_a_vision.main --source csv --csv export.csv
    python -m module_a_vision.main --dry-run                # 不起服务，只打印报文
    python -m module_a_vision.main --emit v2                # 换成原生的 10 秒窗口

本机（无摄像头）的推荐路径是 ``--source synthetic``：它不经过像素、
不加载模型，完全确定性，用来验证 A→B 的链路与 B 的判定。
真实路径（camera / video）需要 ``tools/setup_models.py`` 先下载模型，
且**在本机未经验证**——见 ``face/mediapipe_backend.py`` 的警告。

``--emit`` 选出站格式（默认 ``v1``）：

* ``v1`` —— api_doc §3.2 的逐帧平铺 8 字段，团队仓库里 ``backend_B`` 认的格式。
* ``v2`` —— 原生的 10 秒嵌套 :class:`WindowState`。
"""

from __future__ import annotations

import argparse
import contextlib
import sys

from shared.schema import WindowState

from .capture.base import is_feature_source, is_frame_source
from .server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    EMIT_MODES,
    EMIT_V1,
    VisionServer,
    build_server,
    describe_hints,
)
from .wire import check_contract, describe_typed, describe_v1, to_v1_sample


def _prepare_console() -> None:
    """让控制台能吃中文，并且**别把日志缓冲住**。

    ``line_buffering=True`` 不是可有可无的装饰。stdout 连到终端时 Python
    按行刷新，可一旦重定向到文件（``python main.py > a.log``、后台任务、
    服务化部署都是这样）就变成 8KB 块缓冲：**进程正常运行，日志文件却是空的**，
    要等缓冲区攒满才吐一次。一个"起来了吗"的问题会因此变成一次误判。

    这在开发期就真实发生过：后台起 A、前台看日志，看到的是空文件，
    而 ``netstat`` 显示 8000 明明在监听。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(Exception):
                reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="module_a_vision",
        description="空巢老人居家陪伴系统 · 模块 A（视觉感知）",
    )
    p.add_argument(
        "--source",
        default="synthetic",
        choices=("synthetic", "camera", "video", "csv"),
        help="采集源种类（默认合成源，本机无摄像头时的主路径）",
    )
    p.add_argument("--scenario", default="normal", help="合成剧本名（--source synthetic）")
    p.add_argument("--video", default=None, help="视频文件路径（--source video）")
    p.add_argument("--csv", default=None, help="CSV 文件路径（--source csv）")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--elder-id", default="E1001")
    p.add_argument("--device-id", default="cam-livingroom-01")
    p.add_argument("--fps", type=float, default=10.0, help="目标帧率")
    p.add_argument(
        "--emit",
        default=EMIT_V1,
        choices=EMIT_MODES,
        help=(
            "出站格式。v1=api_doc §3.2 逐帧平铺 8 字段（默认，backend_B 认这个）；"
            "v2=原生 10 秒嵌套窗口"
        ),
    )
    p.add_argument(
        "--every",
        type=int,
        default=10,
        help=(
            "仅 --dry-run：v1 模式下每多少帧打印一行（默认 10，即约 1 行/秒）。"
            "想看每一帧就设 1"
        ),
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="读到多少帧后停止（默认不限；演示时常用 300）",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="回放倍速。调试时可调大，例如 10 表示十倍速跑完整段剧本",
    )
    p.add_argument(
        "--real-time",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "是否按真实节奏产出帧。默认：起服务时开、--dry-run 时关。"
            "关掉的话 180 秒的剧本会在瞬间跑完，B 的冷却与持续性判定会失去意义"
        ),
    )
    p.add_argument(
        "--loop",
        action="store_true",
        help="剧本/视频循环播放，用于长时间演示",
    )
    p.add_argument(
        "--vitals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "是否发 api_doc §3.5 的体征报文（rppg）。只有合成源挂得上。"
            "关掉可以得到一条只含帧报文的干净流，排查协议问题时用"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="不起 TCP 服务，只把窗口打到标准输出（用于验证数据契约）",
    )
    p.add_argument(
        "--list-scenarios",
        action="store_true",
        help="列出可用的合成剧本后退出",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    _prepare_console()
    args = build_parser().parse_args(argv)

    if args.list_scenarios:
        return _list_scenarios()

    try:
        server = build_server(
            source_kind=args.source,
            scenario=args.scenario,
            video=args.video,
            csv=args.csv,
            camera_index=args.camera_index,
            host=args.host,
            port=args.port,
            elder_id=args.elder_id,
            device_id=args.device_id,
            fps=args.fps,
            max_frames=args.max_frames,
            loop=args.loop,
            speed=args.speed,
            real_time=(
                args.real_time if args.real_time is not None else not args.dry_run
            ),
            emit_mode=args.emit,
            vitals=args.vitals,
        )
    except (ValueError, RuntimeError, ImportError) as exc:
        print(f"[A] 启动失败：{exc}", file=sys.stderr)
        return 2

    try:
        if args.dry_run:
            return _dry_run(server, every=max(1, args.every))

        server.serve()
        server.run()
    except KeyboardInterrupt:
        print("\n[A] 收到中断信号，正在收摊……")
    except (ValueError, RuntimeError, ImportError) as exc:
        # **这一段不是多余的。** 采集源与人脸后端都是**惰性打开**的：
        # 装配时不碰硬件、不 import mediapipe，第一次读才碰。所以
        # "没装 mediapipe"、"摄像头被占用"这类错误只在**开始读之后**才抛，
        # 而且本来就带着写好的中文提示（见 face/base.py 的 FaceBackendError）。
        #
        # 不接住它们，那句"请安装：pip install mediapipe"就会被埋进
        # 十几行栈回溯里，看起来像一个崩溃而不是一条安装说明。
        print(f"\n[A] ✗ 运行中断：{exc}", file=sys.stderr)
        return 2
    finally:
        server.stop()

    print(f"[A] 已退出（{_summarize_sent(server)}）")
    return 0


def _summarize_sent(server: VisionServer) -> str:
    """收摊时那行摘要。两种模式发的报文种类不同，数字的含义也不同。"""
    if server.emit_mode == EMIT_V1:
        return (
            f"共推送 {server.stats['v1_frames']} 帧 §3.2 报文"
            f" + {server.stats['rppg_packets']} 条 §3.5 体征报文"
            f"（读取 {server.stats['frames']} 帧）"
        )
    return f"共推送 {server.stats['windows']} 个窗口"


# ================================================================ 子模式

def _dry_run(server: VisionServer, every: int = 10) -> int:
    """不起服务，只把出站报文打出来。

    这是排查"到底是 A 算错了还是 B 判错了"最快的手段：先看 A 吐的报文
    本身对不对，再去看 B。

    ``v1`` 模式下顺带**逐帧跑契约闸**（:func:`assert_v1_contract`），
    所以 ``--dry-run`` 同时是一次数据契约检查，不只是打印。
    """
    if server.emit_mode == EMIT_V1:
        return _dry_run_v1(server, every=every)
    return _dry_run_v2(server)


def _dry_run_v1(server: VisionServer, every: int = 10) -> int:
    """逐帧投影、打印、过契约闸，并**照发** §3.5 的体征报文。

    ``every`` 是打印间隔而不是采样间隔 —— 计数器与契约检查**每一帧都跑**，
    只是不每帧都打印。若拿它当采样阈值，契约检查就会漏掉它跳过的那几帧。

    ⚠️ 这里的契约校验必须走 :func:`~module_a_vision.wire.check_contract`
    （与 :meth:`VisionServer.broadcast` 同一个函数），**不能**再直接调
    ``assert_v1_contract``。分叉的后果很具体：类型化报文会被判"多余字段"，
    于是 ``--dry-run`` —— 排查"到底是 A 算错了还是 B 判错了"最快的那件
    工具 —— 会持续输出与实跑不符的结论。
    """
    source = server.source
    aggregator = server.aggregator
    vitals = server.vitals

    source.open()
    frames = 0
    violations = 0
    vitals_sent = 0
    last: dict | None = None
    try:
        while True:
            frame = _read(source, server.backend)
            if frame is None:
                break
            aggregator.add_frame(frame)
            frames += 1

            payload = to_v1_sample(
                frame,
                blink_total=aggregator.blink_total,
                emotion=aggregator.classify_frame(frame),
            )
            last = payload

            violations += _check(payload, frames)

            if frames % every == 0:
                print(describe_v1(payload))

            # 体征通道与实跑走**同一条**路径（server._emit_side_channels），
            # 时间戳同样取 frame.ts。
            if vitals is not None:
                typed = vitals.feed(frame.ts)
                if typed is not None:
                    problems = _check(typed, frames)
                    violations += problems
                    vitals_sent += 1
                    # 体征报文 1Hz，而打印间隔是 every 帧 —— 不单独判一次
                    # 就会漏掉绝大多数体征行（每 10 帧才印一帧的话，
                    # 印到的多半是帧报文）。
                    if problems or vitals_sent % 5 == 1:
                        print(describe_typed(typed))

            # v1 不消费窗口，但必须排空缓冲（详见 server._emit_v1）。
            if aggregator.should_close(frame.ts):
                aggregator.close_window(frame.ts)
    except KeyboardInterrupt:
        print("\n[A] 已中断")
    finally:
        source.close()

    # 最后一帧总是打出来：它带着最终的 blink_cnt，而且上面的取模
    # 会正好跳过收尾那几帧——恰恰是最值得看的一段。
    if last is not None and frames % every:
        print(describe_v1(last))

    print(
        f"\n[A] 共投影 {frames} 帧 §3.2 报文 + {vitals_sent} 条 §3.5 体征报文"
        f"（未起服务）  契约违规 {violations} 条"
    )
    if violations:
        print("[A] ⚠ 有报文不符合 api_doc，接上 backend_B 会静默失效")
    elif vitals is not None and vitals_sent == 0:
        # 单独点出来：体征一条都没发，是这条通道整个没接上，
        # 而不是"数值不好看"。
        print("[A] ⚠ 体征报文一条都没发出 —— 体征通道是死的")
    return 0


def _check(payload: dict, frame_index: int) -> int:
    """跑契约闸并打印违规项。返回违规条数。"""
    kind, problems = check_contract(payload)
    if not problems:
        return 0
    print(f"  ✗ 第 {frame_index} 帧的 {kind} 报文契约违规：{'；'.join(problems)}")
    return len(problems)


def _dry_run_v2(server: VisionServer) -> int:
    """原生模式：打印每个 10 秒窗口。"""
    source = server.source
    aggregator = server.aggregator

    source.open()
    windows = 0
    try:
        while True:
            frame = _read(source, server.backend)
            if frame is None:
                break
            aggregator.add_frame(frame)
            if aggregator.should_close(frame.ts):
                window = aggregator.close_window(frame.ts)
                windows += 1
                _print_window(window, windows)
    except KeyboardInterrupt:
        print("\n[A] 已中断")
    finally:
        source.close()

    print(f"[A] 共产出 {windows} 个窗口（未起服务）")
    return 0


def _read(source, backend):
    if is_feature_source(source):
        return source.read_features()
    if is_frame_source(source):
        pixels = source.read()
        if pixels is None or backend is None:
            return None
        return backend.process(pixels)
    raise RuntimeError("采集源既不是帧源也不是特征源")


def _print_window(window: WindowState, index: int) -> None:
    obs = window.observations
    print(f"\n── 窗口 #{index}  {window.timestamp:.1f}s ──────────────")
    print(f"  质量      : {'可用' if window.quality.usable else '不可用'}"
          f"（人脸 {window.quality.face_found_ratio:.0%}）")
    print(f"  帧        : {window.window.frames_valid}/{window.window.frames_total}"
          f"  {window.window.duration_sec:.1f}s")
    print(f"  表情      : {obs.emotion.label}  conf={obs.emotion.confidence:.2f}"
          f"  stable={obs.emotion.stable_sec:.0f}s")
    print(f"  头姿      : {obs.head_pose.label}  "
          f"roll={obs.head_pose.roll_deg:.1f} pitch={obs.head_pose.pitch_deg:.1f} "
          f"yaw={obs.head_pose.yaw_deg:.1f}")
    print(f"  注意力    : {obs.attention.label}  conf={obs.attention.confidence:.2f}")
    print(f"  疲劳      : {obs.fatigue.level}  score={obs.fatigue.score:.2f}")
    print(f"  眼部      : {obs.eye_state.label}  "
          f"closure={obs.eye_state.closure_ratio:.2f}  "
          f"blink={obs.eye_state.blink_rate_per_min:.0f}/min  "
          f"long={obs.eye_state.long_closure_sec:.0f}s  "
          f"stable={obs.eye_state.stable_sec:.0f}s")
    print(f"  提示      : {describe_hints(window)}")
    print(f"  隐私自证  : 原图上传={window.privacy.raw_frame_uploaded}  "
          f"人脸图上传={window.privacy.face_image_uploaded}  "
          f"本地处理={window.privacy.on_device_processing}")


def _list_scenarios() -> int:
    from .capture.synthetic import BUILTIN_SCENARIOS

    print("[A] 可用的合成剧本：")
    for name, timeline in BUILTIN_SCENARIOS.items():
        total = sum(dur for _, dur in timeline)
        plan = " → ".join(f"{state}({dur:.0f}s)" for state, dur in timeline)
        print(f"  {name:<10} 共 {total:>5.0f}s  {plan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
