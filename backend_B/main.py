# -*- coding: utf-8 -*-
"""后端 B 启动入口。

职责：把各个部件装起来，管好线程和退出。

    ┌──────────────┐  TCP 8000（B 是客户端）  ┌─────────────────────────┐
    │  模块 A 视觉  │ ───────────────────────▶ │  VisionClient           │
    └──────────────┘                          │    ↓                    │
                                              │  VisionStateEvaluator   │  → normal/sad/tired/absent
    ┌──────────────┐  TCP 8001（B 是服务端）  │    ↓                    │
    │  模块 C 前端  │ ◀─────────────────────── │  StatusBroadcaster      │
    │              │  TCP 8002（B 是服务端）  │    ↓                    │
    │              │ ◀───────────────────────▶ │  ChatServer → Dialogue  │
    └──────────────┘                          └─────────────────────────┘
                                                        ↓
                                                  CsvHistoryStore
                                                  data/history.csv

线程模型（全部是 daemon 线程，主线程负责收尾）：
    1. 视觉读取线程   1 个，跑 VisionClient.run() 或 OfflineVisionFeeder.run()
    2. 状态发布线程   1 个，轮询判定结果并按需推送，**并顺带驱动主动关怀**
    3. 状态服务线程   1(accept) + N(每客户端)，对 C 的 8001
    4. 对话服务线程   1(accept) + N(每客户端)，对 C 的 8002
    5. 语音循环线程   1 个，可选，听麦克风 → 识别 → handle_chat → 朗读
    6. 控制台线程     1 个，可选，--stdin 时启动

主动关怀刻意**挂在状态发布线程上**（每 0.2 秒一个节拍），不新开轮询线程 ——
判定需要的信息（当前状态、用户最近说话时间）在那个节拍上已经齐了。

用法：
    python main.py                              # 连模块 A，正常模式
    python main.py --offline data/sample_vision.csv --speed 5
    python main.py --offline data/sample_vision.csv --speed 5 --demo   # 答辩演示
    python main.py --stdin                      # 键盘代替麦克风，机器人照样朗读
    python main.py --list-audio                 # 看设备名，再用 --audio-out 指定
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from typing import Optional

# 允许 `python backend_B/main.py` 与 `cd backend_B && python main.py` 两种方式都能跑：
# 保证本目录在 sys.path 里，这样 `import config` / `from core import ...` 都能工作。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core import proactive as proactive_lib
from core.dialogue import DialogueEngine, DialogueReply, static_replies
from core.history_store import CsvHistoryStore, TurnRecord
from core.llm import build_chatter
from core.ui_channel import ChatServer, StatusBroadcaster
from core.vision_client import OfflineVisionFeeder, VisionClient
from core.vision_state import VisionState, VisionStateEvaluator
from core.voice import audio as audio_lib
from core.voice.loop import LoopConfig, Speaker, VoiceLoop
from core.voice.speech_cache import SpeechCache
from core.voice.stt import build_recognizer
from core.voice.tts import build_synthesizer
from core.voice.vad import VadConfig

# 语音包内所有第三方 import（sounddevice / vosk / edge_tts …）都写在函数体内部，
# 顶层 import 它们不会引入任何第三方依赖 —— 见 core/voice/base.py 的说明，
# 以及 tests/test_no_third_party_imports.py 的守卫。

logger = logging.getLogger("backend_b")

#: 需要联网合成、因而值得预合成缓存的引擎。
#: SAPI 是本地合成（本机实测一句 37ms），缓存它只是白白读写磁盘。
ONLINE_TTS_ENGINES = ("doubao", "volcengine", "edge_tts")


def _tts_engine_key(synthesizer) -> str:
    """缓存键前缀：引擎 + 音色 + 语速。

    这三项任一变了，合成结果就完全不同，必须让旧缓存自动失效 ——
    否则用户改了音色却听见上一个音色的缓存，会以为是配置没生效。
    """
    return "|".join(
        str(getattr(synthesizer, part, "")) for part in ("name", "voice", "speed")
    )


class BackendB:
    """后端 B 的应用装配与生命周期管理。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        # ---- 核心组件 ----
        self.evaluator = VisionStateEvaluator()

        self.history: Optional[CsvHistoryStore] = None
        if not args.no_history:
            self.history = CsvHistoryStore(args.history)
            logger.info("历史记录文件：%s（已有 %d 条）", self.history.path, self.history.count())
            self._preload_history_hint()

        # 大模型对话（core/llm.py）。没配 B_DEEPSEEK_TOKEN 就拿到一个
        # NullChatClient —— 对话完全走规则模板，和接大模型之前一模一样。
        # 构造里可能有一次探针请求（probe），所以放在最前面，好让启动日志
        # 的顺序和用户读日志的顺序一致。
        self.chatter = build_chatter(args.llm, probe=args.llm_probe)
        self.dialogue = DialogueEngine(history=self.history, llm=self.chatter)

        self.status_server = StatusBroadcaster(
            host=args.status_host, port=args.status_port
        )
        self.chat_server = ChatServer(
            # 不走 self.handle_chat：8002 是"不出声"的文本通道，而 handle_chat
            # 默认会垫应声词（要出声）。见 _handle_chat_from_c。
            on_chat=self._handle_chat_from_c,
            # 注入当前状态，让新连上的 C 端**立刻**拿到状态而不是等 15 秒心跳。
            # C 端默认只连 8002，这条就是它开局的唯一状态来源。
            state_provider=self._effective_state,
            host=args.chat_host, port=args.chat_port,
        )

        # 视觉数据来源：实时 socket 或离线 CSV
        if args.offline:
            self.vision_source = OfflineVisionFeeder(
                csv_path=args.offline,
                evaluator=self.evaluator,
                speed=args.speed,
                loop=args.loop,
            )
            logger.info("离线模式：回放 %s（%.1fx 倍速，loop=%s）",
                        args.offline, args.speed, args.loop)
        else:
            self.vision_source = VisionClient(
                evaluator=self.evaluator,
                host=args.vision_host, port=args.vision_port,
            )

        # ---- 状态覆盖 ----
        # 对话得出的融合状态比纯视觉状态更贴近现实（比如视觉没看出来但文字很消极），
        # 但视觉线程下一秒就会推回它自己的判定，两边打架会让前端状态乱跳。
        # 所以对话产生的状态用"限时覆盖"的方式生效，过期后交还给视觉。
        self._state_lock = threading.Lock()
        self._override_state: Optional[str] = None
        self._override_reason: str = ""
        self._override_until: float = 0.0

        self._stop = threading.Event()
        self._threads = []
        self._last_published_state: Optional[str] = None

        # ---- 主动关怀 ----
        # 挂在 _state_publish_loop 的 0.2 秒节拍上，不新增线程（见该方法的注释）。
        self.proactive = proactive_lib.ProactiveScheduler(self._build_proactive_policy())

        # ---- 语音 ----
        self.speaker: Optional[Speaker] = None
        self.voice_loop: Optional[VoiceLoop] = None
        self._build_voice()

    # ==================================================================
    # 装配：主动关怀
    # ==================================================================

    def _scale_for_speed(self, seconds: float) -> float:
        """把「数据时间」阈值换算成「墙钟」阈值。

        离线回放 ``--speed 5`` 时，CSV 里的 1 秒只占用 0.2 个墙钟秒。
        而 config §9 的阈值和视觉判定窗口 (config §3/§4) 一样是按**数据时间**
        定的 —— 阈值不除以倍速，那条 20 秒的 sad 段在 5 倍速下只剩 4 个墙钟秒，
        主动关怀就永远不会触发，答辩时看起来像没实现。

        注意**只缩放叙事性的时长**（持续多久算低落、两次开口间隔多久、
        离开多久算回来）。``startup_grace`` 和 ``user_cooldown`` 描述的是
        操作者的真实墙钟行为（"刚开机别吓人"、"人刚说完话别抢"），
        和数据跑多快无关，**绝不能缩放**。
        """
        if not self.args.offline or self.args.speed <= 0:
            return seconds
        return seconds / max(0.01, self.args.speed)

    def _build_proactive_policy(self) -> proactive_lib.ProactivePolicy:
        """按运行模式挑一组阈值。

        ``--demo`` 用 config §10 那组更宽松的阈值（原因见该节注释），
        并强制关掉静默时段 —— 答辩多半在白天，但万一是晚上调试，
        一个"该触发却不触发"的静默时段会让人以为功能坏了。
        """
        args = self.args
        if args.demo:
            sustain = config.PROACTIVE_DEMO_SUSTAIN
            min_interval = config.PROACTIVE_DEMO_MIN_INTERVAL
            max_per_hour = config.PROACTIVE_DEMO_MAX_PER_HOUR
            greeting_absent = config.PROACTIVE_DEMO_GREETING_ABSENT
            startup_grace = config.PROACTIVE_DEMO_STARTUP_GRACE
            quiet_enabled = False
        else:
            sustain = config.PROACTIVE_SUSTAIN
            min_interval = config.PROACTIVE_MIN_INTERVAL
            max_per_hour = config.PROACTIVE_MAX_PER_HOUR
            greeting_absent = config.PROACTIVE_GREETING_ABSENT
            startup_grace = config.PROACTIVE_STARTUP_GRACE
            quiet_enabled = not args.no_quiet_hours

        return proactive_lib.ProactivePolicy(
            sustain=self._scale_for_speed(sustain),
            min_interval=self._scale_for_speed(min_interval),
            greeting_absent=self._scale_for_speed(greeting_absent),
            max_per_hour=max_per_hour,
            startup_grace=startup_grace,                    # 真实墙钟，不缩放
            user_cooldown=config.PROACTIVE_USER_COOLDOWN,   # 真实墙钟，不缩放
            quiet_enabled=quiet_enabled,
        )

    def _fire_proactive(self, decision: proactive_lib.ProactiveDecision) -> None:
        """真的开口：出文案 → 写历史 → 推 8002 → 朗读。

        **绝不调 handle_chat()**。那条路会更新 ``_last_user_text_at``
        （于是机器人自己的话被当成"用户在场"的证据，推翻视觉的 absent 判定），
        会写一行 ``role=user`` 但内容是机器人的话，还会刷新 30 秒状态覆盖 ——
        主动关怀只要比 30 秒频繁，视觉状态就被永久钉住。
        详见 core/proactive.py 模块开头第 1 条。
        """
        vision = self.evaluator.get_state()
        text = self.dialogue.proactive_reply(decision.kind, vision)
        if not text:
            return

        logger.info("主动关怀 [%s]：%s", decision.kind, text)

        # 只写 role=robot 这一行（append_turn 会额外写一条 role=user 的假记录）
        if self.history is not None:
            try:
                self.history.append_record(TurnRecord(
                    session_id=self.history.session_id,
                    role="robot",
                    text=text,
                    vision_state=vision.state,
                    intent=decision.kind,
                ))
            except Exception:
                logger.exception("写入主动关怀记录失败（不影响本次开口）")

        # 推给前端 C。状态用的是视觉原状态 —— 主动关怀**不改状态**。
        self.chat_server.broadcast_proactive(
            text, state=vision.state, reason=decision.reason, kind=decision.kind)

        # 记账必须在**真的发出**之后，否则一次被门禁挡下的判定也会占用间隔配额
        self.proactive.note_proactive()

        self._say(text)

    # ==================================================================
    # 装配：语音
    # ==================================================================

    def _build_voice(self) -> None:
        """装配语音层。缺库 / 缺设备 / 缺模型一律**安静降级**，绝不阻止 B 启动。

        B 模块的运行期零第三方依赖是刻意设计（见 core/voice/__init__.py），
        所以这里所有失败路径都只是记一行日志，然后照常以文本模式运行。
        """
        args = self.args

        if not args.voice:
            logger.info("语音功能已关闭（--no-voice），仅走键盘 / 8002 文本通道")
            return

        # ---- 说 ----
        synthesizer = build_synthesizer(
            args.tts, speed=args.voice_speed, voice=args.tts_voice)
        player = None if args.no_play else audio_lib.Player(args.audio_out)

        # 只给在线引擎配缓存：它们每句要等一次网络往返（本机实测 ~3.3 秒），
        # SAPI 是 37ms 的本地合成，缓存它只是白写磁盘。
        cache = None
        if getattr(synthesizer, "name", "") in ONLINE_TTS_ENGINES:
            cache = SpeechCache(config.VOICE_CACHE_DIR, _tts_engine_key(synthesizer))

        self.speaker = Speaker(
            synthesizer=synthesizer,
            player=player,
            on_busy_change=self._on_speaking_change,
            half_duplex=args.half_duplex,
            cache=cache,
            # 只缓存会被反复说起的固定句 —— ``static_replies()`` 正是
            # ``SpeechCache.prune()`` 的保留集，所以"存进去的 = 留得下的"。
            # 接上大模型之后每条回复都是模型现编的、必然唯一，缓存它们
            # 只会让缓存目录一直涨而命中率始终是 0。
            cache_allow=static_replies(),
        )

        if cache is not None and config.VOICE_PREWARM:
            self._start_prewarm(cache, synthesizer)

        if args.no_play:
            logger.info("已关闭播放（--no-play）：只合成不发声，WAV 仍在临时目录生成")

        # ---- 听 ----
        if args.stdin:
            # --stdin 与语音输入二选一：input() 阻塞在控制台时，
            # 语音线程的识别日志会把命令行糊掉，而这正好发生在演示的时候。
            # 朗读保留 —— 「打字代替说话、机器人照样念出来」是个好用的自测姿势。
            logger.info("已开启 --stdin，跳过语音输入（机器人仍会朗读回复）")
            return

        try:
            recorder = self._make_recorder()
        except Exception:
            logger.exception("初始化录音设备失败，跳过语音输入")
            return

        self.voice_loop = VoiceLoop(
            on_text=self.handle_chat,
            speaker=self.speaker,
            recognizer=build_recognizer(args.stt),
            recorder=recorder,
            # 采样率必须跟着设备走：VadConfig 按它算每帧字节数，
            # 写死 16000 而设备是 44100 会让端点检测的毫秒数全部算错。
            vad=VadConfig(sample_rate=recorder.sample_rate),
            config=LoopConfig(half_duplex=args.half_duplex),
        )

    def _start_prewarm(self, cache, synthesizer) -> None:
        """后台把固定回复逐条合成进缓存。**绝不阻塞启动。**

        在线引擎每句话都要等一次网络往返，而机器人说的话里很大一部分是
        固定文案（问候、主动关怀、各种兜底句）—— 提前合成好，用户听到的
        第一句问候就不必先沉默三秒。

        线程是 daemon：B 退出时它跟着走，不会把进程吊住。合成失败的条目
        只记 WARNING 跳过，下次启动再试 —— 预热是优化，不该成为故障点。
        """
        texts = static_replies()
        if not texts:
            return

        def worker() -> None:
            scratch = tempfile.mkdtemp(prefix="b_prewarm_")
            wav_path = os.path.join(scratch, "warm.wav")
            reused = done = failed = 0
            try:
                # 先把换了引擎/音色/语速之后再也命不中的旧文件清掉。
                try:
                    cache.prune(texts)
                except Exception:
                    logger.debug("预合成缓存清理失败（不影响预热）", exc_info=True)

                total = len(texts)
                logger.info("开始预合成固定回复（%d 条，引擎 %s）",
                            total, getattr(synthesizer, "name", "?"))

                for index, text in enumerate(texts, 1):
                    if cache.lookup(text) is not None:
                        reused += 1
                        logger.debug("预热[%d/%d] 已有缓存：%s", index, total, text[:20])
                        continue
                    if not synthesizer.synthesize_wav(text, wav_path):
                        failed += 1
                        logger.warning("预热[%d/%d] 合成失败，跳过：%s",
                                       index, total, text[:20])
                        continue
                    # ⚠️ 和 Speaker 里同一条规矩：**兜底产出的不写缓存。**
                    #    预热是"用主引擎把固定文案提前合成好"，这时候网络
                    #    大概率先不好（不然也不会走兜底）。把 SAPI 的机械音
                    #    存进豆包的缓存键下，等网络恢复了这些句子也永远是
                    #    错的音色 —— 而且**永久**，缓存命中不会再走合成。
                    #    失败就跳过，下次启动再试。
                    if getattr(synthesizer, "used_fallback", False):
                        failed += 1
                        logger.debug("预热[%d/%d] 走了本地兜底，不写入缓存：%s",
                                     index, total, text[:20])
                        continue
                    cache.store(text, wav_path)
                    done += 1
                    logger.debug("预热[%d/%d] 新合成：%s", index, total, text[:20])

                logger.info("预合成完成：新合成 %d，复用缓存 %d，失败 %d",
                            done, reused, failed)
            except Exception:
                logger.exception("预合成线程异常退出（不影响正常运行）")
            finally:
                shutil.rmtree(scratch, ignore_errors=True)

        threading.Thread(target=worker, name="tts-prewarm", daemon=True).start()

    def _make_recorder(self) -> "audio_lib.Recorder":
        """构造录音器，**采样率显式取设备原生值**。

        不硬编码 16000：本机 MME 报 44100、WASAPI 报 48000，
        用不匹配的采样率开流会直接抛 ``PortAudioError: Invalid sample rate``。

        也不交给 ``Recorder.start()`` 自己去问设备：VAD 需要**构造时**就知道
        采样率（每帧字节数 = 采样率 × 帧长），所以这里先问一遍再传给两边。
        """
        device = audio_lib.resolve_input_device(self.args.audio_in)
        rate = audio_lib.native_sample_rate(device, 16000)
        return audio_lib.Recorder(
            device_name=self.args.audio_in,
            sample_rate=rate,
            blocksize_ms=config.VOICE_BLOCK_MS,
        )

    def _on_speaking_change(self, speaking: bool) -> None:
        """播报开始/结束的回调。由 Speaker 在**它自己的线程**里调用。

        两件事，都必须线程安全（两个都是 Event 的原子操作）：
          1. 告诉主动关怀「我正在说话」，别在同一时刻抢话头；
          2. 半双工时把麦克风静音，否则机器人会听见自己 → 自激回路。
        """
        self.proactive.set_speaking(speaking)
        if self.voice_loop is not None and self.voice_loop.recorder is not None:
            self.voice_loop.recorder.set_muted(speaking and self.args.half_duplex)

    def _say(self, text: str) -> None:
        """朗读一句话。会**阻塞到说完**（调用方据此才知道何时该恢复采集）。

        没有语音层时静默跳过 —— 文字回复、表情、历史记录都不受影响。
        """
        if self.speaker is not None and text:
            self.speaker.say(text)

    # ==================================================================
    # 启动 / 停止
    # ==================================================================

    def run(self) -> None:
        """启动全部组件并阻塞直到收到退出信号。"""
        try:
            self.status_server.start()
        except OSError as exc:
            logger.error("状态端口 %s:%d 启动失败：%s", self.args.status_host,
                         self.args.status_port, exc)
            logger.error("端口可能已被占用，用 netstat -ano | findstr :%d 查一下",
                         self.args.status_port)
            raise

        try:
            self.chat_server.start()
        except OSError as exc:
            logger.error("对话端口 %s:%d 启动失败：%s", self.args.chat_host,
                         self.args.chat_port, exc)
            self.status_server.stop()
            raise

        self._spawn(self.vision_source.run, "视觉读取")
        self._spawn(self._state_publish_loop, "状态发布")

        if self.args.stdin:
            self._spawn(self._stdin_loop, "控制台输入")

        # 语音循环自己管线程（VoiceLoop.start 内部创建），不走 _spawn：
        # 它有自己的退避重试与降级逻辑，套一层 _guarded 反而会把
        # "连续出错后主动停用"这条路径盖掉。
        if self.voice_loop is not None and not self.voice_loop.start():
            logger.warning("语音输入未启动，用键盘或 8002 文本通道继续")
            self.voice_loop = None

        self._print_banner()
        self._wait_for_exit()
        self.shutdown()

    def shutdown(self) -> None:
        """优雅关闭：先停数据源，再停语音，最后停服务。"""
        if self._stop.is_set() and not self._threads:
            return
        logger.info("正在关闭后端 B ...")
        self._stop.set()

        # ---- 语音必须先停，且必须在 join() 之前 ----
        # 只在循环里检查标志位是不够的：录音线程可能正卡在
        # sounddevice 的队列上，而常驻的 PowerShell 子进程会**继续把
        # 没说完的话说完**（进程都退出了声音还在响，很难看）。
        # VoiceLoop.stop() 会关音频流并 join 自己的线程，Speaker.close()
        # 会 kill 掉那个常驻 PowerShell。
        if self.voice_loop is not None:
            try:
                self.voice_loop.stop()
            except Exception:
                logger.exception("停止语音循环失败")
            self.voice_loop = None
        if self.speaker is not None:
            try:
                self.speaker.close()
            except Exception:
                logger.exception("关闭语音输出失败")
            self.speaker = None

        # 先停视觉源（可能正在 recv 阻塞）
        stop = getattr(self.vision_source, "stop", None)
        if callable(stop):
            stop()

        self.status_server.stop()
        self.chat_server.stop()

        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()

        logger.info("后端 B 已停止")

    def _wait_for_exit(self) -> None:
        """主线程守候，处理 Ctrl+C。"""
        try:
            while not self._stop.is_set():
                time.sleep(0.3)
        except KeyboardInterrupt:
            print()   # 把 ^C 后面的光标换行，避免日志和提示符粘在一起
            logger.info("收到 Ctrl+C，准备退出")

    def _spawn(self, target, name: str) -> threading.Thread:
        thread = threading.Thread(target=self._guarded(target, name), name=name, daemon=True)
        thread.start()
        self._threads.append(thread)
        return thread

    def _guarded(self, target, name: str):
        """包一层异常保护：任何子线程崩了都要留下日志，而不是静默死掉。"""
        def wrapper():
            try:
                target()
            except Exception:
                logger.exception("线程 %s 异常退出", name)
        return wrapper

    # ==================================================================
    # 状态发布
    # ==================================================================

    def _publish_state(self, state: str, reason: str = "") -> None:
        """把状态**同时**推给 8001（纯文本）和 8002（JSON）。

        ⚠️ 这两件事必须成对做，不能只推 8001。
        C 端默认**只连 8002**（除非加 --also-status），所以"只推 8001"
        等于"C 什么都收不到"。

        这个做法是修一个真实的缺口：原先只有 ``_state_publish_loop`` 会顺手
        镜像到 8002，而对话导致的状态覆盖（:meth:`handle_chat` 里那段）
        只调了 ``status_server.publish`` —— 于是在默认的单 C 配置下，
        跟机器人说一句话把状态改成 sad 之后，C 的表情要等到下一次真实变化
        或 15 秒心跳才跟上，中间 C 的界面和 B 的结论是不一致的，
        而日志还显示"推送成功"。

        去重语义保持不变：8001 那边判定"没变化且没到心跳"时不发，
        这里也就不会镜像 —— 免得 C 收到一堆重复状态把表情闪来闪去。
        """
        # 镜像的目标客户端数要在**发送前**取：日志里那个数字写在
        # status_server.publish 内部，晚了就拿不到"这次要发给几个 C"。
        if not self.status_server.publish(
                state, mirror_clients=self.chat_server.client_count()):
            return
        self.chat_server.broadcast_state(state, reason)

    def _state_publish_loop(self) -> None:
        """定期取当前状态并推给 C。变化时推，没变化时靠心跳推。

        主动关怀也挂在这个节拍上 —— 它需要的输入（当前状态、用户最近说话的
        时刻、是否正在播报）在这个节拍上本来就齐了，单开一个轮询线程
        只会多一处要同步的状态，不会更快。
        """
        while not self._stop.is_set():
            try:
                state, reason = self._effective_state()
                self._publish_state(state, reason)
                self._tick_proactive(state)
            except Exception:
                logger.exception("状态发布出错")
            self._stop.wait(0.2)

    def _tick_proactive(self, state: str) -> None:
        """一个主动关怀节拍。判定放在 core/proactive.py，这里只负责执行。

        ``last_user_at`` 取自 ``dialogue.last_user_text_at`` —— 那是**唯一**
        的「用户在场证据」。麦克风、8002、``--stdin`` 三条输入路径都经由
        ``handle_chat`` 汇到它，所以这里读一个字段就够了。
        """
        if not self.args.proactive:
            return

        decision = self.proactive.tick(
            state, last_user_at=self.dialogue.last_user_text_at)
        if decision is not None:
            self._fire_proactive(decision)

    def _effective_state(self):
        """当前对外的状态：优先用未过期的对话覆盖，否则用视觉判定。"""
        now = time.monotonic()
        with self._state_lock:
            if self._override_state and now < self._override_until:
                return self._override_state, self._override_reason

        vision: VisionState = self.evaluator.get_state()
        return vision.state, vision.reason

    def _set_state_override(self, state: str, reason: str, seconds: float = 30.0) -> None:
        """用对话结论临时覆盖视觉状态。"""
        with self._state_lock:
            self._override_state = state
            self._override_reason = reason
            self._override_until = time.monotonic() + seconds

    # ==================================================================
    # 对话处理（ChatServer 的回调）
    # ==================================================================

    def handle_chat(self, text: str, *, allow_ack: bool = True) -> DialogueReply:
        """收到用户一句话：生成回复 → 写历史 → 更新状态覆盖 → 返回。

        ``allow_ack=False`` 关掉**应声词**。8002 那条路要关 —— 它是模块 C
        与测试脚本走的文本通道，按约定**不朗读**，而应声词是要出声的：
        测试脚本跑一轮，机房里就会响起机器人的"嗯，我听着呢"。
        """
        vision = self.evaluator.get_state()
        reply = self.dialogue.respond(
            text, vision,
            on_thinking=self._on_thinking if allow_ack else None,
        )

        logger.info("用户：%s", text)
        logger.info("机器人：%s  [%s]", reply.reply, reply.reason)

        # 落盘历史（任务要求 4 的另一半：写进去，之后才读得出来）
        if self.history is not None:
            try:
                self.history.append_turn(
                    user_text=text,
                    reply_text=reply.reply,
                    vision_state=reply.state,
                    text_emotion=reply.emotion_label,
                    emotion_score=reply.emotion_score,
                    intent=reply.intent,
                )
            except OSError:
                logger.exception("写入历史记录失败（不影响本次回复）")

        # 对话结论比纯视觉更准，限时覆盖一下，避免前端状态和对话内容对不上。
        # 走 _publish_state 而不是直接 publish：C 端要**立刻**看到这次覆盖，
        # 否则它会带着旧表情再挂 15 秒（见 _publish_state 的说明）。
        if reply.state != vision.state:
            reason = f"对话判定：{reply.reason}"
            self._set_state_override(reply.state, reason)
            self._publish_state(reply.state, reason)

        return reply

    def _handle_chat_from_c(self, text: str) -> DialogueReply:
        """8002 通道的对话入口。和语音那条路的唯一区别：**不出声**。

        单独一个方法而不是让调用方传参：``on_chat`` 是构造 ``ChatServer``
        时按引用传进去的，那边只能给一个无参调用形式。
        """
        return self.handle_chat(text, allow_ack=False)

    def _on_thinking(self, ack: str) -> None:
        """应声词的出口，交给 :meth:`Speaker.say_async` 播。

        ⚠️ **必须立刻返回。** 调用它的线程紧接着就要去做大模型那次网络请求，
        在这里阻塞等于把应声词变成对正式回复的额外延迟 —— 正好是它想解决的
        那个问题。

        没有语音层（未装配 / ``--no-play``）时直接跳过：应声词的全部意义
        就是出声，没有声音就没有意义，没必要让对话路径多绕一圈。
        """
        if self.speaker is not None:
            self.speaker.say_async(ack)

    # ==================================================================
    # 控制台输入（开发自测用）
    # ==================================================================

    def _stdin_loop(self) -> None:
        """从控制台读一行当作"用户说的话"，方便不开前端 C 就能测对话。

        回复会**照常朗读**：这样没有麦克风也能验证整条语音输出通路
        （音色、语速、半双工静音都测得到），是答辩前调 SAPI 音色的主要手段。

        8002 那条路则刻意**不朗读** —— 它是模块 C 与测试脚本走的文本通道，
        回复已经作为 JSON 发回对端了，再出声只会干扰自动化测试。
        """
        print("\n[控制台输入已开启] 直接打字回车，就当作是用户说的话；输入 :q 退出。\n")
        while not self._stop.is_set():
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                break
            line = line.strip()
            if not line:
                continue
            if line in (":q", ":quit", "exit"):
                self._stop.set()
                break
            reply = self.handle_chat(line)
            print(f"  → {reply.reply}   [状态={reply.state} 意图={reply.intent}]\n")
            self._say(reply.reply)

    # ==================================================================
    # 杂项
    # ==================================================================

    def _preload_history_hint(self) -> None:
        """启动时读一把历史，验证"预留的 CSV 读取接口"确实能用。"""
        try:
            recent = self.history.load_recent(limit=config.HISTORY_PRELOAD_ROWS)
            if recent:
                logger.info("已预加载最近 %d 条历史记录（供对话上下文使用）", len(recent))
                last = recent[-1]
                logger.info("  最近一条：%s：%s", last.role, last.text[:40])
            else:
                logger.info("历史记录为空，这是第一次运行")
        except Exception:
            logger.exception("预加载历史记录失败（不影响启动）")

    def _print_banner(self) -> None:
        source = f"离线回放 {self.args.offline}" if self.args.offline else \
            f"模块 A {self.args.vision_host}:{self.args.vision_port}"

        # 语音和主动关怀的**真实**状态要放在最显眼处。
        # 出了问题（没麦克风、没识别引擎、被 --no-voice 关掉）时，
        # 这一屏是唯一能一眼看出"为什么机器人不理我"的地方。
        if self.speaker is None:
            voice = "已关闭（--no-voice）"
        elif self.voice_loop is not None:
            voice = (f"听 {getattr(self.voice_loop.recognizer, 'name', '?')} + "
                     f"说 {self.speaker.engine_name}"
                     f"{'' if self.args.half_duplex else '（可插话）'}")
        else:
            voice = f"只说（{self.speaker.engine_name}），无语音输入"

        # 这行**必须带原因**，不能只写"未启用"。用户抱怨过"回答太硬板"，
        # 而"大模型没接上"正是那句话的唯一解释 —— 一眼看不到原因的话，
        # 下一个问题就是"它怎么还是这么硬板"。
        #
        # 原因字符串自带括号（「已按配置停用（--llm none）」），所以这里
        # 不再套一层括号 —— 套了就是 `未启用（已按配置停用（…））`。
        if getattr(self.chatter, "available", False):
            llm_line = f"已启用（{self.chatter.name} / {getattr(self.chatter, 'model', '?')}）"
        else:
            llm_line = "未启用 —— %s" % getattr(self.chatter, "reason", "无可用大模型")

        proactive = "已关闭" if not self.args.proactive else "已开启"
        if self.args.proactive and self.proactive.policy.quiet_enabled:
            start, end = self.proactive.policy.quiet_hours
            proactive += f"（{start}:00–{end}:00 静默，仅静默主动开口）"

        print("=" * 62)
        print("  居家陪伴机器人 · 后端 B")
        print("-" * 62)
        print(f"  视觉来源    {source}")
        print(f"  状态推送    {self.args.status_host}:{self.args.status_port}  (B=服务端, 推 normal/sad/tired/absent)")
        print(f"  对话通道    {self.args.chat_host}:{self.args.chat_port}  (B=服务端, 收发 JSON)")
        print(f"  历史记录    {self.history.path if self.history else '已禁用'}")
        print(f"  语音        {voice}")
        print(f"  大模型对话  {llm_line}")
        print(f"  主动关怀    {proactive}")
        print("-" * 62)
        print("  Ctrl+C 退出")
        print("=" * 62)
        print()


# ======================================================================
# 命令行参数
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backend_B",
        description="居家陪伴机器人 · 后端 B（视觉接收 + 文本情绪 + 对话管理）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python main.py                                      正常模式，连模块 A\n"
            "  python main.py --offline data/sample_vision.csv     离线回放，无需模块 A\n"
            "  python main.py --offline data/sample_vision.csv --speed 5 --demo --stdin\n"
            "                                                      答辩演示（无摄像头）\n"
            "  python main.py --stdin                              控制台打字，机器人朗读回复\n"
            "  python main.py --no-voice                           纯文本模式，排查语音问题时用\n"
            "  python main.py --list-audio                         看音频设备列表\n"
        ),
    )

    parser.add_argument("--offline", metavar="CSV", default=None,
                        help="离线模式：从该 CSV 回放视觉数据，不连模块 A")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="离线回放倍速（默认 1.0）")
    parser.add_argument("--loop", action="store_true",
                        help="离线回放循环播放（默认只播一遍；单帧测试时很有用）")
    parser.add_argument("--stdin", action="store_true",
                        help="开启控制台输入，直接打字测试对话")

    parser.add_argument("--vision-host", default=config.VISION_HOST,
                        help=f"模块 A 地址（默认 {config.VISION_HOST}）")
    parser.add_argument("--vision-port", type=int, default=config.VISION_PORT,
                        help=f"模块 A 端口（默认 {config.VISION_PORT}）")
    parser.add_argument("--status-host", default=config.STATUS_HOST,
                        help=f"状态推送监听地址（默认 {config.STATUS_HOST}）")
    parser.add_argument("--status-port", type=int, default=config.STATUS_PORT,
                        help=f"状态推送监听端口（默认 {config.STATUS_PORT}）")
    parser.add_argument("--chat-host", default=config.CHAT_HOST,
                        help=f"对话通道监听地址（默认 {config.CHAT_HOST}）")
    parser.add_argument("--chat-port", type=int, default=config.CHAT_PORT,
                        help=f"对话通道监听端口（默认 {config.CHAT_PORT}）")

    parser.add_argument("--history", default=config.HISTORY_CSV,
                        help=f"历史记录 CSV 路径（默认 {config.HISTORY_CSV}）")
    parser.add_argument("--no-history", action="store_true",
                        help="不读写历史记录 CSV")
    parser.add_argument("--log-level", default=config.LOG_LEVEL,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help=f"日志级别（默认 {config.LOG_LEVEL}）")

    # ---- 演示 ----
    parser.add_argument("--demo", action="store_true",
                        help="演示预设：主动关怀改用更宽松的阈值并关闭静默时段"
                             "（配合 --offline --speed 使用，见 config §10）")

    # ---- 主动关怀 ----
    parser.add_argument("--proactive", action=argparse.BooleanOptionalAction,
                        default=config.PROACTIVE_ENABLED,
                        help="主动关怀总开关（--no-proactive 关闭）")
    parser.add_argument("--no-quiet-hours", action="store_true",
                        help="关闭 22:00–07:00 的静默时段。"
                             "注意静默时段只静默**主动**开口，应答永远不受影响")

    # ---- 语音 ----
    parser.add_argument("--voice", action=argparse.BooleanOptionalAction,
                        default=config.VOICE_ENABLED,
                        help="语音总开关（--no-voice 退回纯文本/键盘模式）")
    parser.add_argument("--stt", default=config.STT_ENGINE,
                        choices=["auto", "vosk", "speechrecognition", "dashscope", "none"],
                        help=f"语音识别引擎（默认 {config.STT_ENGINE}）。"
                             "auto = 装了哪个用哪个，都没装则降级")
    parser.add_argument("--tts", default=config.TTS_ENGINE,
                        choices=["auto", "doubao", "edge", "sapi", "null"],
                        help=f"语音合成引擎（默认 {config.TTS_ENGINE}）。"
                             "auto = 依次尝试 豆包 → edge-tts → SAPI，"
                             "挑第一个可用的（豆包要 B_DOUBAO_TOKEN + B_DOUBAO_VOICE）")
    parser.add_argument("--tts-voice", default=config.VOICE_TTS_VOICE,
                        help="指定音色名（留空 = SAPI/edge-tts 自动挑中文音色）。"
                             "豆包引擎下这个值就是 speaker，必须与控制台已开通的一致，"
                             "且要和模型版本配套（默认 Vivi 2.0 配 seed-tts-2.0）")
    parser.add_argument("--llm", default=config.LLM_ENGINE,
                        choices=["auto", "deepseek", "none"],
                        help=f"对话大模型（默认 {config.LLM_ENGINE}）。"
                             "auto = 有 B_DEEPSEEK_TOKEN 就用 DeepSeek，"
                             "没有就完全走规则模板。none = 强制只用模板")
    parser.add_argument("--llm-probe", action=argparse.BooleanOptionalAction,
                        default=config.LLM_PROBE,
                        help="启动时真调一次大模型确认凭证有效（默认开）。"
                             "离线开发 / 跑测试时用 --no-llm-probe 跳过这次网络请求")
    parser.add_argument("--voice-speed", type=float, default=config.VOICE_SPEED,
                        help=f"语速倍数（默认 {config.VOICE_SPEED}，0.9 比正常慢 10%%）")
    parser.add_argument("--barge-in", dest="half_duplex", action="store_false",
                        default=config.VOICE_HALF_DUPLEX,
                        help="允许插话：播报期间不静音麦克风。**只在戴耳机时用** —— "
                             "外放会形成机器人听见自己的自激回路")
    parser.add_argument("--no-play", action="store_true",
                        help="只合成不播放（验证 WAV 生成，或服务器上没声卡时）")
    parser.add_argument("--audio-in", default=config.AUDIO_INPUT_NAME, metavar="名字片段",
                        help="输入设备名子串（如 Realtek）。留空 = 系统默认")
    parser.add_argument("--audio-out", default=config.AUDIO_OUTPUT_NAME, metavar="名字片段",
                        help="输出设备名子串。**不要用设备序号**（会随驱动更新漂移）。"
                             "听不见时先查端点音量，再考虑用这个显式指定真音箱")
    parser.add_argument("--list-audio", action="store_true",
                        help="打印音频设备列表后退出（排错第一步）")

    return parser


def setup_logging(level: str) -> None:
    """统一日志格式。中文日志在 Windows 控制台要注意编码。"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def fix_console_encoding() -> None:
    """把 stdout 切到 UTF-8。

    Windows 控制台默认可能是 GBK，中文日志会乱码或直接抛 UnicodeEncodeError。
    **必须在 parse_args 之前调用**：``--help`` 的说明文字也全是中文，
    而 argparse 解析完就立刻打印了 —— 那时再改编码，用户看到的第一屏已经是乱码。
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def main(argv=None) -> int:
    fix_console_encoding()

    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    # --list-audio 是排错的第一步（"为什么没声音/没在听"），
    # 单独放在装配之前，不要求端口可用、也不连模块 A。
    if args.list_audio:
        print("音频设备列表：")
        print(audio_lib.describe_devices())
        print()
        print("用法：拿名字里的**一段**填给 --audio-in / --audio-out，不要用序号。")
        return 0

    app = BackendB(args)
    try:
        app.run()
    except OSError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
