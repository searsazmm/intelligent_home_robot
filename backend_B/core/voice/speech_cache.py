# -*- coding: utf-8 -*-
"""预合成语音缓存：把固定回复提前合成好，消掉开口前的等待。

--------------------------------------------------------------------------
为什么需要它
--------------------------------------------------------------------------
实测（本机，同一句话「您好呀。看您好像有点乏，先坐下歇一会吧。」）：

    SAPI      合成   37ms   —— 离线，快得没感觉
    edge-tts  合成 3282ms   —— 联网，一次往返
    豆包       合成  同上量级（都是 HTTP 往返）

3 秒的沉默在聊天里是很致命的：用户说完一句，机器人愣三秒才开口，
体感像卡住了而不是在思考。

而机器人要说的话里，**很大一部分是固定的**（问候、主动关怀、各种兜底句），
启动时就知道全文。那就没必要每次都等网络 —— 提前合成成 WAV 存下来，
播放时直接读文件，延迟归零。

带 ``{topic}`` 的模板不缓存：它要拼进用户刚说的话，无法预测。
（哪一句算固定，由 ``core.dialogue.static_replies()`` 决定，这里不判断。）

--------------------------------------------------------------------------
两个必须做对的地方
--------------------------------------------------------------------------
1. **键里要带上引擎、音色、语速。** 换了音色却不失效的话，用户会听到
   上一个音色的缓存，然后怀疑「我明明改了配置怎么没变」。
   把这三个拼进哈希，换配置即自动失效，不需要手动清目录。

2. **写入必须是原子的。** 预热是后台线程，而播报是另一个线程 ——
   如果先创建目标文件再往里写，播放线程可能正好打开一个只写了一半的 WAV，
   于是听见到一半就断。所以先写临时文件，再 ``os.replace`` 换过去
   （同目录内的 rename 在 Windows 上也是原子的）。
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import tempfile
import threading
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

#: 缓存文件后缀。
SUFFIX = ".wav"

#: 自己生成的文件名长什么样：32 位小写十六进制（sha256 截断）+ 后缀。
#: ``prune`` 只认这个名字 —— 见 :meth:`SpeechCache.prune`。
_NAME_PATTERN = re.compile(r"^[0-9a-f]{32}\.wav$")


class SpeechCache:
    """固定文本 → 预合成 WAV 的本地缓存。

    ``engine_key`` 应当是「引擎|音色|语速」这样的串；任何一项变了，
    整批缓存自动作废（旧的由 :meth:`prune` 顺手清掉）。
    """

    def __init__(self, directory: str, engine_key: str) -> None:
        self.directory = directory
        self.engine_key = engine_key
        self._lock = threading.Lock()
        #: 目录建不出来（权限、磁盘满）时整个缓存降级为「永远不命中」，
        #: 而不是让播报失败 —— 缓存是优化，不该成为故障点。
        self._enabled = self._ensure_directory()
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------

    def _ensure_directory(self) -> bool:
        try:
            os.makedirs(self.directory, exist_ok=True)
            return True
        except OSError:
            logger.warning("预合成缓存目录不可用，本次运行不做缓存：%s",
                           self.directory)
            return False

    def key_for(self, text: str) -> str:
        """文本 → 稳定哈希。带引擎/音色/语速，换配置即失效。"""
        raw = "%s\x00%s" % (self.engine_key, text)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def path_for(self, text: str) -> str:
        return os.path.join(self.directory, self.key_for(text) + SUFFIX)

    # ------------------------------------------------------------------

    def lookup(self, text: str) -> Optional[str]:
        """命中返回可用 WAV 的路径，否则 None。

        ``text`` 为空也返回 None —— 空串会被哈希成一个合法文件名，
        那样每次空播报都会命中同一个文件，是个安静的错。
        """
        if not self._enabled or not text.strip():
            return None
        path = self.path_for(text)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            self.hits += 1
            return path
        self.misses += 1
        return None

    def store(self, text: str, wav_path: str) -> Optional[str]:
        """把一份已合成好的 WAV 收进缓存。失败只记日志，不抛。"""
        if not self._enabled or not text.strip():
            return None
        if not os.path.isfile(wav_path):
            return None

        target = self.path_for(text)
        try:
            with self._lock:
                # 先写临时文件再 rename：见模块开头第 2 条。
                handle, temp_path = tempfile.mkstemp(
                    dir=self.directory, suffix=".tmp")
                os.close(handle)
                try:
                    shutil.copyfile(wav_path, temp_path)
                    os.replace(temp_path, target)
                except OSError:
                    # 临时文件没换成目标就把垃圾清掉，别越积越多。
                    if os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except OSError:
                            pass
                    raise
            return target
        except OSError:
            logger.debug("写入预合成缓存失败：%s", text[:20])
            return None

    def prune(self, keep_texts: Iterable[str]) -> int:
        """删掉不属于当前键集的缓存文件，返回删除数量。

        换引擎/音色/语速之后旧文件永远不会再被命中，留着只是占地方。

        **只删认得出是自己生成的文件**（见 :data:`_NAME_PATTERN`），
        不递归、不限后缀、绝不碰名字对不上的东西。之所以要卡这么死：
        这个函数的动作是"删除"，而目录名来自配置。万一 ``VOICE_CACHE_DIR``
        被指到一个有别的用途的目录 —— 比如用户图省事填了 ``data/`` ——
        宽松的匹配就会把人家的 WAV 一起清掉，且不可恢复。
        """
        if not self._enabled:
            return 0

        keep = {self.key_for(text) for text in keep_texts}
        removed = 0
        try:
            names = os.listdir(self.directory)
        except OSError:
            return 0

        with self._lock:
            for name in names:
                if not _NAME_PATTERN.match(name):
                    continue
                if name[: -len(SUFFIX)] in keep:
                    continue
                try:
                    os.unlink(os.path.join(self.directory, name))
                    removed += 1
                except OSError:
                    pass
        if removed:
            logger.info("预合成缓存清理了 %d 个过期文件", removed)
        return removed
