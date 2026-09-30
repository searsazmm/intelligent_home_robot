# -*- coding: utf-8 -*-
"""语音引擎的测试。

一律注入假引擎，**不碰真实设备、不联网、不启动 PowerShell**。
真机冒烟测试另见文件末尾（默认不跑）。
"""

import base64
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
import types
import unittest
import wave
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dialogue import static_replies                                 # noqa: E402
from core.voice import speech_cache, stt, tts                            # noqa: E402
from core.voice.base import (                                            # noqa: E402
    NullRecognizer,
    NullSynthesizer,
    describe,
    resolve_optional,
)
from core.voice.loop import (                                            # noqa: E402
    ECHO_SIMILARITY,
    LoopConfig,
    Speaker,
    VoiceLoop,
    _brief,
    _wav_seconds,
    is_meaningful,
    looks_like_echo,
)


def executable_lines(script: str) -> str:
    """剥掉 PowerShell 注释，只留会真正执行的代码。

    **负向断言必须针对这个而不是全文。** 脚本里的注释正好写着
    「不要硬编码 Huihui」「不要用 SpeakAsync」这类说明，
    直接 assertNotIn 全文的话，注释本身会把测试弄红 ——
    而那是文档，不是行为。
    """
    return "\n".join(
        line for line in script.splitlines()
        if not line.lstrip().startswith("#")
    )


def write_wav(path: str, frames: int = 1600, rate: int = 16000) -> None:
    """写一个合法的 16bit 单声道 WAV，供播放/合成测试用。"""
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<h", 1000) * frames)


# --------------------------------------------------------------------------
# 假引擎
# --------------------------------------------------------------------------

class FakeSynthesizer:
    """假合成器：写出一个真的 WAV，好让下游（播放、文件检查）能正常走。"""

    name = "fake"

    def __init__(self, succeed: bool = True, write_file: bool = True) -> None:
        self.succeed = succeed
        self.write_file = write_file
        self.calls = []
        self.closed = 0

    def synthesize_wav(self, text: str, path: str) -> bool:
        self.calls.append((text, path))
        if not self.succeed:
            return False
        if self.write_file:
            write_wav(path)
        return True

    def close(self) -> None:
        self.closed += 1


class FakePlayer:
    """假播放器：记录调用顺序，并能检测是否真的串行。"""

    def __init__(self) -> None:
        self.played = []
        self.stopped = 0
        self._in_play = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()
        self.delay = 0.0

    def play_wav(self, path: str) -> bool:
        with self._lock:
            self._in_play += 1
            self.max_concurrent = max(self.max_concurrent, self._in_play)
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            self._in_play -= 1
        self.played.append(path)
        return True

    def stop(self) -> None:
        self.stopped += 1


class FakeRecognizer:
    name = "fake"

    def __init__(self, text: str = "你好") -> None:
        self.text = text
        self.seen = []

    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        self.seen.append(len(pcm))
        return self.text

    def close(self) -> None:
        pass


class FakeRecorder:
    """假录音设备：按预设脚本吐音频块。"""

    def __init__(self, blocks=None) -> None:
        self.blocks = list(blocks or [])
        self.sample_rate = 16000
        self.muted = False
        self.started = 0
        self.stopped = 0

    def start(self) -> bool:
        self.started += 1
        return True

    def read(self, timeout: float = 0.5):
        if self.blocks:
            return self.blocks.pop(0)
        time.sleep(0.01)
        return None

    def set_muted(self, muted: bool) -> None:
        self.muted = muted

    def stop(self) -> None:
        self.stopped += 1


# --------------------------------------------------------------------------
# STT 注册表
# --------------------------------------------------------------------------

class TestRecognizerRegistry(unittest.TestCase):

    def test_unknown_name_returns_null_not_raise(self):
        engine = stt.build_recognizer("这个引擎不存在")
        self.assertIsInstance(engine, NullRecognizer)

    def test_explicit_null_returns_null(self):
        for name in ("null", "none", "NULL", " None "):
            with self.subTest(name=name):
                self.assertIsInstance(stt.build_recognizer(name), NullRecognizer)

    def test_auto_with_nothing_installed_returns_null(self):
        """**最关键的一条**：一个引擎都没装时不能抛异常。

        B 模块必须能在没有语音依赖的机器上启动 —— 这是可演示性的一部分。
        用 mock 把可用列表清空来模拟"什么都没装"的机器。
        """
        with mock.patch.object(stt, "available_engines", return_value=[]):
            engine = stt.build_recognizer("auto")
        self.assertIsInstance(engine, NullRecognizer)

    def test_auto_picks_first_available(self):
        with mock.patch.object(stt, "available_engines", return_value=["dashscope"]):
            engine = stt.build_recognizer("auto")
        self.assertEqual(engine.name, "dashscope")

    def test_engine_constructor_failure_falls_back_to_null(self):
        """库装了但初始化失败（缺模型、缺密钥）不能炸掉启动。"""

        def boom(**kwargs):
            raise RuntimeError("模拟：模型文件不存在")

        with mock.patch.dict(stt.ENGINES, {"boom": boom}):
            engine = stt.build_recognizer("boom")
        self.assertIsInstance(engine, NullRecognizer)

    def test_available_engines_does_not_import(self):
        """探测可用性不能用 import —— 见 base.resolve_optional 的说明。"""
        result = stt.available_engines()
        self.assertIsInstance(result, list)
        for name in result:
            self.assertIn(name, stt.ENGINE_REQUIREMENTS)

    def test_null_recognizer_returns_empty_string(self):
        engine = NullRecognizer()
        self.assertEqual(engine.transcribe(b"\x00\x00" * 100, 16000), "")
        engine.close()
        engine.close()                       # 幂等


class TestEngineReadiness(unittest.TestCase):
    """``auto`` 只能挑**装了且配好了**的引擎，不能只看包装没装。

    踩过的坑：本机 dashscope 装了、``DASHSCOPE_API_KEY`` 却没设。只看
    ``find_spec`` 的话它算"可用"，于是 ``auto`` 选中它；而 dashscope 的
    ``transcribe`` 把异常吞掉返回空串 —— 现场表现是"对着麦克风说话，
    机器人毫无反应"，日志里只有一行识别失败，完全不指向真正的原因。

    换句话说：**宁可明确地"没有语音输入"，也不要一个每次静默失败的识别器。**
    前者用户看得出是没启用，后者只会让人以为麦克风坏了。
    """

    #: 两个环境变量都显式置空，避免受开发机实际配置影响
    NO_CONFIG = {"DASHSCOPE_API_KEY": "", "B_STT_MODEL": ""}

    def setUp(self):
        """把模型搜索目录也隔离掉 —— 环境变量置空**还不够**。

        ``B_STT_MODEL`` 为空时 ``find_model_dir()`` 会去翻
        ``backend_B/models/`` 和 ``%LOCALAPPDATA%\\vosk``。开发机上只要装了模型，
        「没有模型所以不可用」这一类断言就会假性失败 —— 而且失败与否取决于
        那台机器装没装过模型，是最难复现的那种红。
        """
        self._dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._dir, ignore_errors=True)
        for name, target in (("DEFAULT_MODELS_DIR", os.path.join(self._dir, "空1")),
                             ("LOCAL_MODELS_DIR", os.path.join(self._dir, "空2"))):
            patch = mock.patch.object(stt, name, target)
            patch.start()
            self.addCleanup(patch.stop)

    def fake_installed(self, *packages):
        """假装只有列出的包装了（不真的 import —— 探测本来就不该 import）。"""
        wanted = set(packages)
        return mock.patch.object(
            stt, "resolve_optional",
            side_effect=lambda package: package if package in wanted else None,
        )

    def test_dashscope_without_key_is_installed_but_not_available(self):
        with self.fake_installed("dashscope"), \
                mock.patch.dict(os.environ, self.NO_CONFIG):
            self.assertIn("dashscope", stt.installed_engines())
            self.assertNotIn("dashscope", stt.available_engines())
            self.assertTrue(
                any("DASHSCOPE_API_KEY" in r for r in stt.unavailable_reasons()),
                "提示里要点名缺的是 API key，只说'没有可用引擎'没法照做",
            )

    def test_dashscope_with_key_is_available(self):
        with self.fake_installed("dashscope"), \
                mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            self.assertIn("dashscope", stt.available_engines())
            self.assertEqual(stt.unavailable_reasons(), [])

    def test_dashscope_key_of_only_spaces_is_not_enough(self):
        with self.fake_installed("dashscope"), \
                mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "   "}):
            self.assertNotIn("dashscope", stt.available_engines())

    def test_vosk_without_model_is_not_available(self):
        """vosk 光装库没用，还要有中文模型目录。"""
        with self.fake_installed("vosk"), \
                mock.patch.dict(os.environ, self.NO_CONFIG):
            self.assertIn("vosk", stt.installed_engines())
            self.assertNotIn("vosk", stt.available_engines())
            self.assertTrue(any("B_STT_MODEL" in r for r in stt.unavailable_reasons()))

    def test_vosk_with_model_dir_is_available(self):
        import tempfile

        with tempfile.TemporaryDirectory() as model_dir:
            with self.fake_installed("vosk"), \
                    mock.patch.dict(os.environ, {"B_STT_MODEL": model_dir}):
                self.assertIn("vosk", stt.available_engines())

    def test_vosk_model_path_must_actually_exist(self):
        """填了环境变量但路径不存在 —— 和不填一样不可用。"""
        with self.fake_installed("vosk"), \
                mock.patch.dict(os.environ,
                                {"B_STT_MODEL": os.path.join("并不存在", "模型目录")}):
            self.assertNotIn("vosk", stt.available_engines())

    def test_speechrecognition_needs_no_extra_config(self):
        """没有额外前提的引擎，装了就算可用（READINESS 为 None）。"""
        with self.fake_installed("speech_recognition"), \
                mock.patch.dict(os.environ, self.NO_CONFIG):
            self.assertIn("speechrecognition", stt.available_engines())

    def test_auto_skips_unconfigured_engine_and_yields_null(self):
        """本机那种情况：只有 dashscope 装了但没 key → 必须是空识别器。"""
        with self.fake_installed("dashscope"), \
                mock.patch.dict(os.environ, self.NO_CONFIG):
            self.assertIsInstance(stt.build_recognizer("auto"), NullRecognizer)

    def test_auto_prefers_ready_engine_over_installed_one(self):
        """vosk 配好了、dashscope 没配 → 选 vosk（优先级顺序里它也靠前）。"""
        import tempfile

        with tempfile.TemporaryDirectory() as model_dir:
            with self.fake_installed("vosk", "dashscope"), \
                    mock.patch.dict(os.environ,
                                    {"DASHSCOPE_API_KEY": "", "B_STT_MODEL": model_dir}):
                self.assertEqual(stt.available_engines(), ["vosk"])

    def test_explicit_engine_name_bypasses_readiness(self):
        """显式指定时不看 readiness —— 用户明确要它，就给他试的机会。

        这是刻意的：``--stt dashscope`` 说明用户知道自己在配什么，
        在启动阶段就替他回退反而更难查（少掉了"构造失败"那条日志）。
        """
        built = []

        def factory(**kwargs):
            built.append(kwargs)
            return NullRecognizer()

        with mock.patch.dict(stt.ENGINES, {"dashscope": factory}), \
                self.fake_installed("dashscope"), \
                mock.patch.dict(os.environ, self.NO_CONFIG):
            stt.build_recognizer("dashscope")

        self.assertEqual(len(built), 1, "显式指定时不该被 readiness 挡掉")


class TestModelAutoDiscovery(unittest.TestCase):
    """``backend_B/models/`` 下的模型要能被自动找到。

    这是为了让「装了 vosk 但机器人还是听不见」不必再经过"设环境变量"这一步：
    把解压出来的模型目录丢进 models/ 就应该能用。

    ⚠️ 同时要守住反面：**不能把随便一个目录当成模型**。
    没解压的 zip、误建的空目录如果被当成模型，vosk 会在加载时抛一个
    难懂的异常 —— 比"找不到模型"难查得多。
    """

    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._dir, ignore_errors=True)

        # ⚠️ **两个**模型目录都要 patch 到临时区。
        #    只 patch DEFAULT_MODELS_DIR 的话，LOCAL_MODELS_DIR 会指向真实的
        #    %LOCALAPPDATA%\vosk —— 那台机器上一旦装了模型，
        #    find_model_dir() 就会找到真的模型，于是"应该返回空"的用例全红，
        #    而失败与否取决于开发机装没装模型。这种测试比没有更糟。
        self._patches = [
            mock.patch.object(stt, "DEFAULT_MODELS_DIR", self._dir),
            mock.patch.object(stt, "LOCAL_MODELS_DIR",
                              os.path.join(self._dir, "不存在的备用目录")),
        ]
        for patch in self._patches:
            patch.start()
            self.addCleanup(patch.stop)

    def make_model(self, name: str) -> str:
        """造一个"像模型"的目录（vosk 模型必备 am/ 与 conf/）。"""
        path = os.path.join(self._dir, name)
        os.makedirs(os.path.join(path, "am"))
        os.makedirs(os.path.join(path, "conf"))
        return path

    # ---- 能找到 ---------------------------------------------------------

    def test_finds_a_model_in_the_default_dir(self):
        made = self.make_model("vosk-model-small-cn-0.22")
        with mock.patch.dict(os.environ, {"B_STT_MODEL": ""}):
            self.assertEqual(stt.find_model_dir(), made)

    def test_this_makes_vosk_available(self):
        """自动发现要真的让引擎变为可用，而不只是 find_model_dir 返回值好看。"""
        self.make_model("vosk-model-small-cn-0.22")
        with mock.patch.dict(os.environ, {"B_STT_MODEL": ""}):
            self.assertTrue(stt._vosk_ready())

    def test_picks_a_model_even_alongside_other_files(self):
        """models/ 里同时有 zip 和 README 时，不能影响挑选。"""
        with open(os.path.join(self._dir, "cn.zip"), "wb") as handle:
            handle.write(b"PK\x03\x04 fake zip")
        with open(os.path.join(self._dir, "说明.txt"), "w", encoding="utf-8") as handle:
            handle.write("下载来的模型解压到这里")
        made = self.make_model("vosk-model-small-cn-0.22")
        with mock.patch.dict(os.environ, {"B_STT_MODEL": ""}):
            self.assertEqual(stt.find_model_dir(), made)

    # ---- 不能误认 -------------------------------------------------------

    def test_unpacked_zip_is_not_a_model(self):
        """只有 zip、没解压 —— 必须返回空，而不是把 zip 当模型。"""
        with open(os.path.join(self._dir, "vosk-model-small-cn-0.22.zip"), "wb") as h:
            h.write(b"PK\x03\x04")
        with mock.patch.dict(os.environ, {"B_STT_MODEL": ""}):
            self.assertEqual(stt.find_model_dir(), "")

    def test_empty_directory_is_not_a_model(self):
        os.makedirs(os.path.join(self._dir, "刚建的空目录"))
        with mock.patch.dict(os.environ, {"B_STT_MODEL": ""}):
            self.assertEqual(stt.find_model_dir(), "")

    def test_half_extracted_model_is_not_a_model(self):
        """解压到一半（只有 am/ 没有 conf/）也不认 —— 加载必然失败。"""
        os.makedirs(os.path.join(self._dir, "解压到一半", "am"))
        with mock.patch.dict(os.environ, {"B_STT_MODEL": ""}):
            self.assertEqual(stt.find_model_dir(), "")

    def test_missing_default_dir_is_not_an_error(self):
        """models/ 不存在是常态（用户没下模型），不能抛。"""
        shutil.rmtree(self._dir)
        with mock.patch.dict(os.environ, {"B_STT_MODEL": ""}):
            self.assertEqual(stt.find_model_dir(), "")

    # ---- 环境变量优先级 -------------------------------------------------

    def test_env_var_wins_over_the_default_dir(self):
        self.make_model("自动发现的那个")
        elsewhere = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, elsewhere, ignore_errors=True)
        with mock.patch.dict(os.environ, {"B_STT_MODEL": elsewhere}):
            self.assertEqual(stt.find_model_dir(), elsewhere)

    def test_broken_env_var_does_not_fall_back(self):
        """环境变量指错了就报错，**不要**偷偷回退到 models/。

        显式配置填错了却被静默忽略，最坏的结果是"我明明配了 A，
        它却用了 B"，两种模型的行为差异极难归因。
        """
        self.make_model("models 里的好模型")
        with mock.patch.dict(os.environ, {"B_STT_MODEL": os.path.join("不存在", "x")}):
            self.assertEqual(stt.find_model_dir(), "")

class TestNonAsciiModelPath(unittest.TestCase):
    """模型路径含中文时必须**跳过并解释**，而不是丢给 vosk 报一个误导性错误。

    这组测试来自一次实测：本仓库的路径是
    ``D:\\Virtually C\\成都东软学院下期\\...``，把模型放进
    ``backend_B/models/`` 后 ``vosk.Model()`` 必定失败，报的是
    ``Folder '...' does not contain model files`` —— 而文件明明都在。
    那个报错会把人引向"是不是没解压全"，而真因是路径编码。

    所以这一组的价值不在于"代码能跑"，而在于**不让这个坑重新长回来**：
    如果哪天有人觉得 ASCII 检查多余把它删了，这里会红。
    """

    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._dir, ignore_errors=True)

        # 两个搜索基目录：一个含中文（模拟本仓库的真实路径），一个纯英文。
        # ⚠️ 模型必须建在这两个基目录**里面** —— 建在它们的兄弟位置
        #    find_model_dir() 是看不到的（它只列 base 下的子目录）。
        self._cn_base = os.path.join(self._dir, "成都东软学院")
        self._ascii_base = os.path.join(self._dir, "ascii_models")
        os.makedirs(self._cn_base)
        os.makedirs(self._ascii_base)

        self._patches = [
            mock.patch.object(stt, "DEFAULT_MODELS_DIR", self._cn_base),
            mock.patch.object(stt, "LOCAL_MODELS_DIR", self._ascii_base),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def _make_model(self, base: str, name: str = "vosk-model-small-cn-0.22") -> str:
        path = os.path.join(base, name)
        os.makedirs(os.path.join(path, "am"))
        os.makedirs(os.path.join(path, "conf"))
        return path

    def test_is_ascii_path_detects_chinese(self):
        self.assertTrue(stt.is_ascii_path(r"C:\models\vosk-model-small-cn-0.22"))
        self.assertFalse(stt.is_ascii_path(r"D:\成都东软学院\models\m"))

    def test_chinese_path_model_is_skipped_with_a_warning(self):
        """含中文的候选要被跳过，并说清原因。"""
        self._make_model(self._cn_base)

        with mock.patch.dict(os.environ, {"B_STT_MODEL": ""}):
            with self.assertLogs("core.voice.stt", level="WARNING") as caught:
                found = stt.find_model_dir()

        self.assertEqual(found, "", "含中文的路径不该被返回，vosk 加载不了")
        joined = "\n".join(caught.output)
        self.assertIn("中文", joined)
        self.assertIn("B_STT_MODEL", joined, "提示要给出可照做的下一步")

    def test_ascii_fallback_dir_is_used_when_the_other_has_chinese(self):
        """中文目录里有模型、英文目录里也有 —— 应该选中英文那个。"""
        self._make_model(self._cn_base)
        good = self._make_model(self._ascii_base)

        with mock.patch.dict(os.environ, {"B_STT_MODEL": ""}):
            self.assertEqual(stt.find_model_dir(), good)

    def test_chinese_only_model_makes_vosk_unavailable(self):
        """只有中文路径可用时，引擎必须报告为不可用 ——
        否则 auto 会选中一个必然失败的识别器。"""
        self._make_model(self._cn_base)

        with mock.patch.dict(os.environ, {"B_STT_MODEL": ""}):
            self.assertFalse(stt._vosk_ready())

    def test_explicit_env_var_with_chinese_warns_but_is_honoured(self):
        """显式指定时**不**替用户否决，但要先提醒一声。

        和 readiness 的取舍一致：用户明确要的东西就给他试的机会，
        我们只负责把可能的原因说在前面。
        """
        model = self._make_model(self._cn_base)
        with mock.patch.dict(os.environ, {"B_STT_MODEL": model}):
            with self.assertLogs("core.voice.stt", level="WARNING") as caught:
                found = stt.find_model_dir()
            self.assertEqual(found, model)
            self.assertIn("B_STT_MODEL", "\n".join(caught.output))


class TestDefaultModelsDirLocation(unittest.TestCase):
    """默认模型目录必须落在 ``backend_B/models``。

    单独一类、且**不 patch** ``DEFAULT_MODELS_DIR`` ——
    上面那组测试的 setUp 会把它换成临时目录，在那种环境里断言
    "默认目录在哪"等于在验证测试自己设的假值，什么也证明不了。

    这里从 ``core`` 包的位置反推，而不是照抄实现里的三层 ``dirname``：
    照抄的话实现要是少写一层 dirname，测试也会跟着一起错，正好把 bug 让过去。
    """

    def test_default_models_dir_is_backend_b_models(self):
        import core

        # 用 __path__[0]（包目录本身），不要用 dirname(core.__file__) ——
        # __file__ 已经是 core/__init__.py，再取一层 dirname 只会得到
        # backend_B/core，于是断言的是个错的值（这个错我犯过一次）。
        core_dir = os.path.abspath(core.__path__[0])
        backend_b = os.path.dirname(core_dir)
        self.assertEqual(
            stt.DEFAULT_MODELS_DIR, os.path.join(backend_b, "models"),
            "默认模型目录跑偏了，提示信息会把用户指到一个不存在的地方",
        )


class TestResolveOptional(unittest.TestCase):
    def test_known_stdlib_module_is_found(self):
        self.assertEqual(resolve_optional("json"), "json")

    def test_nonexistent_module_returns_none(self):
        self.assertIsNone(resolve_optional("definitely_not_a_real_module_xyz"))

    def test_returns_none_instead_of_raising_for_broken_module(self):
        """有些包名会让 find_spec 抛异常（名字非法、命名空间冲突）。

        我们的约定是"探测失败 = 不可用"，而不是把异常抛给调用方。
        """
        self.assertIsNone(resolve_optional(""))


# --------------------------------------------------------------------------
# TTS 注册表
# --------------------------------------------------------------------------

class TestSynthesizerRegistry(unittest.TestCase):

    def test_unknown_name_falls_back_without_raising(self):
        engine = tts.build_synthesizer("不存在的引擎")
        self.assertIsInstance(engine, NullSynthesizer)

    def test_null_synthesizer_is_inert(self):
        engine = tts.build_synthesizer("none")
        self.assertFalse(engine.synthesize_wav("你好", "x.wav"))
        engine.close()
        engine.close()

    def test_describe_helper(self):
        self.assertEqual(describe([NullRecognizer(), NullSynthesizer()]), "null, null")
        self.assertEqual(describe([]), "（无）")


class TestShutdownIsNotAFailure(unittest.TestCase):
    """退出流程里中断的合成**不是故障**，不该报 WARNING。

    踩到的现场：`Ctrl+C` 退出时正好有句话在合成，日志里出现

        语音合成失败（子进程返回 None）

    但什么都没坏 —— 那个 None 是我们自己 kill 掉 PowerShell 时，
    读取线程推的哨兵。**误导性的告警比没有告警更糟**：
    它会训练人忽略这个logger 的 warning，于是真正的合成失败也被一起忽略。

    所以这里用 `_closing` 把两种"子进程没了"区分开。
    不真的起 PowerShell（`_start` 被 patch 掉），全部走假进程。
    """

    def make_synth(self):
        """造一个"已就绪"的假 SAPI 合成器：不启动真进程。"""
        with mock.patch.object(tts.SapiSynthesizer, "_start"):
            synth = tts.SapiSynthesizer()
        synth._usable = True
        synth._process = mock.MagicMock()
        synth._process.poll.return_value = None      # 看起来还活着
        self.addCleanup(synth._terminate)
        return synth

    def test_closing_flag_starts_false(self):
        self.assertFalse(self.make_synth()._closing)

    def test_close_sets_the_closing_flag(self):
        synth = self.make_synth()
        synth.close()
        self.assertTrue(synth._closing)

    def test_start_resets_the_closing_flag(self):
        """重启之后我们就不再是"正在关闭"了。

        ⚠️ 不能 patch 掉 `_start`：那样它**整个函数**都不执行，
        重置标志的那行也跟着不执行，测试就成了自说自话。
        改成把平台伪装成非 Windows —— 真正的 `_start` 会跑，
        先重置标志，然后在那行 `if sys.platform != "win32"` 上安全返回。
        """
        synth = self.make_synth()
        synth._closing = True
        with mock.patch.object(tts.sys, "platform", "linux"):
            synth._start(1.0)
        self.assertFalse(synth._closing)

    def test_interrupted_synthesis_during_shutdown_logs_info_not_warning(self):
        """关停时被打断 → INFO；不能是 WARNING。"""
        synth = self.make_synth()
        synth._lines.put(None)                       # 哨兵：进程没了
        # ⚠️ 直接置标志，不要调 _terminate() —— 它顺手把 _usable 也置成 False，
        #    于是 synthesize_wav 在开头就返回，根本走不到要测的那个分支。
        synth._closing = True

        with self.assertLogs("core.voice.tts", level="INFO") as caught:
            ok = synth.synthesize_wav("你好", "x.wav")

        self.assertFalse(ok)
        joined = "\n".join(caught.output)
        self.assertIn("正在退出", joined)
        self.assertNotIn("WARNING", joined,
                         "退出流程打 WARNING 会让人以为真坏了")

    def test_unexpected_process_death_still_warns(self):
        """**不是**我们关的（子进程自己崩了）→ 必须仍然报警。

        这条是上面那条的反面：如果实现为了"安静"把所有 None 都降成 INFO，
        就再也发现不了 PowerShell 进程意外死掉。
        """
        synth = self.make_synth()
        synth._lines.put(None)
        self.assertFalse(synth._closing, "前提不成立：此时不该处于关闭状态")

        with self.assertLogs("core.voice.tts", level="WARNING") as caught:
            ok = synth.synthesize_wav("你好", "x.wav")

        self.assertFalse(ok)
        self.assertIn("语音合成失败", "\n".join(caught.output))


# --------------------------------------------------------------------------
# SAPI 命令构造 —— 纯函数半边
# --------------------------------------------------------------------------

class TestSapiCommand(unittest.TestCase):
    """SAPI 适配器里唯一能确定性测试的部分，但也是**最容易写错**的部分。"""

    def test_argv_uses_encoded_command(self):
        argv = tts.build_sapi_argv("Write-Output 'hi'")
        self.assertIn("-EncodedCommand", argv)
        self.assertIn("powershell", argv[0])

    def test_argv_contains_no_literal_non_ascii(self):
        """命令行上不能出现任何非 ASCII 字符。

        这是整个方案的目的：脚本里有中文音色名和一堆引号，直接拼命令行
        会被 PowerShell 的转义规则撕碎，而且报错完全指不出原因。
        """
        script = tts.build_sapi_script(rate=-2, voice_hint="Microsoft 中文 音色")
        argv = tts.build_sapi_argv(script)
        for index, argument in enumerate(argv):
            with self.subTest(index=index):
                self.assertTrue(
                    argument.isascii(),
                    f"argv[{index}] 含非 ASCII：{argument[:60]!r}",
                )

    def test_encoded_command_round_trips_as_utf16le(self):
        """base64 必须按 **UTF-16LE** 解回来。

        用 UTF-8 编码的话命令行上看也是 ASCII，但 PowerShell 解出来是乱码，
        而且要到运行时才暴露 —— 所以这条断言必须存在。
        """
        script = tts.build_sapi_script(rate=1, voice_hint="测试音色")
        argv = tts.build_sapi_argv(script)
        encoded = argv[argv.index("-EncodedCommand") + 1]

        self.assertEqual(tts.decode_ps_command(encoded), script)
        self.assertIn("测试音色", tts.decode_ps_command(encoded))

    def test_script_sets_rate_from_argument(self):
        for rate in (-10, -2, 0, 3, 10):
            with self.subTest(rate=rate):
                self.assertIn(f"$synth.Rate = {rate}", tts.build_sapi_script(rate=rate))

    def test_script_uses_synchronous_speak(self):
        """必须是同步的 ``Speak()``。

        COM 的 ``SpeakAsync`` 可能在写盘完成前就返回，得到被截断的 WAV ——
        「文件存在、时长非零、但内容是半句」是最难查的一类 bug。
        """
        code = executable_lines(tts.build_sapi_script())
        self.assertIn("$synth.Speak($text)", code)
        self.assertNotIn("SpeakAsync", code)

    def test_script_forces_wav_format(self):
        """必须强制 16000Hz/16bit/单声道。

        默认输出可能是 22.05kHz 或 WAVE_FORMAT_EXTENSIBLE，
        标准库 wave 读不了。
        """
        script = tts.build_sapi_script()
        self.assertIn("SpeechAudioFormatInfo", script)
        self.assertIn("16000", script)
        self.assertIn("Sixteen", script)
        self.assertIn("Mono", script)

    def test_script_selects_voice_by_culture_prefix(self):
        """音色要按 Culture 前缀挑，不能硬编码名字。

        硬编码 'Microsoft Huihui Desktop' 在这台机器上有，换一台就抛异常。
        """
        code = executable_lines(tts.build_sapi_script())
        self.assertIn("GetInstalledVoices", code)
        self.assertIn("Culture.Name -like 'zh*'", code)
        self.assertNotIn("Huihui", code)

    def test_voice_hint_is_escaped(self):
        """音色名里的单引号必须转义，否则会破坏 PowerShell 字符串。"""
        script = tts.build_sapi_script(voice_hint="It's a voice")
        self.assertIn("It''s a voice", script)

    def test_script_reports_ready_and_ok(self):
        script = tts.build_sapi_script()
        self.assertIn("READY", script)
        self.assertIn("OK", script)
        self.assertIn("ERR", script)

    def test_script_loops_on_stdin(self):
        """必须是常驻循环，而不是一次一进程。

        实测每次 spawn + Add-Type 要 602ms 死寂 —— 那样每句话前都要卡半秒。
        """
        script = tts.build_sapi_script()
        self.assertIn("while ($true)", script)
        self.assertIn("[Console]::In.ReadLine()", script)

    def test_script_resets_output_after_each_utterance(self):
        """每句之后都要 SetOutputToNull，否则下一次 SetOutputToWaveFile 会失败。"""
        self.assertIn("SetOutputToNull", tts.build_sapi_script())


class TestSpeakRequestEncoding(unittest.TestCase):
    """stdin 请求行：之所以再 base64 一次，是为了让管道上全 ASCII。"""

    def test_round_trip_with_chinese(self):
        line = tts.format_speak_request("C:/tmp/a.wav", "您今天气色不错！")
        path, text = tts.parse_speak_request(line)
        self.assertEqual(path, "C:/tmp/a.wav")
        self.assertEqual(text, "您今天气色不错！")

    def test_request_line_is_pure_ascii(self):
        """管道上必须是 ASCII。

        Python 按 UTF-8 写管道，而 PowerShell 读控制台输入用的是系统 OEM
        代码页（本机 GBK）—— 直接写中文必然乱码。
        """
        line = tts.format_speak_request("a.wav", "中文测试 with $pecial `chars`")
        self.assertTrue(line.isascii())

    def test_empty_text_round_trips(self):
        _, text = tts.parse_speak_request(tts.format_speak_request("a.wav", ""))
        self.assertEqual(text, "")

    def test_tab_separates_path_from_payload(self):
        """用制表符分隔：路径里可能有空格，不能按空格切。"""
        line = tts.format_speak_request("C:/Program Files/a b.wav", "你好")
        self.assertEqual(line.count("\t"), 1)
        path, _ = tts.parse_speak_request(line)
        self.assertEqual(path, "C:/Program Files/a b.wav")


class TestSpeedMapping(unittest.TestCase):

    def test_normal_speed_is_zero(self):
        self.assertEqual(tts.speed_to_sapi_rate(1.0), 0)

    def test_slower_gives_negative_rate(self):
        self.assertLess(tts.speed_to_sapi_rate(0.85), 0)

    def test_faster_gives_positive_rate(self):
        self.assertGreater(tts.speed_to_sapi_rate(1.3), 0)

    def test_clamped_to_sapi_range(self):
        """SAPI 的 Rate 只接受 -10~10，超了会抛异常。"""
        self.assertEqual(tts.speed_to_sapi_rate(5.0), 10)
        self.assertEqual(tts.speed_to_sapi_rate(0.01), -10)

    def test_garbage_input_does_not_raise(self):
        for value in (None, "abc", object()):
            with self.subTest(value=value):
                self.assertEqual(tts.speed_to_sapi_rate(value), 0)

    def test_elderly_default_speed_is_slower(self):
        """默认语速给老人要偏慢（对齐 actions.py 的 0.85）。"""
        self.assertLessEqual(tts.speed_to_sapi_rate(0.85), 0)


# --------------------------------------------------------------------------
# 播报器
# --------------------------------------------------------------------------

class TestSpeaker(unittest.TestCase):

    def setUp(self):
        self.synth = FakeSynthesizer()
        self.player = FakePlayer()
        self.busy_log = []
        self.speaker = Speaker(
            synthesizer=self.synth,
            player=self.player,
            on_busy_change=self.busy_log.append,
        )

    def tearDown(self):
        self.speaker.close()

    def test_say_synthesizes_and_plays(self):
        self.assertTrue(self.speaker.say("您好呀"))
        self.assertEqual(self.synth.calls[0][0], "您好呀")
        self.assertEqual(len(self.player.played), 1)

    def test_empty_text_is_a_no_op(self):
        self.assertFalse(self.speaker.say(""))
        self.assertFalse(self.speaker.say("   "))
        self.assertFalse(self.speaker.say(None))
        self.assertEqual(self.synth.calls, [])
        self.assertEqual(self.player.played, [])

    def test_busy_flag_toggles_around_speech(self):
        self.speaker.say("你好")
        self.assertEqual(self.busy_log, [True, False])

    def test_busy_flag_clears_even_when_synthesis_fails(self):
        """合成失败也必须把 busy 清掉，否则机器人会永远"以为自己正在说话"，
        主动关怀再也不会触发。"""
        synth = FakeSynthesizer(succeed=False)
        speaker = Speaker(synthesizer=synth, player=self.player,
                          on_busy_change=self.busy_log.append)
        try:
            self.assertFalse(speaker.say("你好"))
            self.assertEqual(self.busy_log, [True, False])
            self.assertFalse(speaker.busy)
        finally:
            speaker.close()

    def test_failed_synthesis_does_not_play(self):
        synth = FakeSynthesizer(succeed=False)
        speaker = Speaker(synthesizer=synth, player=self.player)
        try:
            self.assertFalse(speaker.say("你好"))
            self.assertEqual(self.player.played, [])
        finally:
            speaker.close()

    def test_last_text_can_be_read_without_deadlock(self):
        """``last_text`` 在回声去重里被别的线程读。

        如果实现成"先拿锁再取值"，而 say() 正持锁播报，
        读 last_text 就会阻塞到整句话播完 —— 语音循环会卡死。
        这条测试用超时来暴露这种实现。
        """
        self.speaker.say("第一句")
        done = threading.Event()

        def reader():
            self.speaker.last_text
            done.set()

        self.player.delay = 0.0
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        self.assertTrue(done.wait(timeout=2.0), "读 last_text 被阻塞了")

    def test_say_is_serialised_across_threads(self):
        """两句并发的话不能叠在一起说。

        没有串行化的话，两个线程会同时播放，老人听到的是两句话糊在一起。
        """
        self.player.delay = 0.05
        threads = [
            threading.Thread(target=self.speaker.say, args=(f"第{i}句",), daemon=True)
            for i in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3.0)

        self.assertEqual(self.player.max_concurrent, 1, "出现了并发播放")
        self.assertEqual(len(self.player.played), 4)

    def test_half_duplex_stops_the_player_first(self):
        """半双工：播放前先打断上一次，避免两句话叠着。"""
        self.speaker.half_duplex = True
        self.speaker.say("你好")
        self.assertGreaterEqual(self.player.stopped, 1)

    def test_temp_files_are_cleaned_up(self):
        self.speaker.say("你好")
        self.speaker.say("再见")
        remaining = os.listdir(self.speaker._tempdir)
        self.assertEqual(remaining, [], f"临时文件没清掉：{remaining}")

    def test_close_is_idempotent(self):
        self.speaker.close()
        self.speaker.close()

    def test_engine_name_survives_missing_attribute(self):
        class Anonymous:
            def synthesize_wav(self, text, path):
                return False

            def close(self):
                pass

        speaker = Speaker(synthesizer=Anonymous())
        self.assertEqual(speaker.engine_name, "?")


class FakeStreamingSynthesizer(FakeSynthesizer):
    """既能落盘、也能"边收边吐"的假引擎。"""

    supports_streaming = True
    stream_rates = (24000,)

    def __init__(self, chunks=None, fail_before_first=False, fail_after=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.chunks = list(chunks if chunks is not None else [b"\x00\x01" * 50])
        self.fail_before_first = fail_before_first
        self.fail_after = fail_after
        self.stream_calls = []

    def synthesize_pcm_stream(self, text, sample_rate):
        self.stream_calls.append((text, sample_rate))
        if self.fail_before_first:
            raise ValueError("模拟：一块音频都没出来就失败")
        for index, chunk in enumerate(self.chunks):
            if self.fail_after is not None and index >= self.fail_after:
                raise ValueError("模拟：吐了几块之后断流")
            yield chunk


class FakeStreamPlayer(FakePlayer):
    """支持流式播放的假播放器。``rate=None`` 表示"设备与引擎谈不拢"。"""

    def __init__(self, rate=24000):
        super().__init__()
        self.rate = rate
        self.streamed = []
        self.rate_queries = []

    def stream_rate(self, candidates):
        self.rate_queries.append(tuple(candidates))
        return self.rate if self.rate in tuple(candidates) else None

    def play_pcm_stream(self, chunks, sample_rate):
        try:
            for chunk in chunks:
                self.streamed.append(chunk)
        except Exception:
            # 和真 Player 一致：异常吃掉，只回答"有没有出过声"。
            pass
        return bool(self.streamed)


class TestSpeakerAsyncSay(unittest.TestCase):
    """应声词：异步垫一句，正式回复必须排在它**后面**。

    这一组盯着的是**顺序**，而不是"能不能异步"。用 Lock 抢锁的写法能通过
    "异步"的所有断言，却会让老人听到"回答……嗯，我听着呢"。
    """

    def setUp(self):
        self.synth = FakeSynthesizer()
        self.player = FakePlayer()
        self.speaker = Speaker(synthesizer=self.synth, player=self.player)
        self.addCleanup(self.speaker.close)

    def test_async_say_returns_immediately(self):
        """**不能阻塞。** 调用它的线程紧接着要去做大模型的网络请求，
        在这里等就等于把应声词变成了对正式回复的额外延迟。"""
        self.player.delay = 0.3
        started = time.monotonic()
        self.assertTrue(self.speaker.say_async("嗯，我听着呢。"))
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.15, "say_async 阻塞了调用线程")
        self.speaker._join_async()

    def test_reply_waits_for_the_ack(self):
        """核心断言：合成顺序必须是 应声词 → 正式回复。"""
        self.player.delay = 0.05
        self.assertTrue(self.speaker.say_async("嗯，我听着呢。"))
        time.sleep(0.01)                 # 给后台线程一点时间先拿到锁
        self.speaker.say("您看着有点乏了。")
        self.assertEqual([text for text, _ in self.synth.calls],
                         ["嗯，我听着呢。", "您看着有点乏了。"],
                         "正式回复抢在应声词前面播了")

    def test_reply_waits_even_if_the_worker_has_not_started_yet(self):
        """应声线程还没被调度时也不能乱序 —— 靠的不是"抢得快"。"""
        self.player.delay = 0.05
        self.speaker.say_async("嗯，让我想想。")
        self.speaker.say("好的。")        # 立刻，不给后台线程任何机会
        self.assertEqual([text for text, _ in self.synth.calls],
                         ["嗯，让我想想。", "好的。"])

    def test_a_second_ack_is_dropped_while_one_is_playing(self):
        """上一句应声还没说完又来一句：丢掉，否则连成"嗯，我想想，嗯，我想想"。"""
        self.player.delay = 0.2
        self.assertTrue(self.speaker.say_async("嗯，我听着呢。"))
        self.assertFalse(self.speaker.say_async("嗯，让我想想。"))
        self.speaker._join_async()
        self.assertEqual([text for text, _ in self.synth.calls], ["嗯，我听着呢。"])

    def test_empty_text_is_a_no_op(self):
        self.assertFalse(self.speaker.say_async(""))
        self.assertFalse(self.speaker.say_async("   "))
        self.assertFalse(self.speaker.say_async(None))
        self.assertEqual(self.synth.calls, [])

    def test_busy_flag_is_cleared_after_async_speech(self):
        """应声词说完必须把 busy 清掉 —— 挂着不清，主动关怀就永远不再触发。"""
        self.speaker.say_async("嗯，我听着呢。")
        self.speaker._join_async()
        self.assertFalse(self.speaker.busy)

    def test_busy_flag_is_cleared_even_if_synthesis_fails(self):
        speaker = Speaker(synthesizer=FakeSynthesizer(succeed=False),
                          player=self.player)
        self.addCleanup(speaker.close)
        speaker.say_async("嗯，我听着呢。")
        speaker._join_async()
        self.assertFalse(speaker.busy)


class TestSpeakerStreaming(unittest.TestCase):
    """边收边播：缓存没命中时，第一块音频到了就开口，不等整句合成完。"""

    def setUp(self):
        self.synth = FakeStreamingSynthesizer()
        self.player = FakeStreamPlayer()
        self.speaker = Speaker(synthesizer=self.synth, player=self.player)
        self.addCleanup(self.speaker.close)

    def test_streams_when_the_engine_supports_it(self):
        self.assertTrue(self.speaker.say("您好呀"))
        self.assertEqual(len(self.synth.stream_calls), 1)
        self.assertEqual(self.synth.calls, [], "不该再落回整句合成")

    def test_negotiated_rate_is_passed_through(self):
        self.speaker.say("您好呀")
        self.assertEqual(self.synth.stream_calls[0][1], 24000)
        self.assertEqual(self.player.rate_queries, [(24000,)])

    def test_engine_without_streaming_uses_the_old_path(self):
        """SAPI 这类只落盘的引擎照旧走老路。"""
        synth = FakeSynthesizer()               # 没有 supports_streaming
        speaker = Speaker(synthesizer=synth, player=self.player)
        self.addCleanup(speaker.close)
        self.assertTrue(speaker.say("您好呀"))
        self.assertEqual(len(synth.calls), 1)

    def test_falls_back_when_rates_do_not_overlap(self):
        """设备采样率与引擎档位没有交集 → 整句合成。

        硬开一个设备开不了的采样率会抛 PortAudioError，比慢一点糟得多。
        """
        player = FakeStreamPlayer(rate=None)
        speaker = Speaker(synthesizer=self.synth, player=player)
        self.addCleanup(speaker.close)
        self.assertTrue(speaker.say("您好呀"))
        self.assertEqual(self.synth.stream_calls, [])
        self.assertEqual(len(self.synth.calls), 1, "没有落回整句合成")

    def test_no_player_means_no_streaming(self):
        """``--no-play``：没有播放端就无所谓"边收边播"，但仍要**合成**。

        返回 True 是既有的语义 —— ``_play_locked`` 在"已合成但没有播放设备"
        这条分支上返回 True（合成成功 ≠ 说了话，但也不是失败）。
        """
        speaker = Speaker(synthesizer=self.synth, player=None)
        self.addCleanup(speaker.close)
        self.assertTrue(speaker.say("您好呀"))
        self.assertEqual(self.synth.stream_calls, [])
        self.assertEqual(len(self.synth.calls), 1)

    def test_falls_back_when_nothing_was_played(self):
        """一块都没吐出来就失败 —— 干净的失败，落回整句合成（那条路带兜底）。"""
        synth = FakeStreamingSynthesizer(fail_before_first=True)
        speaker = Speaker(synthesizer=synth, player=self.player)
        self.addCleanup(speaker.close)
        self.assertTrue(speaker.say("您好呀"))
        self.assertEqual(len(synth.calls), 1, "没有落回整句合成")

    def test_does_not_replay_after_audio_was_already_heard(self):
        """**已经出声之后**断流：绝不能重念。

        这是流式改造里唯一会直接伤到用户的错误 —— 老人会听到同一句话
        被念两遍（而且第二遍还是从头开始）。
        """
        synth = FakeStreamingSynthesizer(
            chunks=[b"\x00\x01" * 10] * 4, fail_after=1)
        speaker = Speaker(synthesizer=synth, player=self.player)
        self.addCleanup(speaker.close)
        speaker.say("您好呀")
        self.assertEqual(synth.calls, [], "已经出声了还重念了一遍")

    def test_cache_hit_never_opens_a_stream(self):
        """缓存命中就直接播文件 —— 那句已经在本机了，没有任何理由再走网络。"""
        with tempfile.TemporaryDirectory() as tmp:
            cache = speech_cache.SpeechCache(tmp, "fake|voice|1.0")
            warm = os.path.join(tmp, "warm.wav")
            write_wav(warm)
            cache.store("您好呀", warm)

            speaker = Speaker(synthesizer=self.synth, player=self.player,
                              cache=cache, cache_allow=["您好呀"])
            self.addCleanup(speaker.close)
            self.assertTrue(speaker.say("您好呀"))
            self.assertEqual(self.synth.stream_calls, [])
            self.assertEqual(self.synth.calls, [])


class TestDoubaoStreamParser(unittest.TestCase):
    """流式解析器：喂一行吐一块。协议的细节全在这里。"""

    @staticmethod
    def _audio(payload: bytes) -> str:
        return json.dumps(
            {"code": 0, "data": base64.b64encode(payload).decode("ascii")})

    def _feed(self, parser, lines):
        out = []
        for line in lines:
            chunk = parser.feed(line)
            if chunk:
                out.append(chunk)
        return b"".join(out)

    def test_basic_stream(self):
        parser = tts.DoubaoStreamParser()
        blob = self._feed(parser, [
            self._audio(b"aaa"),
            self._audio(b"bbb"),
            json.dumps({"code": tts.DOUBAO_DONE_CODE, "message": "OK"}),
        ])
        self.assertEqual(blob, b"aaabbb")
        self.assertTrue(parser.done)
        self.assertEqual(parser.chunks, 2)
        self.assertEqual(parser.failure, "")

    def test_sse_prefix_and_event_lines_are_tolerated(self):
        parser = tts.DoubaoStreamParser()
        blob = self._feed(parser, [
            "event: message",
            "data: " + self._audio(b"zzz"),
            "data: [DONE]",
            "",
            "   ",
        ])
        self.assertEqual(blob, b"zzz")

    def test_error_code_is_recorded_not_raised(self):
        """流式里报错**不抛** —— 音频可能已经进耳朵了，抛只会让调用方重念。"""
        parser = tts.DoubaoStreamParser()
        parser.feed(self._audio(b"aaa"))
        parser.feed(json.dumps({"code": 45000010, "message": "invalid speaker"}))
        self.assertEqual(parser.chunks, 1)
        self.assertIn("45000010", parser.failure)
        self.assertIn("invalid speaker", parser.failure)

    def test_garbage_lines_are_skipped(self):
        parser = tts.DoubaoStreamParser()
        blob = self._feed(parser, [
            "not json", "{", "", "event: x", "data:", self._audio(b"ok")])
        self.assertEqual(blob, b"ok")
        self.assertEqual(parser.failure, "")

    def test_done_is_not_audio(self):
        parser = tts.DoubaoStreamParser()
        self.assertIsNone(
            parser.feed(json.dumps({"code": tts.DOUBAO_DONE_CODE})))
        self.assertTrue(parser.done)
        self.assertEqual(parser.chunks, 0)


class TestSpeakerLogging(unittest.TestCase):
    """播报的日志必须能区分「说了」「没合成出来」「播了但没人听见」。

    这一组是为了一个具体的事故写的：主动关怀在日志里出现了一次，
    但**没有任何迹象表明它有没有真的出声** —— 成功路径完全不记日志，
    而合成失败记的是 DEBUG（默认级别下看不见）。
    结果是「机器人哑了」和「机器人说了话」在日志里长得一模一样。

    真正的难点不是"加一行日志"，而是**「播放成功但听不见」**：
    数据进了虚拟声卡时 ``play_wav`` 会**立刻返回 True**，不报任何错
    （本机就装着 ToDesk Virtual Audio 这个设备）。
    只看成功/失败永远发现不了，所以要看**播放耗时与音频长度的比值**。
    """

    def _speaker(self, synth=None, player=None):
        speaker = Speaker(synthesizer=synth or FakeSynthesizer(),
                          player=player if player is not None else FakePlayer())
        self.addCleanup(speaker.close)
        return speaker

    # ---- 成功路径不再是静默的 -------------------------------------------

    def test_successful_speech_is_logged(self):
        """说了话就一定要有日志，否则根本无法事后确认。"""
        speaker = self._speaker()
        with self.assertLogs("core.voice.loop", level="INFO") as caught:
            self.assertTrue(speaker.say("你好呀"))
        self.assertIn("已播报", "\n".join(caught.output))

    def test_log_contains_the_spoken_text(self):
        """日志要能对上是哪句话 —— 否则一堆"已播报"等于没记。"""
        speaker = self._speaker()
        with self.assertLogs("core.voice.loop", level="INFO") as caught:
            speaker.say("我一直在旁边")
        self.assertIn("我一直在旁边", "\n".join(caught.output))

    # ---- 合成失败必须是 WARNING（本次修复的核心） -----------------------

    def test_synthesis_failure_is_a_warning_not_debug(self):
        """合成失败 = 一个字都没说出去，必须在默认级别下可见。

        ``assertLogs(level="WARNING")`` 在实现退回 DEBUG 时会**直接失败**，
        这正是我们要钉住的回归：曾经的实现就是 DEBUG，于是"机器人突然哑了"
        在日志里什么都没有。
        """
        speaker = self._speaker(synth=FakeSynthesizer(succeed=False))
        with self.assertLogs("core.voice.loop", level="WARNING") as caught:
            self.assertFalse(speaker.say("这句话说不出来"))
        joined = "\n".join(caught.output)
        self.assertIn("合成失败", joined)
        self.assertIn("这句话说不出来", joined)

    def test_playback_failure_is_a_warning(self):
        class FailingPlayer:
            def play_wav(self, path):
                return False

            def stop(self):
                pass

        speaker = self._speaker(player=FailingPlayer())
        with self.assertLogs("core.voice.loop", level="WARNING") as caught:
            self.assertFalse(speaker.say("播放会失败"))
        self.assertIn("播报失败", "\n".join(caught.output))

    # ---- 「播放成功但听不见」的判别 --------------------------------------

    def test_instant_playback_of_long_audio_warns(self):
        """立刻"播完"一段长音频 => 大概率进了虚拟声卡，要报警。

        这是本组最有价值的一条：``play_wav`` 返回 True、没有任何异常，
        只有耗时对不上。FakePlayer 不 sleep，所以等价于"数据被丢进黑洞"。
        """
        synth = FakeSynthesizer()
        synth.synthesize_wav = lambda text, path: (
            write_wav(path, frames=16000 * 3), True)[1]   # 3 秒的音频

        speaker = self._speaker(synth=synth, player=FakePlayer())
        with self.assertLogs("core.voice.loop", level="WARNING") as caught:
            self.assertTrue(speaker.say("一段三秒的音频"))
        joined = "\n".join(caught.output)
        self.assertIn("耗时异常", joined)
        self.assertIn("虚拟声卡", joined)

    def test_real_time_playback_of_long_audio_does_not_warn(self):
        """按真实时长播完就不该报警 —— 否则这条警告会被训练成噪声。

        FakePlayer 睡满音频长度，模拟真的出了声。
        """
        synth = FakeSynthesizer()
        synth.synthesize_wav = lambda text, path: (
            write_wav(path, frames=16000), True)[1]       # 1 秒的音频

        player = FakePlayer()
        player.delay = 1.0
        speaker = self._speaker(synth=synth, player=player)
        with self.assertLogs("core.voice.loop", level="INFO") as caught:
            self.assertTrue(speaker.say("一秒的音频"))
        joined = "\n".join(caught.output)
        self.assertNotIn("耗时异常", joined)
        self.assertIn("已播报", joined)

    def test_short_audio_never_triggers_the_timing_warning(self):
        """短音频（默认假 WAV 只有 0.1 秒）不参与耗时判断。

        理由是耗时测量本身有毫秒级噪声，拿 0.1 秒的音频去比会误报 ——
        阈值设在 0.3 秒就是为了让这类短句走"正常"分支。
        """
        speaker = self._speaker()
        with self.assertLogs("core.voice.loop", level="INFO") as caught:
            self.assertTrue(speaker.say("短句"))
        self.assertNotIn("耗时异常", "\n".join(caught.output))

    # ---- 无播放设备（--no-play）不能和"说了"混为一谈 --------------------

    def test_no_player_logs_synthesized_but_not_played(self):
        speaker = Speaker(synthesizer=FakeSynthesizer(), player=None)
        self.addCleanup(speaker.close)
        with self.assertLogs("core.voice.loop", level="INFO") as caught:
            self.assertTrue(speaker.say("只合成"))
        joined = "\n".join(caught.output)
        self.assertIn("未播放", joined)
        self.assertNotIn("已播报：", joined)


class TestLogHelpers(unittest.TestCase):
    """日志辅助函数本身也要测 —— 它们出错会让上面那组断言失去意义。"""

    def test_brief_truncates_long_text(self):
        brief = _brief("一" * 100)
        self.assertLess(len(brief), 100)
        self.assertIn("…", brief)

    def test_brief_keeps_short_text_intact(self):
        self.assertEqual(_brief("你好"), repr("你好"))

    def test_brief_handles_empty_and_none(self):
        """宁可在日志里少几个字，也不能因为日志把播报搞崩。"""
        self.assertEqual(_brief(""), repr(""))
        self.assertEqual(_brief(None), repr(""))

    def test_brief_flattens_newlines(self):
        """多行文本会把日志撑成好几行，破坏 grep 的可读性。"""
        self.assertNotIn("\n", _brief("第一行\n第二行"))

    def test_wav_seconds_reads_the_duration(self):
        path = os.path.join(self._tmpdir(), "d.wav")
        write_wav(path, frames=16000)                 # 16000 帧 @16kHz = 1.0s
        self.assertAlmostEqual(_wav_seconds(path), 1.0, places=3)

    def test_wav_seconds_on_garbage_returns_sentinel(self):
        """读不出来返回 -1 而不是抛异常 —— 日志不该让播报失败。"""
        path = os.path.join(self._tmpdir(), "bad.wav")
        with open(path, "wb") as handle:
            handle.write(b"not a wav at all")
        self.assertEqual(_wav_seconds(path), -1.0)

    def test_wav_seconds_on_missing_file_returns_sentinel(self):
        self.assertEqual(_wav_seconds(os.path.join(self._tmpdir(), "nope.wav")), -1.0)

    def _tmpdir(self):
        if not hasattr(self, "_dir"):
            self._dir = tempfile.mkdtemp()
            self.addCleanup(shutil.rmtree, self._dir, ignore_errors=True)
        return self._dir


# --------------------------------------------------------------------------
# 回声与噪声过滤
# --------------------------------------------------------------------------

class TestEchoSuppression(unittest.TestCase):

    def test_identical_text_is_echo(self):
        self.assertTrue(looks_like_echo("您今天气色不错", "您今天气色不错"))

    def test_near_identical_text_is_echo(self):
        self.assertTrue(looks_like_echo("您今天气色不错啊", "您今天气色不错"))

    def test_different_text_is_not_echo(self):
        self.assertFalse(looks_like_echo("我今天有点累", "您今天气色不错"))

    def test_empty_inputs_are_not_echo(self):
        self.assertFalse(looks_like_echo("", "你好"))
        self.assertFalse(looks_like_echo("你好", ""))
        self.assertFalse(looks_like_echo("", ""))

    def test_short_agreement_is_not_echo(self):
        """老人的一句"对"不该被当成回声 —— 那会把真实的附和吞掉。"""
        self.assertFalse(looks_like_echo("对", "您说得对，身体比什么都当紧。"))

    def test_threshold_is_configurable(self):
        text = "一"
        other = "二三"
        self.assertFalse(looks_like_echo(text, other, threshold=0.9))


class TestMeaningfulFilter(unittest.TestCase):

    def test_empty_and_whitespace_rejected(self):
        for value in ("", "   ", "\n\t"):
            with self.subTest(value=value):
                self.assertFalse(is_meaningful(value))

    def test_bare_punctuation_rejected(self):
        """STT 在纯噪声上常吐出单个标点。"""
        for value in ("。", "。，", "！", "…", "、"):
            with self.subTest(value=value):
                self.assertFalse(is_meaningful(value))

    def test_single_character_rejected(self):
        """单个字多半是误触发，拿去对话只会得到莫名其妙的回复。"""
        self.assertFalse(is_meaningful("嗯"))
        self.assertFalse(is_meaningful("1"))

    def test_real_speech_accepted(self):
        for value in ("你好", "嗯嗯", "我有点累", "hello", "12"):
            with self.subTest(value=value):
                self.assertTrue(is_meaningful(value))

    def test_none_is_rejected(self):
        self.assertFalse(is_meaningful(None))


# --------------------------------------------------------------------------
# 语音循环（假设备，不碰真麦克风）
# --------------------------------------------------------------------------

class TestVoiceLoop(unittest.TestCase):

    def build(self, recognizer=None, recorder=None, config=None):
        self.synth = FakeSynthesizer()
        self.player = FakePlayer()
        self.speaker = Speaker(synthesizer=self.synth, player=self.player)
        self.received = []

        def on_text(text):
            self.received.append(text)
            return mock.Mock(reply="我听着呢")

        loop = VoiceLoop(
            on_text=on_text,
            speaker=self.speaker,
            recognizer=recognizer or FakeRecognizer(),
            recorder=recorder,
            config=config or LoopConfig(),
        )
        return loop

    def test_no_recorder_does_not_start(self):
        """没有录音设备时返回 False，但不抛异常。"""
        loop = self.build(recorder=None)
        self.assertFalse(loop.start())
        self.assertFalse(loop.alive)

    def test_recognised_text_reaches_the_callback(self):
        """识别结果必须原样送进 on_text —— 它是通往 handle_chat 的唯一入口。"""
        from core.voice.vad import VadConfig

        # 造一段足够长的"说话"音频让 VAD 切出一句
        pcm = b"\x00\x00" * 8000 + struct.pack("<h", 8000) * 16000 + b"\x00\x00" * 16000
        recorder = FakeRecorder([pcm])
        loop = self.build(recognizer=FakeRecognizer("我有点累"), recorder=recorder)

        self.assertTrue(loop.start())
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not self.received:
            time.sleep(0.02)
        loop.stop()

        self.assertEqual(self.received, ["我有点累"], "识别文本没有送进对话回调")

    def test_echo_is_dropped(self):
        """机器人自己的话被收回来时不能又触发一轮对话。"""
        speaker = Speaker(synthesizer=FakeSynthesizer(), player=FakePlayer())
        speaker._last_text = "您今天气色不错"       # 模拟刚说过

        received = []
        loop = VoiceLoop(
            on_text=lambda text: received.append(text) or mock.Mock(reply=""),
            speaker=speaker,
            recognizer=FakeRecognizer("您今天气色不错"),
            recorder=None,
        )
        from core.voice.vad import SpeechSegment

        loop._handle_segment(SpeechSegment(pcm=b"\x00" * 100, sample_rate=16000,
                                           reason="silence"))
        self.assertEqual(received, [], "回声没有被拦下")

    def test_meaningless_recognition_is_dropped(self):
        received = []
        loop = VoiceLoop(
            on_text=lambda text: received.append(text) or mock.Mock(reply=""),
            speaker=Speaker(synthesizer=FakeSynthesizer()),
            recognizer=FakeRecognizer("。"),
            recorder=None,
        )
        from core.voice.vad import SpeechSegment

        loop._handle_segment(SpeechSegment(pcm=b"\x00" * 100, sample_rate=16000,
                                           reason="silence"))
        self.assertEqual(received, [])

    def test_reply_is_spoken(self):
        """对话结果要朗读出来 —— 这是"全程语音"的闭环。"""
        synth, player = FakeSynthesizer(), FakePlayer()
        speaker = Speaker(synthesizer=synth, player=player)
        loop = VoiceLoop(
            on_text=lambda text: mock.Mock(reply="您先歇一会"),
            speaker=speaker,
            recognizer=FakeRecognizer("我有点累"),
            recorder=None,
        )
        from core.voice.vad import SpeechSegment

        loop._handle_segment(SpeechSegment(pcm=b"\x00" * 100, sample_rate=16000,
                                           reason="silence"))
        self.assertEqual(synth.calls[0][0], "您先歇一会")
        self.assertEqual(len(player.played), 1)

    def test_callback_exception_does_not_kill_the_loop(self):
        """对话逻辑抛异常时录/识别的循环要继续。

        否则一次异常就永久静音，而日志里只有一行堆栈。
        """
        loop = VoiceLoop(
            on_text=lambda text: (_ for _ in ()).throw(RuntimeError("模拟故障")),
            speaker=Speaker(synthesizer=FakeSynthesizer()),
            recognizer=FakeRecognizer("你好"),
            recorder=None,
        )
        from core.voice.vad import SpeechSegment

        loop._handle_segment(SpeechSegment(pcm=b"\x00" * 100, sample_rate=16000,
                                           reason="silence"))
        self.assertTrue(True, "异常被吞掉了，没有向外抛")

    def test_stop_is_idempotent_and_joins(self):
        recorder = FakeRecorder([b"\x00\x00" * 320] * 5)
        loop = self.build(recorder=recorder)
        loop.start()
        loop.stop()
        loop.stop()
        self.assertFalse(loop.alive)
        self.assertGreaterEqual(recorder.stopped, 1)

    def test_say_delegates_to_speaker(self):
        loop = self.build(recorder=None)
        loop.say("主动关怀的话")
        self.assertEqual(self.synth.calls[0][0], "主动关怀的话")


# --------------------------------------------------------------------------
# 豆包（火山引擎）在线合成
#
# 全程不联网：请求构造是纯函数，网络那一层用假 urlopen 顶掉，
# MP3 解码用假 soundfile 顶掉 —— 被测的是**我们的接线**，
# 不是 libsndfile 能不能解 MP3（那是它的作者要保证的事）。
# --------------------------------------------------------------------------

def fake_soundfile(rate: int = 16000, channels: int = 1, seconds: float = 0.1):
    """一个替身 soundfile 模块。塞进 sys.modules 即可被函数体内的 import 取到。"""
    import numpy

    module = types.ModuleType("soundfile")
    frames = int(rate * seconds)

    def read(_source, dtype="int16", always_2d=True):
        return numpy.zeros((frames, channels), dtype=dtype), rate

    module.read = read
    return module


def doubao_stream(chunks, *, done=True, extra=None):
    """拼一份 v3 接口的 JSON 行流。

    协议见 `tts.parse_doubao_stream`：**一行一个 JSON**，音频块 `code=0`、
    内容在 `data`（base64），最后一行 `code=DOUBAO_DONE_CODE` 收尾。
    `extra` 用来在音频块之间插一条错误事件，测"有音频但也有错"的情况。
    """
    lines = []
    for index, chunk in enumerate(chunks):
        lines.append(json.dumps({"code": 0, "data": chunk}))
        if extra is not None and index == 0:
            lines.append(json.dumps(extra))
    if done:
        lines.append(json.dumps({"code": tts.DOUBAO_DONE_CODE, "message": "OK"}))
    return ("\n".join(lines) + "\n").encode("utf-8")


def error_stream(code, message):
    """只有一条错误事件、没有任何音频的流。"""
    return (json.dumps({"code": code, "message": message}) + "\n").encode("utf-8")


def fake_http_response(raw):
    """假 urlopen 的返回值。`synthesize_wav` 用 `with` 包着它，所以要支持上下文。"""
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    response = mock.MagicMock()
    response.read.return_value = raw
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def ok_response(mp3: bytes = b"mp3"):
    """一份"合成成功"的响应：一个音频块 + 结束码。"""
    return fake_http_response(
        doubao_stream([base64.b64encode(mp3).decode("ascii")]))


def header_of(request, name: str):
    """按名字取一个 ``urllib.request.Request`` 上的头，**忽略大小写**。

    不能直接用 ``Request.get_header``：它按字面量查，而 urlib 存头的时候
    把键归一化成了 ``key.capitalize()``（``X-Api-Request-Id`` 会变成
    ``X-api-request-id``），于是照抄大小写查出来是 None —— 一个
    "头明明在，查却是空"的假失败。
    """
    for key, value in request.headers.items():
        if key.lower() == name.lower():
            return value
    return None


def make_doubao(**kwargs):
    """造一个"可用"的豆包合成器，不联网也不要求真的配了凭证。

    `__init__` 里那道资格关（凭证齐不齐、soundfile 在不在）另有专门的
    测试盯着，这里直接放行，好专心测合成本身。
    """
    credentials = {"api_key": "test-key", "voice": "test-voice"}
    credentials.update(kwargs)
    with mock.patch.object(tts, "resolve_optional", return_value=object()):
        synth = tts.DoubaoSynthesizer(**credentials)
    synth._usable = True
    return synth


class TestDoubaoRequest(unittest.TestCase):
    """请求构造：豆包适配器里唯一确定性的部分，也是最容易写错的部分。"""

    def test_authorization_header_is_not_used(self):
        """鉴权走 `X-Api-Key` 这一个头，**不要**混进 v1 的 Authorization。

        2026-09-29 实测：v3 接口收到 `Authorization: Bearer;<token>` 会回
        `no token or access_key was found from the header or query` ——
        一个和真实原因（头放错了）毫无关系的报错，够查半天。
        """
        headers = tts.build_doubao_headers("key-123", "seed-tts-2.0", "req-1")
        self.assertEqual(headers["X-Api-Key"], "key-123")
        self.assertNotIn("Authorization", headers)
        self.assertNotIn("Bearer", json.dumps(headers))

    def test_resource_id_header_carries_the_model_version(self):
        """ResourceId 决定调哪个模型。放错的话 2.0 的音色会配到 1.0 的模型上，
        而两者**不能混用**。"""
        headers = tts.build_doubao_headers("k", "seed-tts-2.0", "req-1")
        self.assertEqual(headers["X-Api-Resource-Id"], "seed-tts-2.0")
        self.assertEqual(headers["X-Api-Request-Id"], "req-1")

    def test_headers_are_pure_ascii(self):
        """请求头只能是 ASCII。凭证里混进中文（从控制台复制时很常见）
        会在这里暴露，而不是等到 urllib 抛一个看不懂的编码异常。"""
        headers = tts.build_doubao_headers("abc123", "seed-tts-2.0", "req-1")
        for name, value in headers.items():
            name.encode("ascii")          # 抛异常即失败
            value.encode("ascii")

    def test_endpoint_carries_no_text(self):
        """待合成的文本走 body，**不能**落在 URL 上。

        URL 会进代理日志、浏览器历史和各种中间件的记录，
        把用户对机器人说的话写进去等于到处留副本。
        """
        self.assertNotIn("?", tts.DOUBAO_ENDPOINT)
        self.assertTrue(tts.DOUBAO_ENDPOINT.startswith("https://"),
                        "API Key 在请求头里，明文 HTTP 会把它暴露在链路上")

    def test_endpoint_is_the_v3_seedtts_one(self):
        """端点必须指向 v3。指回 v1 的话，本项目的凭证会**永远**被拒
        （401 load grant: requested grant not found in SaaS storage），
        而且那个报错完全指不出真正的原因是这个。"""
        self.assertIn("/api/v3/", tts.DOUBAO_ENDPOINT)
        self.assertNotIn("/api/v1/", tts.DOUBAO_ENDPOINT)

    def test_body_has_the_two_required_sections(self):
        body = tts.build_doubao_request(
            text="你好", speaker="zh_female_vv_uranus_bigtts")
        for section in ("user", "req_params"):
            self.assertIn(section, body)

        self.assertEqual(body["req_params"]["text"], "你好")
        self.assertEqual(body["req_params"]["speaker"],
                         "zh_female_vv_uranus_bigtts")
        self.assertEqual(body["req_params"]["audio_params"]["format"], "mp3")
        self.assertEqual(body["req_params"]["audio_params"]["sample_rate"],
                         tts.DOUBAO_SAMPLE_RATE)

    def test_body_carries_no_credentials(self):
        """v3 的鉴权全在请求头里，body 里不该再出现任何凭证字段。

        留着 v1 的 app/token/cluster 只会让人以为还要配 appid ——
        而那正是这一轮排查里最费时间的一个误会。
        """
        body = tts.build_doubao_request(text="你好", speaker="s")
        rendered = json.dumps(body)
        self.assertNotIn("app", body)
        self.assertNotIn("appid", rendered)
        self.assertNotIn("cluster", rendered)

    def test_body_is_json_serialisable(self):
        body = tts.build_doubao_request(
            text="您好呀，今天天气不错。", speaker="s")
        json.dumps(body)                  # 抛异常即失败


class TestSpeechRate(unittest.TestCase):
    """倍数 → 服务端 speech_rate 的换算。两套单位不同，是容易写错的地方。"""

    def test_normal_speed_maps_to_zero(self):
        self.assertEqual(tts.speech_rate_from_speed(1.0), 0)

    def test_slower_speech_is_negative(self):
        """陪聊默认 0.9 倍 → -10，即"比正常慢 10%"。
        符号弄反会变成抢话，而这在听感上很明显、在代码里不明显。"""
        self.assertEqual(tts.speech_rate_from_speed(0.9), -10)

    def test_faster_speech_is_positive(self):
        self.assertEqual(tts.speech_rate_from_speed(1.2), 20)

    def test_out_of_range_speeds_are_clamped(self):
        """服务端只收 [-50, 100]。配置里写错一个数量级
        （比如填了 90 而不是 0.9）不该让整句话合成失败。"""
        self.assertEqual(tts.speech_rate_from_speed(90), 100)
        self.assertEqual(tts.speech_rate_from_speed(0.01), -50)

    def test_default_voice_speed_is_not_too_fast(self):
        """用户要求「语速保持正常，适合陪聊安慰对话，不要太快」。
        钉住默认值：调到 1.0 以上就违反了这条需求。"""
        import config
        self.assertLessEqual(config.VOICE_SPEED, 1.0)


class TestParseDoubaoStream(unittest.TestCase):
    """把 JSON 行流还原成 MP3。协议的细节全在这里，
    写错的两种后果都很隐蔽：**有声音但是半句**，或者**没声音却不报错**。
    """

    def test_chunks_are_concatenated_in_order(self):
        raw = doubao_stream([base64.b64encode(part).decode("ascii")
                             for part in (b"aaa", b"bbb", b"ccc")])
        self.assertEqual(tts.parse_doubao_stream(raw), b"aaabbbccc")

    def test_error_code_raises_even_though_http_was_200(self):
        """服务端出错时**同样是 HTTP 200**，所以判据必须在流里面。

        只看状态码的话，`app key not found` 会被当成"合成成功但没有音频"，
        然后表现为机器人哑掉而日志说一切正常。
        """
        with self.assertRaises(ValueError) as caught:
            tts.parse_doubao_stream(error_stream(45000010, "invalid speaker"))
        self.assertIn("45000010", str(caught.exception))
        self.assertIn("invalid speaker", str(caught.exception))

    def test_stream_without_any_audio_raises(self):
        with self.assertRaises(ValueError):
            tts.parse_doubao_stream(doubao_stream([]))

    def test_audio_with_an_error_in_the_middle_still_raises(self):
        """有音频、也有错 = 半句话。宁可整句失败，也不要把半句播出去 ——
        对着一位老人说半句就停，比不说更让人不安。"""
        raw = doubao_stream(
            [base64.b64encode(b"mp3").decode("ascii")],
            extra={"code": 45000010, "message": "invalid speaker"})
        with self.assertRaises(ValueError):
            tts.parse_doubao_stream(raw)

    def test_sse_data_prefix_is_tolerated(self):
        """SSE 版端点每行多一个 `data: ` 前缀。

        必须**先剥前缀再解析**：剥晚了那行就不是合法 JSON，
        会被当成音频内容拼进去，解出一堆乱码 —— 而这一步不会报错。
        """
        payload = base64.b64encode(b"mp3").decode("ascii")
        raw = ("event: message\ndata: %s\n\ndata: %s\n\n"
               % (json.dumps({"code": 0, "data": payload}),
                  json.dumps({"code": tts.DOUBAO_DONE_CODE,
                              "message": "OK"}))).encode("utf-8")
        self.assertEqual(tts.parse_doubao_stream(raw), b"mp3")

    def test_truncated_stream_without_done_code_still_returns_audio(self):
        """连接被掐断时可能收不到结束码。已经拿到的音频还能用就先用着，
        不要因为缺一个收尾标记把整句丢掉。"""
        raw = doubao_stream([base64.b64encode(b"mp3").decode("ascii")],
                            done=False)
        self.assertEqual(tts.parse_doubao_stream(raw), b"mp3")

    def test_junk_lines_are_ignored(self):
        """流的边界上偶尔有零散字符，不该让它把整句带崩。"""
        payload = base64.b64encode(b"mp3").decode("ascii")
        raw = ("not json at all\n\n%s\n"
               % json.dumps({"code": 0, "data": payload})).encode("utf-8")
        self.assertEqual(tts.parse_doubao_stream(raw), b"mp3")


class TestDoubaoAvailability(unittest.TestCase):
    """凭证没配齐时**不能**表现为"机器人突然不说话了"。"""

    def make(self, **patches):
        """在受控环境下构造：不受跑测试这台机器的环境变量影响。"""
        clean = {
            "TTS_DOUBAO_API_KEY": "",
            "TTS_DOUBAO_RESOURCE_ID": "",
            "TTS_DOUBAO_VOICE": "",
        }
        clean.update(patches)
        stack = [mock.patch.object(tts.config, key, value)
                 for key, value in clean.items()]
        for patcher in stack:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_missing_credentials_name_every_variable(self):
        """一个都没配时，两个变量的名字**都要**出现在提示里。

        用户的原话是「无凭证的时候，控制台明确打印缺失的变量名称」。
        只报第一个的话，人补上一个、再跑一次，才知道还缺第二个 ——
        这正是"挨个去猜"。
        """
        self.make()
        with mock.patch.object(tts, "resolve_optional", return_value=object()):
            synth = tts.DoubaoSynthesizer()

        self.assertFalse(synth.available)
        for name in ("B_DOUBAO_TOKEN", "B_DOUBAO_VOICE"):
            self.assertIn(name, synth.reason)

    def test_missing_api_key_says_where_to_get_it(self):
        """API Key 是最容易被误以为"不用填"的那一个，所以单独给取处。

        它和别的凭证不在同一页 —— 提示里点明控制台的哪一栏，
        能省掉一轮"这玩意到底在哪"。
        """
        self.make(TTS_DOUBAO_VOICE="v")
        with mock.patch.object(tts, "resolve_optional", return_value=object()):
            synth = tts.DoubaoSynthesizer()

        self.assertFalse(synth.available)
        self.assertIn("B_DOUBAO_TOKEN", synth.reason)
        self.assertIn("API Key", synth.reason)
        # 别的都配齐了，就别再念叨它们
        self.assertNotIn("B_DOUBAO_VOICE", synth.reason)

    def test_appid_is_deliberately_not_read(self):
        """v3 里没有 appid，所以 config **故意不读** B_DOUBAO_APPID。

        这条不是洁癖。本机的 B_DOUBAO_APPID 里确实躺着一个 33 位的值，
        很容易让人以为"少配了它才不发声" —— 实测把它当 X-Api-Key 发过去，
        服务端回的是 `Invalid X-Api-Key`。留着这个读法，
        只会把下一轮排查再带进同一个坑。
        """
        self.assertFalse(hasattr(tts.config, "TTS_DOUBAO_APPID"),
                         "config 里不该再有 TTS_DOUBAO_APPID —— v3 不用它")
        self.assertFalse(hasattr(tts.config, "TTS_DOUBAO_CLUSTER"),
                         "cluster 是 v1 的概念，v3 不用它")

    def test_resource_id_is_a_model_version_constant(self):
        """ResourceId 要的是**模型版本常量**，不是账号里的某个 id。

        它还必须和音色配套：Vivi 2.0 是 2.0 音色，模型也得是 2.0 ——
        配成 1.0 的话音色会被判不合法（invalid speaker）。
        """
        import config
        self.assertEqual(config.TTS_DOUBAO_RESOURCE_ID, "seed-tts-2.0")
        self.assertEqual(tts.DOUBAO_RESOURCE_ID,
                         config.TTS_DOUBAO_RESOURCE_ID)

    def test_resource_id_is_passed_through_to_the_request(self):
        """构造器收到的 resource_id 要真的进请求头，不能在中途丢掉。"""
        synth = make_doubao(resource_id="seed-tts-1.0")
        self.assertEqual(synth.resource_id, "seed-tts-1.0")

    def test_missing_credentials_do_not_raise(self):
        self.make()
        tts.DoubaoSynthesizer()           # 抛异常即失败

    def test_missing_soundfile_is_reported_at_startup(self):
        """没有 soundfile 就解不开 MP3，等于装了个不能用的引擎。
        这个事实要在**启动时**说清楚，而不是每次合成时才失败。"""
        self.make(TTS_DOUBAO_API_KEY="k", TTS_DOUBAO_VOICE="v")
        with mock.patch.object(tts, "resolve_optional", return_value=None):
            synth = tts.DoubaoSynthesizer()

        self.assertFalse(synth.available)
        self.assertIn("soundfile", synth.reason)

    def test_all_credentials_present_means_available(self):
        self.make(TTS_DOUBAO_API_KEY="k", TTS_DOUBAO_VOICE="v")
        with mock.patch.object(tts, "resolve_optional", return_value=object()):
            synth = tts.DoubaoSynthesizer()
        self.assertTrue(synth.available)
        self.assertEqual(synth.reason, "")

    def test_default_voice_is_the_vivi_companion_voice(self):
        """默认音色是 Vivi 2.0 陪聊音（zh_female_vv_uranus_bigtts）——
        长者陪伴场景选的，钉住它免得被谁顺手改掉。"""
        import config
        self.assertEqual(config.TTS_DOUBAO_VOICE,
                         "zh_female_vv_uranus_bigtts")


class TestDoubaoSynthesis(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp(prefix="b_doubao_test_")
        self.addCleanup(shutil.rmtree, self.tempdir, ignore_errors=True)
        self.wav_path = os.path.join(self.tempdir, "out.wav")

    def test_network_failure_returns_false_without_raising(self):
        """断网是常态，不该让对话中断，更不该把异常抛进播报线程。"""
        synth = make_doubao()
        with mock.patch("urllib.request.urlopen", side_effect=OSError("断网了")):
            with self.assertLogs("core.voice.tts", level="ERROR"):
                ok = synth.synthesize_wav("你好", self.wav_path)
        self.assertFalse(ok)

    def test_http_error_logs_the_server_message(self):
        """音色填错、凭证过期都会走到这里。只留一个数字状态码没法排查，
        要把服务端说的话原样带出来。"""
        import urllib.error

        synth = make_doubao()
        error = urllib.error.HTTPError(
            tts.DOUBAO_ENDPOINT, 401, "Unauthorized", {},
            io.BytesIO('{"message":"invalid voice_type"}'.encode("utf-8")))

        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertLogs("core.voice.tts", level="WARNING") as caught:
                ok = synth.synthesize_wav("你好", self.wav_path)

        self.assertFalse(ok)
        joined = "\n".join(caught.output)
        self.assertIn("401", joined)
        self.assertIn("invalid voice_type", joined)

    def test_error_code_is_a_failure_even_with_http_200(self):
        """服务端失败时走的**仍是 200 + 同一种流**，错误只在 message 里，
        所以不能只看状态码 —— 否则鉴权失败会被当成"合成成功只是没声音"。"""
        synth = make_doubao()
        with mock.patch("urllib.request.urlopen",
                        return_value=fake_http_response(
                            error_stream(45000010, "invalid speaker"))):
            with self.assertLogs("core.voice.tts", level="WARNING") as caught:
                ok = synth.synthesize_wav("你好", self.wav_path)

        self.assertFalse(ok)
        joined = "\n".join(caught.output)
        self.assertIn("45000010", joined)
        self.assertIn("invalid speaker", joined)

    def test_success_without_audio_data_is_a_failure(self):
        """只有结束码、一块音频都没有 —— 典型的"参数被接受但没出声"。

        这里最容易出的错是把它当成功：结束码是 20000000，看起来一切正常，
        然后写出一个 0 帧的 WAV，播放器静静地什么都不放。
        """
        synth = make_doubao()
        with mock.patch("urllib.request.urlopen",
                        return_value=fake_http_response(doubao_stream([]))):
            with self.assertLogs("core.voice.tts", level="WARNING"):
                ok = synth.synthesize_wav("你好", self.wav_path)
        self.assertFalse(ok)

    def test_successful_response_produces_a_valid_16k_mono_wav(self):
        synth = make_doubao()

        with mock.patch.dict(sys.modules,
                             {"soundfile": fake_soundfile(rate=16000)}):
            with mock.patch("urllib.request.urlopen",
                            return_value=ok_response()):
                ok = synth.synthesize_wav("你好", self.wav_path)

        self.assertTrue(ok)
        with wave.open(self.wav_path, "rb") as handle:
            self.assertEqual(handle.getframerate(), 16000)
            self.assertEqual(handle.getnchannels(), 1)
            self.assertEqual(handle.getsampwidth(), 2)
            self.assertEqual(handle.getnframes(), 1600)     # 0.1 秒

    def test_audio_split_across_chunks_is_reassembled(self):
        """真实响应是**十几块**音频，不是一块。

        拼接时少接一块，听感上就是"话说到一半断了" —— 而它不会报任何错。
        """
        synth = make_doubao()
        parts = [b"mp3", b"-part2", b"-part3"]
        whole = b"".join(parts)
        seen = {}

        def read(_source, dtype="int16", always_2d=True):
            import numpy
            seen["bytes"] = _source.read()
            return numpy.zeros((1600, 1), dtype=dtype), 16000

        module = types.ModuleType("soundfile")
        module.read = read

        with mock.patch.dict(sys.modules, {"soundfile": module}):
            with mock.patch("urllib.request.urlopen",
                            return_value=fake_http_response(
                                doubao_stream([base64.b64encode(p).decode("ascii")
                                               for p in parts]))):
                ok = synth.synthesize_wav("你好", self.wav_path)

        self.assertTrue(ok)
        self.assertEqual(seen["bytes"], whole)

    def test_other_sample_rates_are_resampled_to_16k(self):
        """豆包返回的采样率不由我们决定，下游（VAD/播放）要的是统一的 16k。"""
        synth = make_doubao()

        with mock.patch.dict(sys.modules,
                             {"soundfile": fake_soundfile(rate=24000)}):
            with mock.patch("urllib.request.urlopen",
                            return_value=ok_response()):
                ok = synth.synthesize_wav("你好", self.wav_path)

        self.assertTrue(ok)
        with wave.open(self.wav_path, "rb") as handle:
            self.assertEqual(handle.getframerate(), 16000)
            self.assertEqual(handle.getnframes(), 1600)     # 2400 × 16/24

    def test_stereo_response_is_downmixed_to_mono(self):
        synth = make_doubao()

        with mock.patch.dict(sys.modules,
                             {"soundfile": fake_soundfile(channels=2)}):
            with mock.patch("urllib.request.urlopen",
                            return_value=ok_response()):
                ok = synth.synthesize_wav("你好", self.wav_path)

        self.assertTrue(ok)
        with wave.open(self.wav_path, "rb") as handle:
            self.assertEqual(handle.getnchannels(), 1)

    def test_request_id_is_fresh_for_every_call(self):
        """reqid 复用可能拿到上一次的缓存结果甚至被拒，每次合成都要换新的。

        它还必须**头和 body 里是同一个值**：服务端拿它对账，
        两处对不上时，出问题就只能看着两个不相干的 id 发呆。
        """
        synth = make_doubao()
        sent = []

        def capture(request, timeout=None):
            sent.append(request)
            return ok_response()

        with mock.patch.dict(sys.modules,
                             {"soundfile": fake_soundfile(rate=16000)}):
            with mock.patch("urllib.request.urlopen", side_effect=capture):
                synth.synthesize_wav("你好", self.wav_path)
                synth.synthesize_wav("你好", self.wav_path)

        self.assertEqual(len(sent), 2)
        first, second = sent
        first_id = header_of(first, "X-Api-Request-Id")
        self.assertTrue(first_id, "请求头少了 X-Api-Request-Id，v3 认这个才认请求")
        self.assertNotEqual(first_id, header_of(second, "X-Api-Request-Id"))

    def test_text_is_sent_in_the_body_not_the_url(self):
        synth = make_doubao()
        seen = []

        def capture(request, timeout=None):
            seen.append(request)
            return ok_response()

        with mock.patch.dict(sys.modules,
                             {"soundfile": fake_soundfile(rate=16000)}):
            with mock.patch("urllib.request.urlopen", side_effect=capture):
                synth.synthesize_wav("我睡不着觉", self.wav_path)

        request = seen[0]
        self.assertNotIn("我睡不着觉", request.full_url)
        self.assertIn("我睡不着觉",
                      json.loads(request.data.decode("utf-8"))
                      ["req_params"]["text"])

    def test_empty_text_is_a_no_op(self):
        synth = make_doubao()
        with mock.patch("urllib.request.urlopen") as opened:
            self.assertFalse(synth.synthesize_wav("   ", self.wav_path))
        opened.assert_not_called()

    def test_unavailable_synthesizer_never_calls_the_network(self):
        synth = make_doubao()
        synth._usable = False
        with mock.patch("urllib.request.urlopen") as opened:
            self.assertFalse(synth.synthesize_wav("你好", self.wav_path))
        opened.assert_not_called()

    def test_close_is_idempotent(self):
        synth = make_doubao()
        synth.close()
        synth.close()


class TestFallbackSynthesizer(unittest.TestCase):
    """在线引擎这一句失败时，本地引擎要把这句话接过来。

    这是"网络抖一下 = 机器人突然不吭声"的正面回答：兜底不是启动时挑一次
    就完事，而是**每一句**都还有一次机会。
    """

    def setUp(self):
        self.tempdir = tempfile.mkdtemp(prefix="b_fallback_test_")
        self.addCleanup(shutil.rmtree, self.tempdir, ignore_errors=True)
        self.wav_path = os.path.join(self.tempdir, "out.wav")
        self.spoken = []            # 备用引擎真的合成了哪几句
        self.built = 0              # 工厂被调用了几次

    def make_primary(self, result=True, raises=None):
        class Primary:
            name = "doubao"
            voice = "zh_female_vv_uranus_bigtts"
            speed = 0.9
            available = True
            requires_network = True

            def synthesize_wav(inner, text, path):
                if raises is not None:
                    raise raises
                return result

            def close(inner):
                pass

        return Primary()

    def make_factory(self, result=True, raises=None):
        recorded = self.spoken
        counter = self

        def factory():
            counter.built += 1
            if raises is not None:
                raise raises

            class Backup:
                name = "sapi"
                available = True

                def synthesize_wav(inner, text, path):
                    recorded.append(text)
                    return result

                def close(inner):
                    pass

            return Backup()

        return factory

    def test_primary_success_never_touches_the_backup(self):
        """主引擎好着的时候不能白起一个本地引擎 ——
        SapiSynthesizer 一建就拉一个 PowerShell 进程，那不是免费的。"""
        engine = tts.FallbackSynthesizer(self.make_primary(),
                                         self.make_factory())
        self.assertTrue(engine.synthesize_wav("你好", self.wav_path))
        self.assertEqual(self.built, 0, "主引擎成功时不该创建备用引擎")
        self.assertEqual(self.spoken, [])
        self.assertFalse(engine.used_fallback)

    def test_primary_failure_falls_back_to_the_local_engine(self):
        engine = tts.FallbackSynthesizer(self.make_primary(result=False),
                                         self.make_factory())
        with self.assertLogs("core.voice.tts", level="WARNING") as caught:
            ok = engine.synthesize_wav("你好", self.wav_path)

        self.assertTrue(ok, "主引擎失败但本地兜住了，对外应当算发声成功")
        self.assertEqual(self.spoken, ["你好"])
        self.assertIn("兜底", "\n".join(caught.output))

    def test_fallback_is_flagged_so_it_is_not_cached(self):
        """兜底产出的是**另一种音色**。缓存键记的是主引擎，
        把它存进去的话，等网络恢复后这句话就永远是错的音色了。"""
        engine = tts.FallbackSynthesizer(self.make_primary(result=False),
                                         self.make_factory())
        with self.assertLogs("core.voice.tts", level="WARNING"):
            engine.synthesize_wav("你好", self.wav_path)
        self.assertTrue(engine.used_fallback)

    def test_primary_exception_also_falls_back(self):
        """引擎违约抛异常时也要兜 —— 契约说永不抛，但兜底不能依赖它守约。"""
        engine = tts.FallbackSynthesizer(
            self.make_primary(raises=OSError("断网了")), self.make_factory())
        with self.assertLogs("core.voice.tts", level="ERROR"):
            ok = engine.synthesize_wav("你好", self.wav_path)
        self.assertTrue(ok)
        self.assertEqual(self.spoken, ["你好"])

    def test_the_backup_is_built_only_once(self):
        """失败会反复发生，备用引擎不该每句都重建一次。"""
        engine = tts.FallbackSynthesizer(self.make_primary(result=False),
                                         self.make_factory())
        with self.assertLogs("core.voice.tts", level="WARNING"):
            engine.synthesize_wav("第一句", self.wav_path)
            engine.synthesize_wav("第二句", self.wav_path)
        self.assertEqual(self.built, 1)
        self.assertEqual(self.spoken, ["第一句", "第二句"])

    def test_both_engines_failing_reports_failure_quietly(self):
        """两个都不行时返回 False（由 Speaker 记一条"只进了文字通道"），
        **不要抛异常** —— 播报线程不该被 TTS 搞崩。"""
        engine = tts.FallbackSynthesizer(self.make_primary(result=False),
                                         self.make_factory(result=False))
        with self.assertLogs("core.voice.tts", level="WARNING"):
            ok = engine.synthesize_wav("你好", self.wav_path)
        self.assertFalse(ok)
        self.assertFalse(engine.used_fallback, "兜底也没出声，不能记成兜底成功")

    def test_unbuildable_backup_is_not_reported_as_a_success(self):
        """备用引擎建不出来（比如非 Windows）时，不能假装兜住了。

        这里以前会把 NullSynthesizer 当成"可用"，日志里出现一句
        "改用本地引擎 'null' 兜底" —— 看着像成功了，其实一个字都没发出去。
        """
        engine = tts.FallbackSynthesizer(
            self.make_primary(result=False),
            self.make_factory(raises=OSError("没有 PowerShell")))
        with self.assertLogs("core.voice.tts", level="WARNING"):
            ok = engine.synthesize_wav("你好", self.wav_path)
        self.assertFalse(ok)
        self.assertFalse(engine.used_fallback)

    def test_name_and_speed_are_the_primary_engines(self):
        """缓存键按 `name|voice|speed` 算，对外必须报**主引擎**的身份 ——
        不然预热时算出来的键和播放时查的键会对不上。"""
        engine = tts.FallbackSynthesizer(self.make_primary(),
                                         self.make_factory())
        self.assertEqual(engine.name, "doubao")
        self.assertEqual(engine.voice, "zh_female_vv_uranus_bigtts")
        self.assertAlmostEqual(engine.speed, 0.9)
        self.assertTrue(engine.requires_network)

    def test_close_closes_both_engines(self):
        engine = tts.FallbackSynthesizer(self.make_primary(result=False),
                                         self.make_factory())
        with self.assertLogs("core.voice.tts", level="WARNING"):
            engine.synthesize_wav("你好", self.wav_path)
        engine.close()                # 抛异常即失败


class TestProbeSynthesizer(unittest.TestCase):
    """选型时的试合成。

    `available` 只回答"库装好没有"，答不了"能不能出声"。
    本机的实际故障就是这两者脱节：edge-tts 装好了，TCP 也连得上，
    但 TLS 被重置，于是每句都在运行期失败 —— 而候选链后面那个
    离线可用的 SAPI 永远轮不到。
    """

    def test_returns_true_when_the_engine_produces_audio(self):
        write_wav_capable = FakeSynthesizer()
        self.assertTrue(tts.probe_synthesizer(write_wav_capable))
        self.assertEqual(len(write_wav_capable.calls), 1)

    def test_returns_false_when_the_engine_fails(self):
        self.assertFalse(tts.probe_synthesizer(FakeSynthesizer(succeed=False)))

    def test_never_raises_even_if_the_engine_breaks_its_contract(self):
        """synthesize_wav 的契约是"永不抛异常"，但探测不能因为
        某个引擎违约就把整个选型流程带崩。"""
        class Broken:
            name = "broken"

            def synthesize_wav(self, text, path):
                raise RuntimeError("我不守契约")

        self.assertFalse(tts.probe_synthesizer(Broken()))

    def test_uses_a_short_text_and_the_callers_own_writer(self):
        """探测花的是一次真实往返，所以句子要短 —— 钉住这一点，
        免得哪天有人把它改成一段长文案。"""
        synth = FakeSynthesizer()
        tts.probe_synthesizer(synth)
        self.assertEqual(synth.calls[0][0], tts.PROBE_TEXT)
        self.assertLessEqual(len(tts.PROBE_TEXT), 4)

    def test_cleans_up_its_scratch_directory(self):
        synth = FakeSynthesizer()
        tts.probe_synthesizer(synth)
        scratch = os.path.dirname(synth.calls[0][1])
        self.assertFalse(os.path.isdir(scratch), "探测的临时目录没清掉")


class TestDoubaoSynthSelection(unittest.TestCase):
    """`--tts doubao` 忘了填凭证 ≠ 机器人一声不吭。

    ⚠️ 这些用例一律带 ``probe=False``：默认的试合成会真的联网，
    而本文件的第一条约定是"不碰真实设备、不联网"。
    试合成本身的行为由 TestProbeSynthesizer 用假引擎覆盖。
    """

    def test_named_engine_falls_back_when_credentials_are_missing(self):
        engine = tts.build_synthesizer("doubao", probe=False)
        self.assertIsNot(engine, None)
        # 关键断言：**不是**哑巴。退到哪个引擎取决于本机装了什么，
        # 但无论如何都要能发声。
        self.assertNotIsInstance(engine, NullSynthesizer)
        engine.close()

    def test_volcengine_alias_is_the_same_engine(self):
        self.assertIs(tts.ENGINES["volcengine"], tts.ENGINES["doubao"])

    def test_fallback_logs_which_engine_was_used_instead(self):
        """悄悄换个音色会让人以为 --tts 生效了，必须留下一条日志。"""
        with self.assertLogs("core.voice.tts", level="WARNING") as caught:
            engine = tts.build_synthesizer("doubao", probe=False)
        engine.close()
        self.assertIn("回退", "\n".join(caught.output))

    def test_engine_that_fails_its_probe_is_skipped(self):
        """**这条是本机的现场**：联网引擎"看起来可用"但试合成失败时，
        必须继续往后找，而不是把它交出去、让之后每一句都失败。"""
        class Alive:
            name = "alive"
            available = True
            requires_network = True

            def __init__(self, **kwargs):
                pass

            def synthesize_wav(self, text, path):
                return True

            def close(self):
                pass

        class LooksUsableButIsNot:
            """联网引擎的典型故障态：装好了、`available` 为真、
            但请求根本发不出去。"""

            name = "dead"
            available = True
            requires_network = True
            probed = 0

            def __init__(self, **kwargs):
                pass

            def synthesize_wav(self, text, path):
                LooksUsableButIsNot.probed += 1
                return False

            def close(self):
                pass

        def dead_factory(**kwargs):
            return LooksUsableButIsNot()

        def alive_factory(**kwargs):
            return Alive()

        engines = {"sapi": alive_factory, "doubao": dead_factory,
                   "edge_tts": alive_factory}
        with mock.patch.dict(tts.ENGINES, engines, clear=False):
            with self.assertLogs("core.voice.tts", level="WARNING") as caught:
                engine = tts.build_synthesizer("auto")

        # 在线引擎外面还会再套一层**运行期**兜底（FallbackSynthesizer），
        # 所以交出来的不是 Alive 本身 —— 看它包着谁，而不是直接 isinstance。
        self.assertIsInstance(getattr(engine, "primary", engine), Alive,
                              "试合成失败的引擎不该被选中")
        self.assertIn("试合成失败", "\n".join(caught.output))

    def test_offline_engines_are_never_probed(self):
        """SAPI 不联网，没有 requires_network 属性 —— 不该为它花一次合成。"""
        class Offline:
            name = "sapi"
            available = True

            def __init__(self, **kwargs):
                pass

            def synthesize_wav(self, text, path):
                raise AssertionError("离线引擎不该被试合成")

            def close(self):
                pass

        with mock.patch.dict(tts.ENGINES, {"sapi": Offline}, clear=False):
            engine = tts.build_synthesizer("sapi")

        self.assertIsInstance(engine, Offline)

    def test_auto_prefers_doubao_when_it_is_available(self):
        """auto 的候选顺序就是"默认用豆包"的落地方式，钉住它。"""
        made = []

        class FakeAvailable:
            available = True

            def __init__(self, **kwargs):
                made.append("fake")
                self.kwargs = kwargs

            def close(self):
                pass

        with mock.patch.dict(tts.ENGINES, {"doubao": FakeAvailable},
                             clear=False):
            engine = tts.build_synthesizer("auto")

        self.assertIsInstance(engine, FakeAvailable)
        self.assertEqual(made, ["fake"], "auto 应当第一个就选中豆包")

    def test_auto_falls_to_null_when_every_engine_is_unavailable(self):
        """全挂了必须是 NullSynthesizer —— 只说话不发声，而不是崩掉。

        这条在 STT 侧已有对应测试，TTS 侧一直缺着。
        """
        class FakeDead:
            available = False

            def __init__(self, **kwargs):
                pass

            def close(self):
                pass

        dead = {name: FakeDead for name in ("sapi", "doubao", "edge_tts")}
        with mock.patch.dict(tts.ENGINES, dead, clear=False):
            with self.assertLogs("core.voice.tts", level="WARNING"):
                engine = tts.build_synthesizer("auto")

        self.assertIsInstance(engine, NullSynthesizer)


# --------------------------------------------------------------------------
# 预合成缓存
# --------------------------------------------------------------------------

class TestSpeechCache(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp(prefix="b_cache_test_")
        self.addCleanup(shutil.rmtree, self.tempdir, ignore_errors=True)
        self.cache = speech_cache.SpeechCache(self.tempdir, "edge_tts|voice|1.0")

    def store(self, text: str, cache=None):
        """造一份"已合成"的 WAV 并存进缓存。"""
        path = os.path.join(self.tempdir, "src_%d.wav" % abs(hash(text)))
        write_wav(path)
        (cache or self.cache).store(text, path)
        return path

    def test_miss_then_hit(self):
        self.assertIsNone(self.cache.lookup("你好"))
        self.store("你好")
        self.assertEqual(self.cache.lookup("你好"),
                         self.cache.path_for("你好"))

    def test_hit_returns_a_readable_wav(self):
        self.store("你好")
        with wave.open(self.cache.lookup("你好"), "rb") as handle:
            self.assertGreater(handle.getnframes(), 0)

    def test_empty_text_never_hits(self):
        """空串会被哈希成一个合法文件名，于是每次空播报都命中同一个文件 ——
        一个很安静的错。这里明确挡住。"""
        self.store("")
        self.assertIsNone(self.cache.lookup(""))
        self.assertIsNone(self.cache.lookup("   "))

    def test_key_changes_with_engine_voice_and_speed(self):
        """换了引擎/音色/语速却不失效的话，用户会听到上一个音色的缓存，
        然后怀疑"我明明改了配置怎么没变"。"""
        other = speech_cache.SpeechCache(self.tempdir, "edge_tts|voice|1.5")
        third = speech_cache.SpeechCache(self.tempdir, "doubao|voice|1.0")

        self.store("你好")
        self.assertIsNotNone(self.cache.lookup("你好"))
        self.assertIsNone(other.lookup("你好"), "语速变了却没失效")
        self.assertIsNone(third.lookup("你好"), "引擎变了却没失效")

    def test_store_is_atomic_no_temp_files_left_behind(self):
        self.store("你好")
        leftovers = [n for n in os.listdir(self.tempdir)
                     if n.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_store_of_a_missing_source_is_a_no_op(self):
        self.assertIsNone(self.cache.store("你好", os.path.join(self.tempdir,
                                                                "没有这个文件.wav")))
        self.assertIsNone(self.cache.lookup("你好"))

    def test_prune_removes_stale_entries_and_keeps_current_ones(self):
        self.store("留下")
        self.store("删掉")
        removed = self.cache.prune(["留下"])
        self.assertEqual(removed, 1)
        self.assertIsNotNone(self.cache.lookup("留下"))
        self.assertIsNone(self.cache.lookup("删掉"))

    def test_prune_leaves_foreign_files_alone(self):
        """目录里万一有别人的东西（目录填错、有别的用途），绝不能误删。

        这个函数的动作是"删除"，而目录名来自配置 —— 卡得死一点，
        才有可能在配置填错时不造成不可恢复的损失。
        """
        strangers = ["笔记.txt", "别人的录音.wav", "backup.wav.bak"]
        for name in strangers:
            with open(os.path.join(self.tempdir, name), "w",
                      encoding="utf-8") as handle:
                handle.write("不要删我")
        self.store("你好")

        self.cache.prune(["你好"])
        for name in strangers:
            self.assertTrue(os.path.exists(os.path.join(self.tempdir, name)),
                            f"{name} 被误删了")

    def test_prune_still_cleans_up_entries_from_another_engine(self):
        """上面那条不能矫枉过正：换引擎产生的旧缓存仍然要清掉 ——
        它们的名字是同一套格式。"""
        other = speech_cache.SpeechCache(self.tempdir, "doubao|voice|1.0")
        self.store("你好", cache=other)

        self.cache.prune(["无关的文本"])
        self.assertIsNone(other.lookup("你好"))

    def test_unusable_directory_degrades_instead_of_raising(self):
        """缓存是优化，不该成为故障点：目录建不出来就永远不命中。"""
        # 拿一个**文件**当目录用，makedirs 必然失败。
        blocker = os.path.join(self.tempdir, "blocker")
        with open(blocker, "w", encoding="utf-8") as handle:
            handle.write("x")

        with self.assertLogs("core.voice.speech_cache", level="WARNING"):
            cache = speech_cache.SpeechCache(
                os.path.join(blocker, "sub"), "k")

        self.assertIsNone(cache.lookup("你好"))
        self.assertIsNone(cache.store("你好", blocker))
        self.assertEqual(cache.prune(["你好"]), 0)

    def test_directory_is_created_on_demand(self):
        target = os.path.join(self.tempdir, "新建", "深一层")
        speech_cache.SpeechCache(target, "k")
        self.assertTrue(os.path.isdir(target))


class TestSpeakerUsesCache(unittest.TestCase):
    """Speaker 命中缓存时必须**跳过合成** —— 那才是省下来的三秒。"""

    def setUp(self):
        self.tempdir = tempfile.mkdtemp(prefix="b_speaker_cache_")
        self.addCleanup(shutil.rmtree, self.tempdir, ignore_errors=True)
        self.cache = speech_cache.SpeechCache(self.tempdir, "fake||")
        self.synth = FakeSynthesizer()
        self.player = FakePlayer()

    def test_second_say_of_the_same_text_does_not_synthesize(self):
        speaker = Speaker(synthesizer=self.synth, player=self.player,
                          cache=self.cache)

        self.assertTrue(speaker.say("您好呀"))
        self.assertEqual(len(self.synth.calls), 1)

        self.assertTrue(speaker.say("您好呀"))
        self.assertEqual(len(self.synth.calls), 1,
                         "第二次应当命中缓存，不该再合成一遍")
        self.assertEqual(len(self.player.played), 2)

    def test_cache_hit_is_logged(self):
        speaker = Speaker(synthesizer=self.synth, player=self.player,
                          cache=self.cache)
        speaker.say("您好呀")
        with self.assertLogs("core.voice.loop", level="INFO") as caught:
            speaker.say("您好呀")
        self.assertIn("命中预合成缓存", "\n".join(caught.output))

    def test_a_cache_hit_still_plays_a_real_wav(self):
        """命中时播的必须是缓存目录里的真文件 —— 不是一张已经删掉的临时文件。

        第一次播的临时文件说完就被清掉了，所以只查最后一条：
        那正是走缓存的那次。
        """
        speaker = Speaker(synthesizer=self.synth, player=self.player,
                          cache=self.cache)
        speaker.say("您好呀")
        speaker.say("您好呀")

        self.assertEqual(self.player.played[-1], self.cache.path_for("您好呀"))
        self.assertTrue(os.path.isfile(self.player.played[-1]))

    def test_cached_file_survives_the_temporary_file_cleanup(self):
        """`_speak_locked` 的 finally 会删临时 WAV。
        缓存文件在另一个目录、且下次还要用 —— 被顺手删掉的话，
        缓存会永远不命中，而日志里什么都看不出来。"""
        speaker = Speaker(synthesizer=self.synth, player=self.player,
                          cache=self.cache)
        speaker.say("您好呀")
        cached = self.cache.path_for("您好呀")
        self.assertTrue(os.path.isfile(cached))

        speaker.say("您好呀")
        self.assertTrue(os.path.isfile(cached), "缓存文件被临时文件清理误删了")

    def test_no_cache_means_every_say_synthesizes(self):
        speaker = Speaker(synthesizer=self.synth, player=self.player)
        speaker.say("您好呀")
        speaker.say("您好呀")
        self.assertEqual(len(self.synth.calls), 2)

    def test_failed_synthesis_is_not_cached(self):
        """合成失败时临时文件根本没写出来，不能被当成有效缓存存进去 ——
        否则以后每次都会"命中"一个空文件，机器人永久哑掉。"""
        speaker = Speaker(synthesizer=FakeSynthesizer(succeed=False),
                          player=self.player, cache=self.cache)
        with self.assertLogs("core.voice.loop", level="WARNING"):
            self.assertFalse(speaker.say("您好呀"))
        self.assertIsNone(self.cache.lookup("您好呀"))


class TestSpeakerCacheAllow(unittest.TestCase):
    """``cache_allow``：只有会被反复说起的固定句才值得写进缓存。

    背景：``SpeechCache.prune()`` 的保留集就是 ``dialogue.static_replies()``，
    所以存了别的文本只是等着被 prune 删掉 —— 涨磁盘、命中率 0。接上大模型
    之后每条回复都是模型现编的、必然唯一，这个问题从"偶尔几个孤儿文件"
    变成"每句话都留一份"。
    """

    def setUp(self):
        self.tempdir = tempfile.mkdtemp(prefix="b_speaker_allow_")
        self.addCleanup(shutil.rmtree, self.tempdir, ignore_errors=True)
        self.cache = speech_cache.SpeechCache(self.tempdir, "fake||")
        self.synth = FakeSynthesizer()
        self.player = FakePlayer()

    def make(self, allow):
        return Speaker(synthesizer=self.synth, player=self.player,
                       cache=self.cache, cache_allow=allow)

    def test_dynamic_reply_is_not_cached(self):
        speaker = self.make(allow={"您好呀"})
        self.assertTrue(speaker.say("一句一次性的动态回复"))
        self.assertIsNone(self.cache.lookup("一句一次性的动态回复"))

    def test_allowed_text_is_cached(self):
        speaker = self.make(allow={"您好呀"})
        speaker.say("您好呀")
        self.assertIsNotNone(self.cache.lookup("您好呀"))

    def test_not_caching_does_not_stop_the_audio(self):
        """不缓存 ≠ 不说话。这两件事必须分开 —— 合并了就变成静音。"""
        speaker = self.make(allow={"您好呀"})
        self.assertTrue(speaker.say("一句一次性的动态回复"))
        self.assertEqual(len(self.player.played), 1)

    def test_dynamic_reply_synthesizes_every_time(self):
        """不缓存的直接后果：每次都要重合成。这是**已知代价**，不是 bug ——
        反正那些句子本来也永远不会被说起第二次，缓存它们才是纯浪费。"""
        speaker = self.make(allow={"您好呀"})
        speaker.say("一句一次性的动态回复")
        speaker.say("一句一次性的动态回复")
        self.assertEqual(len(self.synth.calls), 2)

    def test_whitespace_is_ignored_on_both_sides(self):
        """``say`` 会先 strip 再查缓存，白名单也必须按 strip 后的文本比 ——
        否则 ``static_replies()`` 里哪句末尾多个空格就会静默失效。"""
        speaker = self.make(allow={"  您好呀  "})
        speaker.say("您好呀")
        self.assertIsNotNone(self.cache.lookup("您好呀"))

    def test_static_replies_are_all_cacheable(self):
        """``main.py`` 传的就是 ``static_replies()``，这里按那个姿势走一遍。

        用的是真文件（不是 fixture）：这个白名单一旦和真实的
        ``say()`` 文本对不上，缓存就从"预热过的句子秒回"退化成"每次都要
        重新合成三秒"，而日志里只会显示一切正常。
        """
        from core.dialogue import static_replies
        texts = static_replies()
        self.assertTrue(texts, "static_replies() 不该是空的")
        speaker = self.make(allow=texts)
        for text in texts[:5]:
            with self.subTest(text=text[:12]):
                self.assertTrue(speaker.say(text))
                self.assertIsNotNone(
                    self.cache.lookup(text.strip()),
                    "static_replies() 里的句子没被缓存：%r" % text)

    def test_none_means_cache_everything(self):
        """不传白名单 = 保持原来的行为（其他构造点和测试都靠这个默认值）。"""
        speaker = self.make(allow=None)
        speaker.say("一句一次性的动态回复")
        self.assertIsNotNone(self.cache.lookup("一句一次性的动态回复"))


# --------------------------------------------------------------------------
# 语音输出里不该有儿化音
# --------------------------------------------------------------------------

class TestNoErhuaInSpokenText(unittest.TestCase):
    """机器人说的话里不能带儿化音（用户明确要求去掉）。

    只查**不带占位符**的固定回复：带 `{topic}` 的句子要拼进用户自己的话，
    文本在运行时才定下来，这里看不到。
    """

    #: 儿化音最常见的几种写法。不用笼统的「儿」字，免得哪天出现
    #: 「女儿」「儿子」这类正常词被误判。
    ERHUA = ("这儿", "那儿", "哪儿", "会儿", "事儿", "一块儿", "活儿")

    def test_static_replies_are_not_empty(self):
        """兜底：真去掉了所有回复的话，下面的循环会空转通过。"""
        self.assertGreater(len(static_replies()), 20)

    def test_no_static_reply_contains_erhua(self):
        offenders = [
            text for text in static_replies()
            if any(word in text for word in self.ERHUA)
        ]
        self.assertEqual(
            offenders, [],
            "这些固定回复里还有儿化音，改掉它们：\n" + "\n".join(offenders),
        )


class TestSingleEngineImportGuard(unittest.TestCase):
    """守住"B 运行期零第三方依赖"这条硬约束。

    语音层必须能 import 而**不加载** sounddevice / vosk / edge_tts ——
    它们的 import 写在函数体内部。这条测试用"模块顶层不出现这些名字"
    来近似验证：真的 import 了的话，sys.modules 里就会有它们。
    """

    THIRD_PARTY = ("sounddevice", "vosk", "edge_tts", "speech_recognition",
                   "dashscope", "soundfile", "numpy")

    def test_importing_voice_package_loads_no_third_party(self):
        import subprocess

        # 在干净的子进程里 import，避免本测试进程（已经加载过别的东西）干扰
        code = (
            "import sys;"
            "sys.path.insert(0, r'%s');"
            "import core.voice, core.voice.stt, core.voice.tts,"
            " core.voice.vad, core.voice.base, core.voice.loop,"
            " core.voice.speech_cache;"
            "bad=[m for m in %r if m in sys.modules];"
            "print(','.join(bad))"
        ) % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
             self.THIRD_PARTY)

        result = subprocess.run([sys.executable, "-c", code],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        loaded = result.stdout.strip()
        self.assertEqual(
            loaded, "",
            f"import 语音层时加载了第三方库：{loaded}。"
            "这些 import 必须写在函数体内部，否则 B 就不再是零依赖的了。",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
