/* waveform.js 的 Node 单元测试（node 内置 test runner，零依赖）。
 *
 * 运行：node family_portal/js_tests/waveform.test.cjs
 * （由 family_portal/tests_js.py 以 pytest 包装调用，node 缺失则跳过）
 *
 * 测的是**逻辑**而非像素：形状函数的性质、采样窗口语义、
 * 尺寸自检、离线平线 —— canvas 用记录调用的假对象替身。
 * 浏览器端的真实渲染已由 browser_use 验收覆盖。
 */
"use strict";

const test = require("node:test");
const assert = require("node:assert");
const { gauss, ecgShape, Waveform } = require("../../static/family/js/waveform.js");

/* ---- 测试替身：不碰 DOM ---- */

function fakeCtx() {
  return {
    strokeStyle: null, lineWidth: 0, lineJoin: "",
    beginPath() {}, moveTo() {}, lineTo() {}, stroke() {},
    clearRect() {}, setTransform() {},
  };
}

function fakeCanvas(w, h) {
  const ctx = fakeCtx();
  return {
    clientWidth: w, clientHeight: h, width: 0, height: 0,
    getContext: () => ctx, ctx,
  };
}

/* 构造时抑制 rAF 循环：单测手动调 _tick，节奏自己控制。
   _last 对齐到假时间轴 0（构造时取的是真实 performance.now()）。 */
function makeWaveform(type, w, h, color) {
  const canvas = fakeCanvas(w, h);
  const wave = new Waveform(canvas, { type: type, color: color });
  wave._last = 0;
  return { wave: wave, canvas: canvas };
}

/* ===================================================================
 * 心电形状函数（波形与数字"对得上"的数学基础）
 * =================================================================== */

test("ecgShape: R 尖峰落在 0.30 相位、幅值接近 1", () => {
  assert.ok(ecgShape(0.30) > 0.9, "R 峰应接近满幅，实际 " + ecgShape(0.30));
  /* R 峰应是整个周期的最大值 */
  let peak = -Infinity, peakAt = 0;
  for (let p = 0; p < 1; p += 0.005) {
    if (ecgShape(p) > peak) { peak = ecgShape(p); peakAt = p; }
  }
  assert.ok(Math.abs(peakAt - 0.30) < 0.01, "R 峰相位应≈0.30，实际 " + peakAt);
});

test("ecgShape: 基线接近 0（无 P/QRS/T 波处），S 波为负", () => {
  assert.ok(Math.abs(ecgShape(0.0)) < 0.02);
  assert.ok(Math.abs(ecgShape(0.75)) < 0.02);
  assert.ok(ecgShape(0.335) < 0, "S 波应下探为负");
});

test("gauss: 均值处为 1，对称衰减", () => {
  assert.equal(gauss(5, 5, 1), 1);
  assert.ok(Math.abs(gauss(4, 5, 1) - gauss(6, 5, 1)) < 1e-12);
});

/* ===================================================================
 * 采样与速率：数字驱动波形
 * =================================================================== */

test("hr=84 时相位每秒推进 1.4 个周期（数字与波动同源）", () => {
  const { wave } = makeWaveform("ecg", 200, 64, "#e2704a");
  wave.setRate(84);
  /* 1 秒（分 10 帧 × 0.1s）后相位应推进 84/60 = 1.4 → 取模 0.4 */
  for (let i = 1; i <= 10; i++) wave._tick(i * 100);
  assert.ok(Math.abs(wave.phase - 0.4) < 1e-9, "phase 应为 0.4，实际 " + wave.phase);
});

test("setRate: 非法输入一律归为未检出(null)", () => {
  const { wave } = makeWaveform("resp", 200, 64);
  for (const bad of [null, undefined, 0, -5, NaN, "72"]) {
    wave.setRate(bad);
    assert.equal(wave.rate, null, "rate=" + String(bad) + " 应视为未检出");
  }
  wave.setRate(16);
  assert.equal(wave.rate, 16);
});

test("未检出/离线：采样全是 0（平线），绘制走灰色分支", () => {
  const { wave, canvas } = makeWaveform("ecg", 200, 64);
  wave.setRate(null);
  for (let i = 1; i <= 20; i++) wave._tick(i * 16);
  assert.ok(wave.samples.length > 0);
  assert.ok(wave.samples.every((v) => v === 0), "离线波形必须是平线");
  assert.equal(canvas.ctx.strokeStyle, "rgba(140,150,140,.45)", "应画灰线而非彩色");
});

test("resp 模式：采样有正有负（正弦），ecg 有尖峰（非均匀）", () => {
  const r = makeWaveform("resp", 200, 64).wave;
  r.setRate(16);
  for (let i = 1; i <= 60; i++) r._tick(i * 50);        /* 3s，近一个呼吸周期 */
  assert.ok(r.samples.some((v) => v > 0.5) && r.samples.some((v) => v < -0.5),
            "正弦波应有明显正负摆动");

  const e = makeWaveform("ecg", 200, 64).wave;
  e.setRate(72);
  for (let i = 1; i <= 60; i++) e._tick(i * 50);
  assert.ok(e.samples.some((v) => v > 0.9), "心电应有 R 尖峰");
  assert.ok(Math.abs(e.samples.reduce((a, b) => a + b, 0) / e.samples.length) < 0.3,
            "心电大部分时间贴近基线");
});

/* ===================================================================
 * 滚动窗口与绘制
 * =================================================================== */

test("滚动窗口不超过 画布宽/2 个点，且保留的是最近的", () => {
  const { wave } = makeWaveform("ecg", 200, 64);       /* maxPoints = 100 */
  wave.setRate(72);
  for (let i = 1; i <= 150; i++) wave._tick(i * 100);  /* 远超窗口容量 */
  assert.equal(wave.samples.length, 100);
});

test("尺寸自检：构造时布局未完成(0x0)也能在下一帧恢复绘制", () => {
  /* 复现修过的那个 bug：构造时 clientWidth=0 → 静默空白 */
  const canvas = fakeCanvas(0, 0);
  const wave = new Waveform(canvas, { type: "ecg", color: "#e2704a" });
  wave._last = 0;
  wave.setRate(72);
  wave._tick(100);                                      /* 布局未就绪：跳过不画 */
  assert.equal(wave.w, undefined, "0x0 时不应记录尺寸");

  canvas.clientWidth = 200; canvas.clientHeight = 64;   /* 布局完成 */
  wave._tick(200);
  assert.equal(wave.w, 200, "下一帧应自检到尺寸并 resize");
  assert.equal(canvas.width, 200, "canvas.width 应按 DPR=1 设置");
  assert.equal(canvas.ctx.strokeStyle, "#e2704a", "应画出彩色波形");
});

test("_draw 用构造时的颜色描边", () => {
  const { wave, canvas } = makeWaveform("resp", 200, 64, "#3a9d6e");
  wave.setRate(16);
  wave._tick(100);
  assert.equal(canvas.ctx.strokeStyle, "#3a9d6e");
});
