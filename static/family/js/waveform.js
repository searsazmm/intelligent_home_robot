/* 实时体征波形（轻量 canvas 动画，无第三方依赖）。

   数字与波形"对得上"的实现：波动频率由实时体征值驱动 ——
   - 心率（ecg 模式）：简化心电形状，每 60/hr 秒一个心搏脉冲，
     hr=84 → 相位每秒推进 1.4 个搏动周期，波形与数字同源；
   - 呼吸（resp 模式）：正弦波，周期 = 60/rr 秒，rr=16 → 3.75s 一周期。
   setRate(null) = 未检出/离线：波形停在平线（灰），不假装还在测。

   绘制方式：滚动窗口（保留最近 N 个采样点，逐帧重绘）——
   canvas 只有几百像素宽，全量重绘开销可忽略，换取实现简单可靠。
   感知上与监护仪的"扫掠"效果等效（新波从右推入、旧波左移）。
   */
(function () {
  "use strict";

  var DPR = window.devicePixelRatio || 1;

  /* 简化心电周期形状（0..1 相位 → -0.2..1.0 振幅）：
     P 小丘 → Q 微降 → R 尖峰 → S 回落 → T 圆丘 → 基线。
     用窄高斯叠加合成，够"像心电"且无任何外部资源。 */
  function gauss(x, mu, sigma) {
    var d = (x - mu) / sigma;
    return Math.exp(-d * d / 2);
  }
  function ecgShape(p) {
    return 0.12 * gauss(p, 0.16, 0.030)   /* P 波 */
         - 0.14 * gauss(p, 0.265, 0.012)  /* Q 波 */
         + 1.00 * gauss(p, 0.30, 0.016)   /* R 尖峰 */
         - 0.22 * gauss(p, 0.335, 0.014)  /* S 波 */
         + 0.26 * gauss(p, 0.52, 0.055);  /* T 波 */
  }

  function Waveform(canvas, opts) {
    this.canvas = canvas;
    this.type = opts.type;                 /* "ecg" | "resp" */
    this.color = opts.color || "#3a9d6e";
    this.rate = null;                      /* 次/分钟；null = 未检出 */
    this.phase = 0;                        /* 0..1 周期相位 */
    this.samples = [];                     /* 滚动窗口内的归一化值 */
    this._last = performance.now();
    this._resize();
    var self = this;
    window.addEventListener("resize", function () { self._resize(); });
    requestAnimationFrame(function loop(now) {
      self._tick(now);
      requestAnimationFrame(loop);
    });
  }

  Waveform.prototype.setRate = function (rate) {
    this.rate = (typeof rate === "number" && rate > 0) ? rate : null;
  };

  Waveform.prototype._resize = function () {
    var w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.canvas.width = w * DPR;
    this.canvas.height = h * DPR;
    this.w = w; this.h = h;
  };

  Waveform.prototype._tick = function (now) {
    /* 尺寸自检：构造时布局可能未完成（clientWidth=0），且窗口缩放
       也要跟上 —— 每帧对一下，不一致就重设。这是无报错空白画的根源。 */
    var cw = this.canvas.clientWidth, ch = this.canvas.clientHeight;
    if ((cw && cw !== this.w) || (ch && ch !== this.h)) this._resize();
    if (!this.w || !this.h) return;        /* 布局仍不可用：本帧跳过 */

    var dt = Math.min((now - this._last) / 1000, 0.1);   /* 切后台回来防跳变 */
    this._last = now;

    if (this.rate) {
      this.phase = (this.phase + dt * this.rate / 60) % 1;
      var v = (this.type === "ecg") ? ecgShape(this.phase) : Math.sin(this.phase * 2 * Math.PI);
      this.samples.push(v);
    } else {
      this.samples.push(0);              /* 未检出：平线 */
    }
    /* 窗口长度 = 画布宽的一半像素：1px 一点，滚动速度恒定 */
    var maxPoints = Math.max(40, Math.floor(this.w / 2));
    while (this.samples.length > maxPoints) this.samples.splice(0, this.samples.length - maxPoints);

    this._draw();
  };

  Waveform.prototype._draw = function () {
    var ctx = this.canvas.getContext("2d");
    var w = this.w, h = this.h, mid = h * 0.62;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, w, h);

    /* 基线 */
    ctx.strokeStyle = "rgba(58,157,110,.18)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, mid); ctx.lineTo(w, mid);
    ctx.stroke();

    if (!this.rate) {
      ctx.strokeStyle = "rgba(140,150,140,.45)";       /* 灰：未测量 */
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(0, mid); ctx.lineTo(w, mid);
      ctx.stroke();
      return;
    }

    ctx.strokeStyle = this.color;
    ctx.lineWidth = 1.8;
    ctx.lineJoin = "round";
    ctx.beginPath();
    var step = w / (this.samples.length - 1 || 1);
    var amp = this.type === "ecg" ? h * 0.42 : h * 0.30;
    for (var i = 0; i < this.samples.length; i++) {
      var x = i * step;
      var y = mid - this.samples[i] * amp;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };

  window.FamilyWaveform = Waveform;
})();
