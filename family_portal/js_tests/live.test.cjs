/* live.js 轮询页**纯逻辑**的 Node 单元测试（node 内置 test runner，零依赖）。
 *
 * 运行：node family_portal/js_tests/live.test.cjs
 * （由 family_portal/tests_js.py 以 pytest 包装调用）
 *
 * live.js 的 DOM/fetch/setInterval 装配在 Node 下不执行（文件内
 * `typeof window === "undefined"` 直接返回），这里测的是**判定规则**：
 * 离线判定（服务端时间优先）、信号徽标分级、状态文案拼接。
 * 页面真实渲染与轮询节拍由浏览器验收覆盖。
 */
"use strict";

const test = require("node:test");
const assert = require("node:assert");
const live = require("../../static/family/js/live.js");

/* ===================================================================
 * isOffline —— 二期核心：服务端时间差判定
 * =================================================================== */

test("isOffline: 没有样本 → 离线（'暂无数据'与'离线'由上层区分）", () => {
  assert.equal(live.isOffline(null, 30, 1000), true);
  assert.equal(live.isOffline(undefined, 30, 1000), true);
});

test("isOffline: server_ts 优先，与浏览器时钟无关", () => {
  /* 服务端时间差 40s > 30s 阈值 → 离线，即使浏览器时钟"看起来"很新 */
  assert.equal(live.isOffline(1000, 30, 1040), true);
  assert.equal(live.isOffline(1000, 30, 1020), false);
});

test("isOffline: 恰好等于阈值不算离线（严格大于）", () => {
  assert.equal(live.isOffline(1000, 30, 1030), false);
  assert.equal(live.isOffline(1000, 30, 1030.5), true);
});

test("isOffline: server_ts 缺失时回退浏览器时钟（向后兼容旧接口）", () => {
  const realNow = Date.now;
  try {
    Date.now = () => 1_000_000_000_000;            /* 固定"现在" */
    assert.equal(live.isOffline(1_000_000_000 - 60, 30, undefined), true);
    assert.equal(live.isOffline(1_000_000_000 - 10, 30, undefined), false);
  } finally {
    Date.now = realNow;
  }
});

/* ===================================================================
 * 信号徽标 —— hrFlag / rrFlag
 * =================================================================== */

test("hrFlag: 无测量值 → 灰'未检出'", () => {
  assert.deepEqual(live.hrFlag(null, 0.9), { level: "none", text: "未检出" });
  assert.deepEqual(live.hrFlag(undefined, null), { level: "none", text: "未检出" });
});

test("hrFlag: SQI < 0.5 → 橙'信号减弱'；正常 → 绿'波形稳定'", () => {
  assert.deepEqual(live.hrFlag(72, 0.3), { level: "mid", text: "信号减弱" });
  assert.deepEqual(live.hrFlag(72, 0.8), { level: "good", text: "波形稳定" });
});

test("hrFlag: SQI 恰为 0.5 不算减弱（严格小于）；SQI 缺失也算稳定", () => {
  assert.deepEqual(live.hrFlag(72, 0.5), { level: "good", text: "波形稳定" });
  assert.deepEqual(live.hrFlag(72, undefined), { level: "good", text: "波形稳定" });
  assert.deepEqual(live.hrFlag(72, null), { level: "good", text: "波形稳定" });
});

test("rrFlag: 无值灰'未检出'，有值绿'节律均匀'（呼吸无 SQI 分级）", () => {
  assert.deepEqual(live.rrFlag(null), { level: "none", text: "未检出" });
  assert.deepEqual(live.rrFlag(16.0), { level: "good", text: "节律均匀" });
});

/* ===================================================================
 * 文案拼接 —— updateLabel / todayLine / fmtTime
 * =================================================================== */

test("updateLabel: 在线显示'更新于 服务端时间'；离线显示'最后数据'", () => {
  /* ts=0 是 1970-01-01T00:00:00 本地零点，小时字段随时区，只断言前缀 */
  assert.ok(live.updateLabel(false, null, 0).startsWith("更新于 "));
  assert.ok(live.updateLabel(true, { ts: 0 }, null).startsWith("最后数据："));
});

test("updateLabel: 离线且连 vitals 都没有 → '最后数据：—'（不显示空时间）", () => {
  assert.equal(live.updateLabel(true, null, null), "最后数据：—");
});

test("updateLabel: 时间戳取整秒后格式为 HH:MM:SS", () => {
  assert.match(live.updateLabel(true, { ts: 1_000_000 }, null), /^\u6700\u540e\u6570\u636e\uff1a\d{2}:\d{2}:\d{2}$/);
});

test("todayLine: hr_avg/rr_avg 为 null 显示'未检出'，不出现 'null 次/分'", () => {
  const line = live.todayLine({
    sample_count: 5, hr_avg: null, rr_avg: null, alert_count: 0,
    state_distribution: [],
  });
  assert.ok(line.includes("今日样本 5 条"));
  assert.ok(line.includes("平均心率 未检出"));
  assert.ok(line.includes("平均呼吸 未检出"));
  assert.ok(!line.includes("null"));
  assert.ok(line.includes("今日还没有状态记录"));
});

test("todayLine: 状态分布拼接占比与分钟数", () => {
  const line = live.todayLine({
    sample_count: 100, hr_avg: 72, rr_avg: 16, alert_count: 1,
    state_distribution: [{ state: "normal", ratio: 0.6, seconds: 3600 },
                         { state: "sad", ratio: 0.4, seconds: 90 }],
  });
  assert.ok(line.includes("状态分布：normal 60%（约 60 分钟），sad 40%（约 2 分钟）"));
  assert.ok(line.includes("今日预警 1 次"));
});

test("todayLine: today 为 null 返回空串（页面不动今日栏）", () => {
  assert.equal(live.todayLine(null), "");
});
