/* 实时监测页：1.5 秒轮询 api/live + api/alerts。

   离线判定（前端实现，方案文档 v3 追加项 2；二期已改服务端时间）：
   用 api_live 返回的 server_ts（服务端生成响应的时刻）减最新样本 ts，
   超过阈值(30s) → 离线徽标。Django 与机器人主机同机部署，server_ts
   与样本 ts 同源 —— 判定与家属浏览器本地时钟无关，时钟不同步不再误判。
   （旧版用浏览器本地时间对比，见 git 历史；server_ts 缺失时回退旧法。）

   测试友好：判定/徽标/文案拼接都是**纯函数**，Node 下直接单测
   （family_portal/js_tests/live.test.cjs）；DOM 装配只在浏览器执行。
   */
(function (global) {
  "use strict";

  /* ================= 纯逻辑（node:test 覆盖） ================= */

  function isOffline(latestSampleTs, threshold, serverTs) {
    if (!latestSampleTs) return true;
    // 优先用服务端时间差（与浏览器时钟无关）；server_ts 缺失才回退浏览器时钟
    var now = (typeof serverTs === "number") ? serverTs : (Date.now() / 1000);
    return (now - latestSampleTs) > threshold;
  }

  /* 徽标语义：绿"波形稳定"= 有测量值且信号质量可用；橙"信号减弱"= 有值
     但 SQI 低（<0.5）；灰"未检出"= 没有测量值。 */
  function hrFlag(hr, sqi) {
    if (hr === null || hr === undefined) return { level: "none", text: "未检出" };
    if (sqi !== null && sqi !== undefined && sqi < 0.5) {
      return { level: "mid", text: "信号减弱" };
    }
    return { level: "good", text: "波形稳定" };
  }

  function rrFlag(rr) {
    if (rr === null || rr === undefined) return { level: "none", text: "未检出" };
    return { level: "good", text: "节律均匀" };
  }

  function fmtTime(ts) {
    var d = new Date(ts * 1000);
    function p(n) { return (n < 10 ? "0" : "") + n; }
    return p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
  }

  function updateLabel(offline, vitals, serverTs) {
    return offline
      ? "最后数据：" + (vitals ? fmtTime(vitals.ts) : "—")
      : "更新于 " + fmtTime(serverTs);   /* 响应生成时刻（服务端时间） */
  }

  function todayLine(today) {
    if (!today) return "";
    var dist = today.state_distribution || [];
    var parts = dist.map(function (d) {
      var minutes = Math.round(d.seconds / 60);
      return d.state + " " + Math.round(d.ratio * 100) + "%（约 " + minutes + " 分钟）";
    });
    return "今日样本 " + today.sample_count + " 条 · 平均心率 " +
      (today.hr_avg !== null && today.hr_avg !== undefined
        ? today.hr_avg + " 次/分" : "未检出") +
      " · 平均呼吸 " +
      (today.rr_avg !== null && today.rr_avg !== undefined
        ? today.rr_avg + " 次/分" : "未检出") +
      " · 今日预警 " + today.alert_count + " 次" +
      (parts.length ? "<br>状态分布：" + parts.join("，") : "<br>今日还没有状态记录");
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      isOffline: isOffline, hrFlag: hrFlag, rrFlag: rrFlag,
      fmtTime: fmtTime, updateLabel: updateLabel, todayLine: todayLine,
    };
  }

  /* ================= 浏览器装配 ================= */
  if (typeof window === "undefined" || typeof document === "undefined") return;

  var POLL_MS = 1500;
  var seen = { alerts: null };
  var everHadData = false;   // 区分"暂无任何数据"与"离线（有历史）"

  function $(id) { return document.getElementById(id); }

  function setStateBadge(state, reason) {
    var badge = $("state-badge");
    badge.className = "badge state-" + (state || "absent");
    badge.textContent = { normal: "正常", sad: "情绪低落",
                          tired: "疲惫", absent: "不在场" }[state] || state || "—";
    $("state-reason").textContent = reason || "";
  }

  function metric(id, subId, value, unit, digits) {
    var el = $(id);
    if (value === null || value === undefined) {
      el.className = "metric none";
      el.textContent = "未检出";
      $(subId).textContent = "当前窗口未测出，属正常波动";
    } else {
      el.className = "metric";
      el.innerHTML = Number(value).toFixed(digits) + '<span class="unit">' + unit + "</span>";
    }
  }

  function setFlag(id, flag) {
    var el = $(id);
    el.className = "wave-flag " + flag.level;
    el.textContent = flag.text;
  }

  function renderLive(data) {
    var vitals = data.vitals, state = data.state;
    var threshold = data.offline_threshold || 30;
    var offline = isOffline(data.latest_sample_ts, threshold, data.server_ts);

    if (!vitals && !state) {
      if (!everHadData) {
        $("nodata-banner").style.display = "";
        $("live-area").style.display = "none";
        return;
      }
      offline = true;   // 有过历史数据但查询不到 → 离线
    }
    everHadData = true;
    $("nodata-banner").style.display = "none";
    $("live-area").style.display = "";

    $("online-badge").style.display = offline ? "" : "none";
    $("conn-dot").className = "dot " + (offline ? "off" : "on");
    $("last-update").textContent = updateLabel(offline, vitals, data.server_ts);

    if (state) setStateBadge(state.state, state.reason);

    metric("hr", "hr-sub", vitals && vitals.hr, "次/分", 0);
    metric("rr", "rr-sub", vitals && vitals.rr, "次/分", 1);

    /* 波形（数字与波动同源：脉冲/正弦频率由实时 hr/rr 驱动，见 waveform.js）。
       离线时波形停成灰平线（setRate(null)），不假装还在测。 */
    var hr = vitals && vitals.hr, rr = vitals && vitals.rr;
    var sqi = vitals ? vitals.sqi : null;
    hrWave.setRate(offline ? null : hr);
    rrWave.setRate(offline ? null : rr);
    setFlag("hr-flag", hrFlag(hr, sqi));
    setFlag("rr-flag", rrFlag(rr));

    var sqiEl = $("sqi");
    if (vitals && vitals.sqi !== null && vitals.sqi !== undefined) {
      sqiEl.className = "metric";
      sqiEl.innerHTML = Number(vitals.sqi).toFixed(2);
    } else {
      sqiEl.className = "metric none";
      sqiEl.textContent = "—";
    }

    var line = todayLine(data.today);
    if (line) $("today-body").innerHTML = line;
  }

  function renderAlerts(alerts) {
    var rows = $("alert-rows");
    if (!alerts || !alerts.length) {
      rows.innerHTML = '<tr><td colspan="3" class="sub">暂无预警记录</td></tr>';
      return;
    }
    rows.innerHTML = alerts.map(function (a) {
      return "<tr><td>" + a.time + '</td><td class="kind">' + a.kind +
             "</td><td>" + (a.text || "—") + "</td></tr>";
    }).join("");
  }

  function poll() {
    /* API 地址由模板注入（静态 JS 不经过 Django 模板引擎） */
    var urls = window.FAMILY_URLS || {};
    fetch(urls.live).then(function (r) { return r.json(); })
      .then(function (res) {
        if (res.status === 0) renderLive(res.data);
      }).catch(function () { /* 网络断开：保留上次画面，下一轮重试 */ });
    fetch(urls.alerts).then(function (r) { return r.json(); })
      .then(function (res) {
        if (res.status === 0) {
          var alerts = res.data.alerts;
          if (JSON.stringify(alerts) !== seen.alerts) {
            seen.alerts = JSON.stringify(alerts);
            renderAlerts(alerts);
          }
        }
      }).catch(function () { });
  }

  /* 波形实例：先建再轮询 —— 首次 poll() 里 renderLive 就会调 setRate。
     本脚本在 body 尾部，canvas 元素已存在，clientWidth 可同步取到。 */
  var hrWave = new FamilyWaveform($("hr-wave"), { type: "ecg", color: "#e2704a" });
  var rrWave = new FamilyWaveform($("rr-wave"), { type: "resp", color: "#3a9d6e" });

  poll();
  setInterval(poll, POLL_MS);
})(typeof window !== "undefined" ? window : globalThis);
