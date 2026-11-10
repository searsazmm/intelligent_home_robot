/* 实时监测页：1.5 秒轮询 api/live + api/alerts。

   离线判定（前端实现，方案文档 v3 追加项 2）：
   最新样本 ts 与**浏览器本地时间**差值超过阈值(30s) → 离线徽标。
   ⚠️ 已知局限：机器人主机与家属浏览器时钟不一致会误判（提前/延迟
   判离线）。二期改为 api_live 返回服务端时间，用服务端时间差判断。
   */
(function () {
  "use strict";

  var POLL_MS = 1500;
  var seen = { vitals: null, state: null, today: null, alerts: null };
  var everHadData = false;   // 区分"暂无任何数据"与"离线（有历史）"

  function $(id) { return document.getElementById(id); }

  function fmtTime(ts) {
    var d = new Date(ts * 1000);
    function p(n) { return (n < 10 ? "0" : "") + n; }
    return p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
  }

  function setStateBadge(state, reason) {
    var badge = $("state-badge");
    badge.className = "badge state-" + (state || "absent");
    badge.textContent = { normal: "正常", sad: "情绪低落",
                          tired: "疲惫", absent: "不在场" }[state] || state || "—";
    $("state-reason").textContent = reason || "";
  }

  function isOffline(latestSampleTs, threshold) {
    if (!latestSampleTs) return true;
    // 浏览器本地时间与样本 ts 对比（时钟不一致的局限见文件头注释）
    return (Date.now() / 1000 - latestSampleTs) > threshold;
  }

  function renderLive(data) {
    var vitals = data.vitals, state = data.state;
    var threshold = data.offline_threshold || 30;
    var offline = isOffline(data.latest_sample_ts, threshold);

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
    $("last-update").textContent = offline
      ? "最后数据：" + (vitals ? fmtTime(vitals.ts) : "—")
      : "更新于 " + fmtTime(data.timestamp);

    if (state) setStateBadge(state.state, state.reason);

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
    metric("hr", "hr-sub", vitals && vitals.hr, "次/分", 0);
    metric("rr", "rr-sub", vitals && vitals.rr, "次/分", 1);
    var sqiEl = $("sqi");
    if (vitals && vitals.sqi !== null && vitals.sqi !== undefined) {
      sqiEl.className = "metric";
      sqiEl.innerHTML = Number(vitals.sqi).toFixed(2);
    } else {
      sqiEl.className = "metric none";
      sqiEl.textContent = "—";
    }

    var today = data.today;
    if (today) {
      var dist = today.state_distribution || [];
      var parts = dist.map(function (d) {
        var minutes = Math.round(d.seconds / 60);
        return d.state + " " + Math.round(d.ratio * 100) + "%（约 " + minutes + " 分钟）";
      });
      $("today-body").innerHTML =
        "今日样本 " + today.sample_count + " 条 · 平均心率 " +
        (today.hr_avg !== null ? today.hr_avg + " 次/分" : "未检出") +
        " · 平均呼吸 " +
        (today.rr_avg !== null ? today.rr_avg + " 次/分" : "未检出") +
        " · 今日预警 " + today.alert_count + " 次" +
        (parts.length ? "<br>状态分布：" + parts.join("，") : "<br>今日还没有状态记录");
    }
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

  poll();
  setInterval(poll, POLL_MS);
})();
