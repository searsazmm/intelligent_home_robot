/* 一周统计页：折线图（心率/呼吸日均值 + 心率区间带）、状态饼图、时长条形图。 */
(function () {
  "use strict";

  var STATE_COLORS = { normal: "#2e9e5b", tired: "#d9822b",
                       sad: "#3c7dd9", absent: "#8a95a1" };
  var STATE_LABELS = { normal: "正常", tired: "疲惫",
                       sad: "情绪低落", absent: "不在场" };

  function hasChart() { return typeof window.Chart === "function"; }

  /* 心率波动范围带的两条隐藏线在 datasets 里的固定位置（0=最高 1=最低）。 */
  var BAND_UPPER = 0, BAND_LOWER = 1;

  function mergedBandLegend(chart) {
    /* 复用默认图例条目，把最高/最低两条合并成一条"心率波动范围"。 */
    var items = window.Chart.defaults.plugins.legend.labels.generateLabels(chart);
    var out = [], bandDone = false;
    items.forEach(function (item) {
      if (item.datasetIndex === BAND_UPPER || item.datasetIndex === BAND_LOWER) {
        if (!bandDone) {
          var hidden = chart.getDatasetMeta(BAND_UPPER).hidden === true;
          out.push({ text: "心率波动范围", fillStyle: "rgba(47,111,178,.12)",
                     strokeStyle: "rgba(47,111,178,.35)", lineWidth: 1,
                     hidden: hidden, datasetIndex: BAND_UPPER });
          bandDone = true;
        }
      } else {
        out.push(item);
      }
    });
    return out;
  }

  function toggleBandLegend(e, legendItem, legend) {
    var chart = legend.chart;
    if (legendItem.text !== "心率波动范围") {
      window.Chart.defaults.plugins.legend.onClick(e, legendItem, legend);
      return;
    }
    [BAND_UPPER, BAND_LOWER].forEach(function (i) {
      var meta = chart.getDatasetMeta(i);
      meta.hidden = meta.hidden === null ? !chart.data.datasets[i].hidden : !meta.hidden;
    });
    chart.update();
  }

  function draw(series, distribution) {
    if (!hasChart()) return;   // 本地 vendor 与 CDN 都没加载成功：留白，不崩

    var labels = series.map(function (d) { return d.date.slice(5); });
    var hrData = series.map(function (d) { return d.hr_avg; });
    var rrData = series.map(function (d) { return d.rr_avg; });

    /* 心率区间带 = 每天真实测量的最低~最高（不裁剪、不美化）。
       Chart.js 没有原生 band，用填充面积曲线实现（上/下界各一条
       无点隐藏线，fill 相互夹出区间）。正常的 60~100 参考口径
       在报告页词卡与 hover 提示里呈现，这里只画真实数据。 */
    var hrHi = series.map(function (d) { return d.hr_max; });
    var hrLo = series.map(function (d) { return d.hr_min; });

    new Chart(document.getElementById("chart-vitals"), {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          { label: "心率最高", data: hrHi, borderColor: "transparent",
            pointRadius: 0, backgroundColor: "rgba(47,111,178,.12)", fill: "+1", tension: .3 },
          { label: "心率最低", data: hrLo, borderColor: "transparent",
            pointRadius: 0, backgroundColor: "rgba(47,111,178,.12)", fill: "-1", tension: .3 },
          { label: "心率日均值", data: hrData, borderColor: "#c0392b",
            backgroundColor: "#c0392b", spanGaps: false, tension: .3 },
          { label: "呼吸日均值", data: rrData, borderColor: "#2e9e5b",
            backgroundColor: "#2e9e5b", spanGaps: false, tension: .3, yAxisID: "y2" }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            /* 最高/最低两条隐藏线在图例里合并成一条"心率波动范围"；
               点击这一条同时隐藏/显示两条，带子整体开合 */
            labels: { generateLabels: mergedBandLegend },
            onClick: toggleBandLegend
          }
        },
        scales: {
          y: { min: 40, max: 130, title: { display: true, text: "心率（次/分）" } },
          y2: { position: "right", min: 0, max: 40,
                title: { display: true, text: "呼吸（次/分）" }, grid: { drawOnChartArea: false } }
        }
      }
    });

    var pieLabels = distribution.map(function (d) { return STATE_LABELS[d.state] || d.state; });
    var pieData = distribution.map(function (d) {
      return Math.round(d.ratio * 1000) / 10;   // 百分比，保留 1 位
    });
    var pieColors = distribution.map(function (d) { return STATE_COLORS[d.state] || "#ccc"; });

    new Chart(document.getElementById("chart-state-pie"), {
      type: "pie",
      data: { labels: pieLabels,
              datasets: [{ data: pieData, backgroundColor: pieColors }] },
      options: { responsive: true, maintainAspectRatio: false,
                 plugins: { tooltip: { callbacks: {
                   label: function (ctx) { return ctx.label + " " + ctx.parsed + "%"; } } } } }
    });

    var barLabels = distribution.map(function (d) { return STATE_LABELS[d.state] || d.state; });
    var barData = distribution.map(function (d) { return Math.round(d.seconds / 60); });

    new Chart(document.getElementById("chart-state-bar"), {
      type: "bar",
      data: { labels: barLabels,
              datasets: [{ label: "日均时长（分钟）",
                           data: barData, backgroundColor: pieColors }] },
      options: { responsive: true, maintainAspectRatio: false,
                 plugins: { legend: { display: false } } }
    });

    var covered = distribution.length && distribution[distribution.length - 1].span_covered;
    if (covered !== undefined && covered < 0.99) {
      document.getElementById("gap-note").textContent =
        "说明：状态记录共覆盖 " + Math.round(covered * 100) +
        "% 的时间，其余为机器人未运行的时段（含重启断档），不计入统计。";
    }
  }

  fetch(window.FAMILY_URLS.stats).then(function (r) { return r.json(); })
    .then(function (res) {
      if (res.status === 0) draw(res.data.series, res.data.distribution);
    });
})();
