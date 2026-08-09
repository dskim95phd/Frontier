#!/usr/bin/env python3
"""Generate a self-contained time-series HTML report for a CPU capacity sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "tracelab_vera_rubin_p8_d16_cpu_capacity_sweep_continuous_4h_r0p6_20260808"
)


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def ratio_percent(numerator: float, denominator: float) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


def finite_or_none(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_case_series(
    *,
    case_dir: Path,
    utilization_rows: list[dict[str, str]],
    horizon_s: float,
    bin_seconds: int,
) -> dict[str, Any]:
    bin_count = math.ceil(horizon_s / bin_seconds)
    bins: list[dict[str, Any]] = []
    for index in range(bin_count):
        start = index * bin_seconds
        end = min((index + 1) * bin_seconds, horizon_s)
        bins.append(
            {
                "start_s": start,
                "end_s": end,
                "duration_s": end - start,
                "arrival_count": 0,
                "root_start_count": 0,
                "split_start_count": 0,
                "completion_count": 0,
                "completed_arrival_count": 0,
                "ttft_s": [],
                "tpot_ms": [],
                "query": 0,
                "gpu_hit": 0,
                "cpu_query": 0,
                "cpu_hit": 0,
                "combined_hit": 0,
                "restore_bytes": 0,
                "offload_bytes": 0,
                "busy_pct": None,
                "compute_active_pct": None,
                "attention_t": None,
            }
        )

    for row in read_csv(case_dir / "request_arrivals_5min.csv"):
        index = min(int(float(row["start_time_s"]) // bin_seconds), bin_count - 1)
        bins[index]["arrival_count"] += int(row["total_request_arrivals"])
        bins[index]["root_start_count"] += int(row["source_root_session_starts"])
        bins[index]["split_start_count"] += int(row["split_session_starts"])

    for row in utilization_rows:
        index = min(int(float(row["start_time_s"]) // bin_seconds), bin_count - 1)
        bins[index]["busy_pct"] = float(row["pool_busy_pct"])
        bins[index]["compute_active_pct"] = float(row["compute_active_pct"])
        bins[index]["attention_t"] = float(row["prefill_attention_pairs_trillion"])

    with (case_dir / "requests.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            arrived = float(row["arrived_at_s"])
            completed = float(row["completed_at_s"])
            arrival_index = min(max(int(arrived // bin_seconds), 0), bin_count - 1)
            completion_index = min(max(int(completed // bin_seconds), 0), bin_count - 1)
            arrival_bin = bins[arrival_index]
            bins[completion_index]["completion_count"] += 1
            arrival_bin["completed_arrival_count"] += 1
            arrival_bin["ttft_s"].append(float(row["ttft_ms"]) / 1000.0)
            decode_tokens = int(row["num_decode_tokens"])
            if decode_tokens > 1:
                tail_ms = (
                    float(row["completed_at_s"])
                    - float(row["first_token_completed_at_s"])
                ) * 1000.0
                arrival_bin["tpot_ms"].append(tail_ms / (decode_tokens - 1))
            arrival_bin["query"] += int(row["prefix_cache_query_blocks"])
            arrival_bin["gpu_hit"] += int(row["gpu_prefix_hit_blocks"])
            arrival_bin["cpu_query"] += int(row["cpu_prefix_query_blocks"])
            arrival_bin["cpu_hit"] += int(row["cpu_prefix_hit_blocks"])
            arrival_bin["combined_hit"] += int(row["prefix_cache_hit_blocks"])
            arrival_bin["restore_bytes"] += int(row["cpu_restore_bytes"])
            arrival_bin["offload_bytes"] += int(row["cpu_offload_bytes"])

    series: dict[str, list[float | None]] = {
        "time_h": [],
        "busy_pct": [],
        "compute_active_pct": [],
        "attention_t": [],
        "arrival_rps": [],
        "root_start_rps": [],
        "split_start_rps": [],
        "exogenous_start_rps": [],
        "successor_arrival_rps": [],
        "completion_rps": [],
        "backlog_delta_rps": [],
        "completion_coverage_pct": [],
        "ttft_mean_s": [],
        "ttft_p50_s": [],
        "ttft_p90_s": [],
        "ttft_p99_s": [],
        "tpot_mean_ms": [],
        "tpot_p90_ms": [],
        "gpu_hit_pct": [],
        "cpu_conditional_hit_pct": [],
        "combined_hit_pct": [],
        "miss_pct": [],
        "restore_gbps": [],
        "offload_gbps": [],
    }
    for item in bins:
        duration = item["duration_s"]
        query = item["query"]
        combined = item["combined_hit"]
        ttft = item["ttft_s"]
        tpot = item["tpot_ms"]
        series["time_h"].append(round(item["end_s"] / 3600.0, 6))
        series["busy_pct"].append(finite_or_none(item["busy_pct"]))
        series["compute_active_pct"].append(finite_or_none(item["compute_active_pct"]))
        series["attention_t"].append(finite_or_none(item["attention_t"]))
        series["arrival_rps"].append(item["arrival_count"] / duration)
        series["root_start_rps"].append(item["root_start_count"] / duration)
        series["split_start_rps"].append(item["split_start_count"] / duration)
        series["exogenous_start_rps"].append(
            (item["root_start_count"] + item["split_start_count"]) / duration
        )
        series["successor_arrival_rps"].append(
            (
                item["arrival_count"]
                - item["root_start_count"]
                - item["split_start_count"]
            )
            / duration
        )
        series["completion_rps"].append(item["completion_count"] / duration)
        series["backlog_delta_rps"].append(
            (item["arrival_count"] - item["completion_count"]) / duration
        )
        series["completion_coverage_pct"].append(
            ratio_percent(item["completed_arrival_count"], item["arrival_count"])
        )
        series["ttft_mean_s"].append(fmean(ttft) if ttft else None)
        series["ttft_p50_s"].append(percentile(ttft, 0.50))
        series["ttft_p90_s"].append(percentile(ttft, 0.90))
        series["ttft_p99_s"].append(percentile(ttft, 0.99))
        series["tpot_mean_ms"].append(fmean(tpot) if tpot else None)
        series["tpot_p90_ms"].append(percentile(tpot, 0.90))
        series["gpu_hit_pct"].append(ratio_percent(item["gpu_hit"], query))
        series["cpu_conditional_hit_pct"].append(
            ratio_percent(item["cpu_hit"], item["cpu_query"])
        )
        series["combined_hit_pct"].append(ratio_percent(combined, query))
        series["miss_pct"].append(ratio_percent(query - combined, query))
        series["restore_gbps"].append(item["restore_bytes"] / duration / 1e9)
        series["offload_gbps"].append(item["offload_bytes"] / duration / 1e9)
    return series


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>P8 CPU KV cache capacity time report</title>
  <style>
    :root { color-scheme: light dark; --bg:#f7f8fb; --panel:#fff; --fg:#172033; --muted:#5b6475; --border:#d8dde8; --grid:#e7eaf0; --s1:#2563eb; --s2:#ea580c; --s3:#059669; --s4:#7c3aed; --s5:#dc2626; --s6:#0891b2; --soft:#eef3ff; }
    @media (prefers-color-scheme: dark) { :root { --bg:#0f1219; --panel:#171b24; --fg:#edf1f7; --muted:#a6afbf; --border:#31394a; --grid:#252c39; --s1:#60a5fa; --s2:#fb923c; --s3:#34d399; --s4:#a78bfa; --s5:#f87171; --s6:#22d3ee; --soft:#1c2940; } }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
    main { max-width:1480px; margin:0 auto; padding:28px; }
    h1 { margin:0 0 6px; font-size:26px; font-weight:650; }
    h2 { margin:28px 0 12px; font-size:19px; }
    h3 { margin:0; font-size:15px; font-weight:650; }
    p { margin:4px 0; }
    .muted { color:var(--muted); }
    .toolbar { display:flex; gap:16px; align-items:end; flex-wrap:wrap; margin:20px 0; padding:16px; background:var(--panel); border:1px solid var(--border); border-radius:12px; }
    label { display:grid; gap:6px; color:var(--muted); font-size:12px; }
    select { min-width:170px; padding:8px 10px; border:1px solid var(--border); border-radius:8px; background:var(--panel); color:var(--fg); font:inherit; }
    .facts { display:grid; grid-template-columns:repeat(4,minmax(150px,1fr)); gap:12px; margin:0 0 20px; }
    .fact { padding:14px 16px; background:var(--panel); border:1px solid var(--border); border-radius:12px; }
    .fact span { display:block; color:var(--muted); font-size:12px; }
    .fact strong { display:block; margin-top:3px; font-size:22px; font-weight:650; }
    .charts { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
    .chart { position:relative; min-width:0; padding:16px; background:var(--panel); border:1px solid var(--border); border-radius:12px; }
    .chart-note { min-height:21px; color:var(--muted); font-size:12px; }
    .chart svg { width:100%; height:auto; display:block; margin-top:8px; }
    .axis { stroke:var(--muted); stroke-width:1; }
    .grid { stroke:var(--grid); stroke-width:1; }
    .tick { fill:var(--muted); font-size:11px; }
    .legend { display:flex; flex-wrap:wrap; gap:12px; min-height:21px; color:var(--muted); font-size:12px; }
    .legend i { display:inline-block; width:14px; height:3px; margin-right:5px; vertical-align:middle; }
    .tooltip { position:absolute; display:none; pointer-events:none; z-index:2; max-width:280px; padding:8px 10px; color:var(--fg); background:var(--panel); border:1px solid var(--border); border-radius:8px; box-shadow:0 5px 18px rgba(0,0,0,.16); font-size:12px; white-space:nowrap; }
    .table-wrap { overflow-x:auto; background:var(--panel); border:1px solid var(--border); border-radius:12px; }
    table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
    th,td { padding:9px 11px; border-bottom:1px solid var(--border); text-align:right; white-space:nowrap; }
    th:first-child,td:first-child { text-align:left; }
    th { color:var(--muted); font-size:12px; font-weight:600; }
    tr:last-child td { border-bottom:0; }
    tr.recommended { background:var(--soft); }
    .notes { margin-top:18px; padding:14px 16px; border-left:4px solid var(--s1); background:var(--panel); }
    code { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }
    @media (max-width:900px) { main{padding:18px}.charts{grid-template-columns:1fr}.facts{grid-template-columns:repeat(2,minmax(0,1fr))} }
    @media (max-width:520px) { .facts{grid-template-columns:1fr} h1{font-size:21px} }
  </style>
</head>
<body>
<main>
  <h1>P8 CPU KV cache capacity sweep</h1>
  <p class="muted" id="subtitle"></p>

  <div class="toolbar">
    <label>CPU DRAM 용량
      <select id="capacity-select"></select>
    </label>
    <p class="muted" id="capacity-status">표의 용량은 PREFILL GPU 2개가 공유하는 물리 CPU DRAM이며, GPU별 slice는 그 절반입니다.</p>
  </div>

  <section class="facts" id="facts"></section>

  <section class="charts">
    <article class="chart" id="util-chart"><h3>PREFILL GPU 사용률</h3><p class="chart-note">90% 선은 지속 포화 경계입니다.</p><div class="legend"></div><svg viewBox="0 0 720 270"></svg><div class="tooltip"></div></article>
    <article class="chart" id="rate-chart"><h3>Request 유입 구성과 완료</h3><p class="chart-note">Root/split은 외생 주입, successor는 이전 turn 완료 후 생성되는 closed-loop 유입입니다.</p><div class="legend"></div><svg viewBox="0 0 720 270"></svg><div class="tooltip"></div></article>
    <article class="chart" id="ttft-chart"><h3>TTFT</h3><p class="chart-note">도착 시각 기준, 완료된 요청만 포함합니다.</p><div class="legend"></div><svg viewBox="0 0 720 270"></svg><div class="tooltip"></div></article>
    <article class="chart" id="tpot-chart"><h3>TPOT</h3><p class="chart-note">도착 시각 기준, decode token이 2개 이상인 완료 요청입니다.</p><div class="legend"></div><svg viewBox="0 0 720 270"></svg><div class="tooltip"></div></article>
    <article class="chart" id="hit-chart"><h3>Prefix cache hit</h3><p class="chart-note">CPU hit은 GPU miss 후 CPU에서 찾은 block의 conditional hit입니다.</p><div class="legend"></div><svg viewBox="0 0 720 270"></svg><div class="tooltip"></div></article>
    <article class="chart" id="work-chart"><h3>PREFILL attention work</h3><p class="chart-note">5분 동안 실행된 attention token-pair 총량입니다.</p><div class="legend"></div><svg viewBox="0 0 720 270"></svg><div class="tooltip"></div></article>
    <article class="chart" id="transfer-chart"><h3>CPU KV transfer bandwidth</h3><p class="chart-note">요청 도착 구간에 귀속한 평균 restore/offload 처리량입니다.</p><div class="legend"></div><svg viewBox="0 0 720 270"></svg><div class="tooltip"></div></article>
    <article class="chart" id="coverage-chart"><h3>완료 coverage</h3><p class="chart-note">해당 구간 도착 요청 중 시뮬레이션 종료 전 완료된 비율입니다.</p><div class="legend"></div><svg viewBox="0 0 720 270"></svg><div class="tooltip"></div></article>
  </section>

  <h2>용량별 전체 구간 요약</h2>
  <div class="table-wrap"><table id="summary-table"></table></div>
  <div class="notes">
    <strong>해석 기준</strong>
    <p>후반 TTFT와 cache hit은 완료된 요청만 집계하므로, coverage가 낮은 포화 case에서는 실제 체감보다 낙관적으로 보일 수 있습니다. 저용량의 낮은 TPOT도 PREFILL 병목으로 DECODE까지 도달한 요청이 적은 효과를 포함합니다.</p>
  </div>
</main>
<script>
const REPORT = __REPORT_DATA__;
const COLORS = ['var(--s1)','var(--s2)','var(--s3)','var(--s4)','var(--s5)','var(--s6)'];
const NS = 'http://www.w3.org/2000/svg';
const fmt = (value, digits=1) => value == null || !Number.isFinite(value) ? '—' : Number(value).toFixed(digits);
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function lineChart(id, xValues, definitions, options={}) {
  const host = document.getElementById(id);
  const svg = host.querySelector('svg');
  const legend = host.querySelector('.legend');
  const tooltip = host.querySelector('.tooltip');
  svg.replaceChildren();
  legend.innerHTML = definitions.map((d,i) => `<span><i style="background:${COLORS[i]}"></i>${esc(d.name)}</span>`).join('');
  const W=720,H=270,m={l:58,r:18,t:16,b:38},pw=W-m.l-m.r,ph=H-m.t-m.b;
  const all = definitions.flatMap(d => d.values.filter(v => v != null && Number.isFinite(v)));
  let yMin = options.yMin ?? 0;
  let yMax = options.yMax ?? (all.length ? Math.max(...all) : 1);
  if (options.log) {
    const positive = all.filter(v => v > 0);
    yMin = Math.max(options.yMin ?? (positive.length ? Math.min(...positive) : .001), .001);
    yMax = Math.max(yMax, yMin*10);
  } else if (options.yMax == null) {
    yMax = Math.max(yMax*1.1, options.minRange ?? 1);
  }
  const xMin=0, xMax=Math.ceil(xValues[xValues.length-1] ?? 1);
  const sx=v => m.l + (v-xMin)/(xMax-xMin || 1)*pw;
  const sy=v => options.log ? m.t+(Math.log10(yMax)-Math.log10(Math.max(v,yMin)))/(Math.log10(yMax)-Math.log10(yMin))*ph : m.t+(yMax-v)/(yMax-yMin || 1)*ph;
  const add=(tag,attrs,text)=>{const e=document.createElementNS(NS,tag);Object.entries(attrs||{}).forEach(([k,v])=>e.setAttribute(k,v));if(text!=null)e.textContent=text;svg.appendChild(e);return e;};
  for(let i=0;i<=4;i++){
    const value = options.log ? Math.pow(10,Math.log10(yMin)+(Math.log10(yMax)-Math.log10(yMin))*i/4) : yMin+(yMax-yMin)*i/4;
    const py=sy(value); add('line',{x1:m.l,y1:py,x2:W-m.r,y2:py,class:'grid'}); add('text',{x:m.l-8,y:py+4,'text-anchor':'end',class:'tick'},options.formatY?options.formatY(value):fmt(value,1));
  }
  const hourStep=xMax>8?2:1;
  for(let hour=0;hour<=xMax;hour+=hourStep){const px=m.l+hour/xMax*pw;add('line',{x1:px,y1:m.t,x2:px,y2:H-m.b,class:'grid'});add('text',{x:px,y:H-13,'text-anchor':'middle',class:'tick'},`${hour}h`);}
  add('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,class:'axis'});
  add('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,class:'axis'});
  if(options.threshold!=null){const py=sy(options.threshold);add('line',{x1:m.l,y1:py,x2:W-m.r,y2:py,stroke:'var(--s5)','stroke-dasharray':'5 4','stroke-width':'1'});add('text',{x:W-m.r-3,y:py-5,'text-anchor':'end',class:'tick'},`${options.threshold}${options.unit||''}`);}
  definitions.forEach((d,index)=>{
    let path=''; let drawing=false;
    d.values.forEach((v,i)=>{if(v==null||!Number.isFinite(v)){drawing=false;return;}path+=(drawing?'L':'M')+sx(xValues[i]).toFixed(1)+' '+sy(v).toFixed(1)+' ';drawing=true;});
    add('path',{d:path.trim(),fill:'none',stroke:COLORS[index],'stroke-width':index===0?'2.5':'2','stroke-linejoin':'round','stroke-linecap':'round'});
  });
  const guide=add('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,stroke:'var(--muted)','stroke-width':'1','stroke-dasharray':'3 3',visibility:'hidden'});
  const overlay=add('rect',{x:m.l,y:m.t,width:pw,height:ph,fill:'transparent'});
  overlay.addEventListener('pointerleave',()=>{tooltip.style.display='none';guide.setAttribute('visibility','hidden');});
  overlay.addEventListener('pointermove',event=>{
    const box=svg.getBoundingClientRect(); const localX=(event.clientX-box.left)*W/box.width; const ratio=Math.max(0,Math.min(1,(localX-m.l)/pw)); const idx=Math.max(0,Math.min(xValues.length-1,Math.round(ratio*(xValues.length-1)))); const px=sx(xValues[idx]);
    guide.setAttribute('x1',px);guide.setAttribute('x2',px);guide.setAttribute('visibility','visible');
    tooltip.innerHTML=`<strong>${fmt(xValues[idx],2)}h</strong><br>`+definitions.map((d,i)=>`${esc(d.name)}: ${fmt(d.values[idx],options.tooltipDigits??2)}${options.unit||''}`).join('<br>');
    tooltip.style.display='block'; const hostBox=host.getBoundingClientRect(); const left=Math.min(hostBox.width-tooltip.offsetWidth-8,Math.max(8,event.clientX-hostBox.left+12)); tooltip.style.left=left+'px'; tooltip.style.top=Math.max(48,event.clientY-hostBox.top-tooltip.offsetHeight-10)+'px';
  });
}

function render() {
  const key=document.getElementById('capacity-select').value;
  const entry=REPORT.cases[key], s=entry.series, o=entry.overall;
  document.getElementById('facts').innerHTML=[
    ['전체 TTFT 평균',`${fmt(o.ttft_mean_ms/1000,2)} s`],['전체 TTFT p90',`${fmt(o.ttft_p90_ms/1000,2)} s`],['통합 cache hit',`${fmt(o.combined_hit_pct,2)}%`],['마지막 1h PREFILL',`${fmt(entry.last_hour_busy_pct,2)}%`]
  ].map(([k,v])=>`<div class="fact"><span>${k}</span><strong>${v}</strong></div>`).join('');
  lineChart('util-chart',s.time_h,[{name:'Pool busy',values:s.busy_pct},{name:'Compute active',values:s.compute_active_pct}],{yMin:0,yMax:105,threshold:90,unit:'%',tooltipDigits:1});
  lineChart('rate-chart',s.time_h,[{name:'Total arrival',values:s.arrival_rps},{name:'Completion',values:s.completion_rps},{name:'Source root',values:s.root_start_rps},{name:'Split root',values:s.split_start_rps},{name:'Successor',values:s.successor_arrival_rps}],{yMin:0,unit:' req/s',tooltipDigits:2});
  lineChart('ttft-chart',s.time_h,[{name:'Mean',values:s.ttft_mean_s},{name:'p50',values:s.ttft_p50_s},{name:'p90',values:s.ttft_p90_s},{name:'p99',values:s.ttft_p99_s}],{yMin:.01,log:true,unit:' s',tooltipDigits:2,formatY:v=>v>=100?fmt(v,0):v>=10?fmt(v,1):fmt(v,2)});
  lineChart('tpot-chart',s.time_h,[{name:'Mean',values:s.tpot_mean_ms},{name:'p90',values:s.tpot_p90_ms}],{yMin:0,unit:' ms',tooltipDigits:2});
  lineChart('hit-chart',s.time_h,[{name:'GPU hit',values:s.gpu_hit_pct},{name:'CPU conditional hit',values:s.cpu_conditional_hit_pct},{name:'Combined hit',values:s.combined_hit_pct},{name:'Miss',values:s.miss_pct}],{yMin:0,yMax:100,unit:'%',tooltipDigits:1});
  lineChart('work-chart',s.time_h,[{name:'Attention pairs',values:s.attention_t}],{yMin:0,unit:' T/5min',tooltipDigits:2});
  lineChart('transfer-chart',s.time_h,[{name:'Restore',values:s.restore_gbps},{name:'Offload',values:s.offload_gbps}],{yMin:0,unit:' GB/s',tooltipDigits:2});
  lineChart('coverage-chart',s.time_h,[{name:'Completed coverage',values:s.completion_coverage_pct}],{yMin:0,yMax:105,threshold:90,unit:'%',tooltipDigits:1});
}

function buildSummaryTable(){
  const head=['CPU DRAM','TTFT mean','TTFT p90','TPOT','GPU hit','CPU hit','Combined','마지막 1h busy'];
  const rows=Object.values(REPORT.cases).map(e=>{const o=e.overall;return `<tr class="${e.capacity_gb===REPORT.meta.recommended_capacity_gb?'recommended':''}"><td>${fmt(e.capacity_gb/1000,1)} TB</td><td>${fmt(o.ttft_mean_ms/1000,2)} s</td><td>${fmt(o.ttft_p90_ms/1000,2)} s</td><td>${fmt(o.tpot_mean_ms,2)} ms</td><td>${fmt(o.gpu_hit_pct,2)}%</td><td>${fmt(o.cpu_conditional_hit_pct,2)}%</td><td>${fmt(o.combined_hit_pct,2)}%</td><td>${fmt(e.last_hour_busy_pct,2)}%</td></tr>`;}).join('');
  document.getElementById('summary-table').innerHTML=`<thead><tr>${head.map(x=>`<th>${x}</th>`).join('')}</tr></thead><tbody>${rows}</tbody>`;
}

document.getElementById('subtitle').textContent=`${fmt(REPORT.meta.session_rate,1)} source session/s · ${fmt(REPORT.meta.horizon_s/3600,0)}시간 연속 주입 · P${REPORT.meta.prefill_gpus}/D${REPORT.meta.decode_gpus} · ${fmt(REPORT.meta.bin_seconds/60,0)}분 집계`;
const select=document.getElementById('capacity-select');
Object.values(REPORT.cases).forEach(e=>{const option=document.createElement('option');option.value=String(e.capacity_gb);option.textContent=`${fmt(e.capacity_gb/1000,1)} TB (GPU당 ${fmt(e.capacity_gb/2000,2)} TB)`;option.selected=e.capacity_gb===2500;select.appendChild(option);});
select.value=String(REPORT.meta.default_capacity_gb);
document.getElementById('capacity-status').textContent=REPORT.meta.recommended_capacity_gb==null
  ? '이 시간 범위에서는 5분 PREFILL 사용률을 90% 미만으로 유지한 용량이 없습니다. 최대 용량을 기본 표시합니다.'
  : `5분 PREFILL 사용률이 90% 미만인 최소 용량은 ${fmt(REPORT.meta.recommended_capacity_gb/1000,1)} TB입니다.`;
select.addEventListener('change',render);
buildSummaryTable(); render();
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-html", type=Path)
    parser.add_argument("--bin-seconds", type=int, default=300)
    args = parser.parse_args()
    root = args.output_root.resolve()
    output_html = (args.output_html or (root / "capacity_time_report.html")).resolve()
    if args.bin_seconds <= 0:
        raise SystemExit("--bin-seconds must be positive")
    sweep = json.loads((root / "sweep.json").read_text(encoding="utf-8"))
    horizon_s = float(sweep["simulation_end_time_s"])
    overall_rows = {
        int(row["physical_cpu_dram_gb"]): row
        for row in read_csv(root / "sweep_metrics.csv")
    }
    utilization_by_capacity: dict[int, list[dict[str, str]]] = {}
    for row in read_csv(root / "prefill_gpu_utilization_5min.csv"):
        utilization_by_capacity.setdefault(int(row["physical_cpu_dram_gb"]), []).append(row)

    workload = Path(sweep["workload"])
    metadata_path = workload.with_name(workload.stem + "_metadata.json")
    session_rate = None
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        session_rate = metadata["sampling"]["session_arrival_rate_per_second"]

    cases: dict[str, Any] = {}
    for capacity_gb in sorted(overall_rows):
        case_dir = root / f"cpu{capacity_gb:04d}gb" / "r1"
        series = build_case_series(
            case_dir=case_dir,
            utilization_rows=utilization_by_capacity[capacity_gb],
            horizon_s=horizon_s,
            bin_seconds=args.bin_seconds,
        )
        last_hour_start = max(0.0, horizon_s - 3600.0)
        last_hour_busy_numerator = 0.0
        last_hour_busy_denominator = 0.0
        for row in utilization_by_capacity[capacity_gb]:
            start = float(row["start_time_s"])
            end = float(row["end_time_s"])
            overlap = max(0.0, min(end, horizon_s) - max(start, last_hour_start))
            if overlap:
                last_hour_busy_numerator += overlap * float(row["pool_busy_pct"])
                last_hour_busy_denominator += overlap
        overall: dict[str, Any] = {}
        for key, value in overall_rows[capacity_gb].items():
            if value == "":
                overall[key] = None
            elif key in {"physical_cpu_dram_gb", "requests", "restore_operations", "offload_operations", "migrations"}:
                overall[key] = int(float(value))
            elif key in {"prefix_query_blocks", "gpu_hit_blocks", "cpu_query_blocks", "cpu_hit_blocks", "prefill_scheduled_tokens", "prefill_attention_token_pairs"}:
                overall[key] = int(value)
            else:
                try:
                    overall[key] = float(value)
                except ValueError:
                    overall[key] = value
        cases[str(capacity_gb)] = {
            "capacity_gb": capacity_gb,
            "overall": overall,
            "last_hour_busy_pct": (
                last_hour_busy_numerator / last_hour_busy_denominator
                if last_hour_busy_denominator
                else None
            ),
            "series": series,
        }

    stable_capacities = [
        int(key)
        for key, case in cases.items()
        if max(
            value for value in case["series"]["busy_pct"] if value is not None
        )
        < 90.0
    ]
    recommended_capacity_gb = min(stable_capacities) if stable_capacities else None
    default_capacity_gb = (
        recommended_capacity_gb
        if recommended_capacity_gb is not None
        else max(int(key) for key in cases)
    )
    report = {
        "meta": {
            "horizon_s": horizon_s,
            "bin_seconds": args.bin_seconds,
            "session_rate": session_rate,
            "prefill_gpus": int(sweep["prefill_gpus"]),
            "decode_gpus": int(sweep["decode_gpus"]),
            "capacity_mapping": sweep["capacity_mapping"],
            "recommended_capacity_gb": recommended_capacity_gb,
            "default_capacity_gb": default_capacity_gb,
        },
        "cases": cases,
    }
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(HTML_TEMPLATE.replace("__REPORT_DATA__", payload), encoding="utf-8")
    print(json.dumps({"output_html": str(output_html), "bytes": output_html.stat().st_size, "cases": len(cases)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
