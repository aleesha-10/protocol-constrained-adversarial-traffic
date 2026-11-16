"""Build a self-contained interactive evidence dashboard from the stored per-sample attack results.

  python scripts/build_dashboard.py
  python scripts/build_dashboard.py --npz results/metrics/adversarial_per_sample_OFFICIAL.npz --out results/dav/dashboard.html

Open the HTML file in any browser. It needs no internet connection, no server and no extra libraries.

Guardrails built into the page (so the evidence cannot quietly overstate the attack):
  - the denominators are always visible: sampled -> with a result -> known-class -> correct before the attack
  - flip rate counts only flows that were correct before the attack, with a Wilson 95% interval
  - every confidence axis is fixed to [0, 1] and never rescaled when the selection changes
  - classes with fewer than 30 flows are labelled 'too few flows' instead of being given a rate
"""
import argparse
import json
from pathlib import Path

import numpy as np

NAMES = ["Benign", "DDoS-UDP_Flood", "DoS-TCP_Flood", "Recon-PortScan"]
CLASSES = ["DDoS-UDP_Flood", "DoS-TCP_Flood", "Recon-PortScan", "DictionaryBruteForce"]


def build_payload(npz_path, json_path):
    d = np.load(npz_path, allow_pickle=True)
    eps = sorted({int(k.split("_")[0]) for k in d.files if k.split("_")[0].isdigit()})
    lab = d[f"{eps[0]}_label"].astype(str)
    cls_code = np.array([CLASSES.index(l) if l in CLASSES else -1 for l in lab])
    if (cls_code < 0).any():
        raise SystemExit("Unexpected class names in the per-sample file: " + str(sorted(set(lab[cls_code < 0]))))
    n_sampled = None
    if Path(json_path).exists():
        cfg = json.load(open(json_path)).get("config", {})
        if "max_flows_per_class" in cfg and "attack_classes" in cfg:
            n_sampled = int(cfg["max_flows_per_class"]) * len(cfg["attack_classes"])
    return dict(
        eps=eps, classes=CLASSES, names=NAMES, n_sampled=n_sampled,
        cls=cls_code.tolist(),
        p0=d[f"{eps[0]}_clean_pred"].astype(int).tolist(),
        c0=[round(float(x), 4) for x in d[f"{eps[0]}_clean_conf"]],
        cap=[round(float(x), 1) for x in d[f"{eps[0]}_n_changed"]],
        adv={str(e): dict(p=d[f"{e}_adv_pred"].astype(int).tolist(),
                          c=[round(float(x), 4) for x in d[f"{e}_adv_conf"]]) for e in eps},
    )


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Protocol-constrained attack: evidence dashboard</title>
<style>
 :root{--fg:#1f2328;--mut:#59636e;--bd:#d0d7de;--bg:#ffffff;--card:#f6f8fa}
 @media (prefers-color-scheme: dark){:root{--fg:#e6edf3;--mut:#9da7b3;--bd:#30363d;--bg:#0d1117;--card:#161b22}}
 body{margin:0;font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif;color:var(--fg);background:var(--bg)}
 header,main{max-width:1180px;margin:0 auto;padding:12px 16px}
 h1{font-size:19px;margin:6px 0}
 h2{font-size:14px;margin:0 0 6px}
 .banner{background:var(--card);border:1px solid var(--bd);border-left:4px solid #d29922;border-radius:6px;padding:8px 12px;margin:8px 0}
 .controls{display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin:10px 0}
 .controls label{font-weight:600}
 select,input[type=range]{margin-left:6px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}
 .card{background:var(--card);border:1px solid var(--bd);border-radius:6px;padding:10px 12px}
 table{border-collapse:collapse;width:100%}
 td,th{padding:3px 6px;border-bottom:1px solid var(--bd);text-align:left}
 td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
 .mut{color:var(--mut);font-size:12px}
 svg text{fill:var(--fg);font-size:10px}
 .warn{color:#b35900;font-weight:600}
 footer{max-width:1180px;margin:0 auto;padding:8px 16px 24px;color:var(--mut);font-size:12px}
</style></head><body>
<header>
 <h1>Protocol-constrained adversarial attack: evidence dashboard</h1>
 <div class="mut">ResNet-18 on image-encoded CICIoT2023 flows. Payload-only edits; header, packet count and lengths fixed; checksums repaired.</div>
 <div class="banner" id="banner"></div>
 <div class="controls">
  <span><label>Budget (bytes)</label><select id="eps"></select></span>
  <span><label>Class</label><select id="cls"></select></span>
  <span><label>Reject if confidence below</label><input id="tau" type="range" min="0" max="1" step="0.01" value="0.9"><b id="tauv">0.90</b></span>
 </div>
</header>
<main>
 <div class="grid">
  <div class="card"><h2>1. What happened to the selected flows</h2><div id="outcome"></div></div>
  <div class="card"><h2>2. Where they went (before &rarr; after)</h2><svg id="trans" width="340" height="250"></svg></div>
  <div class="card"><h2>3. Confidence per flow</h2><svg id="scatter" width="340" height="300"></svg><div class="mut">Axes fixed to 0&ndash;1. Dashed lines mark the rejection threshold.</div></div>
  <div class="card"><h2>4. Confidence threshold as a defense</h2><div id="abstain"></div></div>
  <div class="card"><h2>5. Editable capacity and flips</h2><div id="capacity"></div></div>
 </div>
</main>
<footer id="foot"></footer>
<script>
const DATA = /*DATA*/;
const OUT = {unch:["Correct, unchanged","#8b949e"], ben:["Flipped to Benign","#d62728"], oth:["Flipped to another attack class","#ff7f0e"],
             wrong:["Already wrong before the attack","#1f77b4"], uchg:["Unseen class: prediction changed","#8e44ad"], usame:["Unseen class: prediction unchanged","#c9d1d9"]};
const state = {eps: DATA.eps[DATA.eps.length-1], cls: "known", tau: 0.9};
const N = DATA.cls.length;
const CLS_LABEL = {known:"All known classes", 0:"DDoS-UDP_Flood", 1:"DoS-TCP_Flood", 2:"Recon-PortScan", 3:"DictionaryBruteForce (unseen)"};
const isKnown = i => DATA.cls[i] !== 3;
const trueIdx = i => ({0:1, 1:2, 2:3}[DATA.cls[i]]);   // index into NAMES for known classes

function wilson(k, n){ if(!n) return [NaN,NaN]; const z=1.96,p=k/n,d=1+z*z/n,c=p+z*z/(2*n),h=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return [(c-h)/d,(c+h)/d]; }
const pct = x => isNaN(x) ? "n/a" : (100*x).toFixed(1)+"%";
function selection(){ const out=[]; for(let i=0;i<N;i++){ if(state.cls==="known"? isKnown(i) : DATA.cls[i]===+state.cls) out.push(i); } return out; }
function outcome(i, e){
  const a = DATA.adv[e];
  if(!isKnown(i)) return a.p[i]!==DATA.p0[i] ? "uchg" : "usame";
  if(DATA.p0[i]!==trueIdx(i)) return "wrong";
  if(a.p[i]===DATA.p0[i]) return "unch";
  return a.p[i]===0 ? "ben" : "oth";
}
const svgEl = (tag, attrs, text) => { const s=`<${tag} ${Object.entries(attrs).map(([k,v])=>`${k}="${v}"`).join(" ")}>`; return s+(text!==undefined?text:"")+`</${tag}>`; };

function renderBanner(){
  let known=0, correct=0; for(let i=0;i<N;i++){ if(isKnown(i)){ known++; if(DATA.p0[i]===trueIdx(i)) correct++; } }
  const s = DATA.n_sampled ? `${DATA.n_sampled} flows sampled &rarr; <b>${N}</b> had editable payload and produced a result (<b>${DATA.n_sampled-N} produced none</b>)` : `<b>${N}</b> flows produced a result`;
  document.getElementById("banner").innerHTML = `<b>Denominators.</b> ${s} &rarr; ${known} known-class &rarr; ${correct} classified correctly before the attack. Flip rates use only the last group. One run, one seed.`;
}
function renderOutcome(){
  const idx = selection(), e = String(state.eps), counts = {};
  idx.forEach(i => { const o = outcome(i,e); counts[o]=(counts[o]||0)+1; });
  let html = `<table><tr><th>Outcome</th><th class="n">Flows</th></tr>`;
  Object.keys(OUT).forEach(k => { if(counts[k]) html += `<tr><td><span style="color:${OUT[k][1]}">&#9632;</span> ${OUT[k][0]}</td><td class="n">${counts[k]}</td></tr>`; });
  html += `<tr><td><b>Selected flows</b></td><td class="n"><b>${idx.length}</b></td></tr></table>`;
  if(state.cls===3){ html += `<p class="mut">The unseen class has no correct clean label, so no flip rate is defined. Counts show how many predictions changed.</p>`; }
  else {
    const okN = idx.filter(i => DATA.p0[i]===trueIdx(i)).length, k = (counts.ben||0)+(counts.oth||0), kb = counts.ben||0;
    if(okN < 30) html += `<p class="warn">Too few flows (${okN} correct before the attack) for a reliable rate.</p>`;
    const [lo,hi] = wilson(k, okN), [lo2,hi2] = wilson(kb, okN);
    html += `<p><b>Flip rate:</b> ${k} of ${okN} = ${pct(k/okN)} (95% CI ${pct(lo)}&ndash;${pct(hi)})<br><b>Evasion (flip to Benign):</b> ${kb} of ${okN} = ${pct(kb/okN)} (${pct(lo2)}&ndash;${pct(hi2)})</p><p class="mut">Intervals assume independent flows, so they understate the true uncertainty.</p>`;
  }
  document.getElementById("outcome").innerHTML = html;
}
function renderTrans(){
  const idx = selection(), e = String(state.eps), M = [0,1,2,3].map(()=>[0,0,0,0]);
  idx.forEach(i => { M[DATA.p0[i]][DATA.adv[e].p[i]]++; });
  const mx = Math.max(1, ...M.flat()), x0=92, y0=26, cw=54, ch=40; let s="";
  DATA.names.forEach((n,j)=>{ s+=svgEl("text",{x:x0+j*cw+cw/2,y:y0-8,"text-anchor":"middle"},n.split("-")[0]); s+=svgEl("text",{x:x0-6,y:y0+j*ch+ch/2+3,"text-anchor":"end"},n.split("-")[0]); });
  for(let i=0;i<4;i++)for(let j=0;j<4;j++){ const v=M[i][j], a=v/mx;
    s+=svgEl("rect",{x:x0+j*cw,y:y0+i*ch,width:cw-2,height:ch-2,fill:`rgba(31,119,180,${0.08+0.85*a})`,stroke:"#8b949e","stroke-width":i===j?1.5:0.3});
    s+=svgEl("text",{x:x0+j*cw+cw/2-1,y:y0+i*ch+ch/2+3,"text-anchor":"middle",style:"font-size:11px;"+(a>0.55?"fill:#fff":"")},v); }
  s+=svgEl("text",{x:x0+2*cw,y:y0+4*ch+16,"text-anchor":"middle"},"prediction after the attack");
  s+=svgEl("text",{x:12,y:y0+2*ch,"text-anchor":"middle",transform:`rotate(-90 12 ${y0+2*ch})`},"prediction before");
  document.getElementById("trans").innerHTML = s;
}
function renderScatter(){
  const idx = selection(), e = String(state.eps), L=38,T=10,W=290,H=250; let s="";
  s+=svgEl("rect",{x:L,y:T,width:W,height:H,fill:"none",stroke:"#8b949e"});
  for(let t=0;t<=1.001;t+=0.25){ s+=svgEl("text",{x:L-4,y:T+H-t*H+3,"text-anchor":"end"},t.toFixed(2)); s+=svgEl("text",{x:L+t*W,y:T+H+12,"text-anchor":"middle"},t.toFixed(2)); }
  s+=svgEl("line",{x1:L,y1:T+H,x2:L+W,y2:T,stroke:"#8b949e","stroke-dasharray":"3 3"});
  const ty = T+H-state.tau*H, tx = L+state.tau*W;
  s+=svgEl("line",{x1:L,y1:ty,x2:L+W,y2:ty,stroke:"#d29922","stroke-dasharray":"4 3"});
  s+=svgEl("line",{x1:tx,y1:T,x2:tx,y2:T+H,stroke:"#d29922","stroke-dasharray":"4 3"});
  idx.forEach(i => { const o=outcome(i,e), c0=DATA.c0[i], c1=DATA.adv[e].c[i];
    s+=svgEl("circle",{cx:(L+c0*W).toFixed(1),cy:(T+H-c1*H).toFixed(1),r:o==="ben"||o==="oth"?3.6:2.6,fill:OUT[o][1],"fill-opacity":0.75},svgEl("title",{},`flow ${i}: ${DATA.classes[DATA.cls[i]]}; before ${c0.toFixed(2)}, after ${c1.toFixed(2)}; ${OUT[o][0]}`)); });
  s+=svgEl("text",{x:L+W/2,y:T+H+26,"text-anchor":"middle"},"confidence before the attack");
  s+=svgEl("text",{x:9,y:T+H/2,"text-anchor":"middle",transform:`rotate(-90 9 ${T+H/2})`},"confidence after");
  document.getElementById("scatter").innerHTML = s;
}
function renderAbstain(){
  const idx = selection(), e = String(state.eps), tau = state.tau;
  if(state.cls===3){ document.getElementById("abstain").innerHTML = `<p class="mut">Not defined for the unseen class (no correct clean label).</p>`; return; }
  const flipped = idx.filter(i => ["ben","oth"].includes(outcome(i,e))), okc = idx.filter(i => DATA.p0[i]===trueIdx(i));
  const caught = flipped.filter(i => DATA.adv[e].c[i] < tau).length, rej = okc.filter(i => DATA.c0[i] < tau).length;
  const acc = idx.filter(i => DATA.adv[e].c[i] >= tau), accOk = acc.filter(i => DATA.adv[e].p[i]===trueIdx(i)).length;
  const bar = (v, col) => `<div style="background:var(--bd);height:8px;border-radius:4px"><div style="width:${(100*v).toFixed(0)}%;background:${col};height:8px;border-radius:4px"></div></div>`;
  let h = `<p>Rejecting every flow with confidence below <b>${tau.toFixed(2)}</b>:</p>`;
  h += `<p>Flipped flows caught: <b>${caught} of ${flipped.length}</b> (${pct(flipped.length? caught/flipped.length : NaN)})</p>${bar(flipped.length? caught/flipped.length:0,"#2da44e")}`;
  h += `<p>Clean-correct flows wrongly rejected: <b>${rej} of ${okc.length}</b> (${pct(okc.length? rej/okc.length : NaN)})</p>${bar(okc.length? rej/okc.length:0,"#d1242f")}`;
  h += `<p>Accepted flows classified correctly (under attack): <b>${accOk} of ${acc.length}</b> (${pct(acc.length? accOk/acc.length : NaN)})</p>`;
  if(flipped.length < 10) h += `<p class="warn">Only ${flipped.length} flipped flows here, so the catch rate is unreliable.</p>`;
  document.getElementById("abstain").innerHTML = h;
}
function renderCapacity(){
  const idx = selection().filter(i => isKnown(i) && DATA.p0[i]===trueIdx(i)), e = String(state.eps);
  const el = document.getElementById("capacity");
  if(state.cls===3 || idx.length < 30){ el.innerHTML = `<p class="warn">Too few flows (${idx.length} correct before the attack) for terciles.</p>`; return; }
  const caps = idx.map(i => DATA.cap[i]).sort((a,b)=>a-b), q1=caps[Math.floor(caps.length/3)], q2=caps[Math.floor(2*caps.length/3)];
  const bins = [[0,0,"low"],[0,0,"medium"],[0,0,"high"]];
  idx.forEach(i => { const c=DATA.cap[i], b = c<=q1?0:(c<=q2?1:2); bins[b][1]++; if(["ben","oth"].includes(outcome(i,e))) bins[b][0]++; });
  let h = `<table><tr><th>Capacity (bytes changed at the smallest budget)</th><th class="n">Flows</th><th class="n">Flips</th><th class="n">Rate (95% CI)</th></tr>`;
  const ranges = [`up to ${q1}`, `${q1} to ${q2}`, `above ${q2}`];
  bins.forEach((b,i)=>{ const [lo,hi]=wilson(b[0],b[1]); h += `<tr><td>${bins[i][2]} (${ranges[i]})</td><td class="n">${b[1]}</td><td class="n">${b[0]}</td><td class="n">${pct(b[0]/b[1])} (${pct(lo)}&ndash;${pct(hi)})</td></tr>`; });
  el.innerHTML = h + `</table><p class="mut">Capacity is a proxy: the number of payload bytes the attack changed at the smallest budget.</p>`;
}
function renderAll(){ renderOutcome(); renderTrans(); renderScatter(); renderAbstain(); renderCapacity();
  document.getElementById("tauv").textContent = state.tau.toFixed(2); }
function init(){
  const se = document.getElementById("eps"), sc = document.getElementById("cls"), st = document.getElementById("tau");
  DATA.eps.forEach(e => se.add(new Option(e, e))); se.value = state.eps;
  [["known",CLS_LABEL.known],[0,CLS_LABEL[0]],[1,CLS_LABEL[1]],[2,CLS_LABEL[2]],[3,CLS_LABEL[3]]].forEach(([v,t]) => sc.add(new Option(t, v)));
  se.onchange = () => { state.eps = +se.value; renderAll(); };
  sc.onchange = () => { state.cls = sc.value==="known" ? "known" : +sc.value; renderAll(); };
  st.oninput = () => { state.tau = +st.value; renderAll(); };
  document.getElementById("foot").innerHTML = "Source: adversarial_per_sample file from the official run. Flip rate = flows correct before the attack whose prediction changed. Wilson 95% intervals assume independent flows. Classes with fewer than 30 flows are not given a rate.";
  renderBanner(); renderAll();
}
window.__dash = {state, renderAll, wilson, outcome, selection, DATA};
init();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/metrics/adversarial_per_sample_OFFICIAL.npz")
    ap.add_argument("--json", default="results/metrics/adversarial_results_OFFICIAL.json")
    ap.add_argument("--out", default="results/dav/dashboard.html")
    args = ap.parse_args()
    payload = build_payload(args.npz, args.json)
    html = HTML.replace("/*DATA*/", json.dumps(payload, separators=(",", ":")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size/1e3:.0f} kB). Open it in a browser; no internet needed.")
    print(f"flows: {len(payload['cls'])} | budgets: {payload['eps']}")


if __name__ == "__main__":
    main()
