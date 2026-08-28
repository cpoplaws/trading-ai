import { useMemo, useState } from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Bell,
  ChevronRight,
  CircleHelp,
  Crosshair,
  Gauge,
  Menu,
  Pause,
  Play,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  Wallet,
  X,
} from "lucide-react";

type Position = { ticker: string; name: string; allocation: number; value: string; move: string; positive: boolean; color: string };

const initialPositions: Position[] = [
  { ticker: "NVDA", name: "NVIDIA Corp.", allocation: 31, value: "$12,460.80", move: "+4.82%", positive: true, color: "#8c7bff" },
  { ticker: "BTC", name: "Bitcoin / USD", allocation: 24, value: "$9,644.20", move: "+2.16%", positive: true, color: "#efb36c" },
  { ticker: "MSFT", name: "Microsoft Corp.", allocation: 18, value: "$7,235.44", move: "-0.38%", positive: false, color: "#69c7c0" },
];

export default function OrbitCommand() {
  const [activeTab, setActiveTab] = useState("Command");
  const [running, setRunning] = useState(true);
  const [positions, setPositions] = useState(initialPositions);
  const [notice, setNotice] = useState<string | null>(null);
  const total = useMemo(() => positions.reduce((sum, item) => sum + item.allocation, 0), [positions]);

  const flash = (message: string) => {
    setNotice(message);
    window.setTimeout(() => setNotice(null), 2200);
  };

  return (
    <div className="orbit-shell">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
        .orbit-shell { min-height:100%; background:#0a0e18; color:#edf0f5; font-family:'Plus Jakarta Sans',sans-serif; padding:22px 18px 30px; position:relative; overflow:hidden; }
        .orbit-shell:before { content:""; position:absolute; inset:0; pointer-events:none; opacity:.16; background-image:radial-gradient(#a5a7b2 0.5px,transparent .5px); background-size:12px 12px; mask-image:linear-gradient(to bottom,black,transparent 80%); }
        .orbit-content { position:relative; max-width:440px; margin:auto; }
        .mono { font-family:'DM Mono',monospace; }
        .topline { display:flex; justify-content:space-between; align-items:center; margin-bottom:26px; }
        .brand { display:flex; gap:10px; align-items:center; font-size:12px; letter-spacing:.16em; font-weight:800; color:#bdc6db; }
        .brand-mark { width:26px;height:26px;border:1px solid #7667d5;border-radius:8px;display:grid;place-items:center;color:#a696ff;background:#17152c; }
        .icon-ghost { border:1px solid #252d3e; background:#111725; border-radius:11px; color:#aab4c8; padding:9px; cursor:pointer; }
        .greeting { font-size:27px; line-height:1.12; letter-spacing:-.06em; font-weight:800; margin:0; }
        .subtle { color:#7d879b; font-size:11px; line-height:1.55; }
        .mode-card { margin-top:20px; border:1px solid #28334a; border-radius:18px; padding:15px; background:linear-gradient(135deg,#151b2a,#101522); box-shadow:0 18px 45px #05070d80; }
        .mode-row { display:flex; align-items:center; justify-content:space-between; }
        .mode-label { display:flex; gap:10px; align-items:center; }
        .pulse { width:9px;height:9px;border-radius:50%;background:#62d0ac;box-shadow:0 0 0 5px #62d0ac1c; }
        .mode-title { font-size:12px; font-weight:700; }
        .mode-value { font-size:10px;color:#62d0ac;letter-spacing:.08em;margin-top:3px; }
        .toggle { border:0;border-radius:20px;width:48px;height:26px;background:#5c4fd1;padding:3px;cursor:pointer; }
        .toggle span { display:block;width:20px;height:20px;background:#fff;border-radius:50%;transform:translateX(22px);transition:transform .2s; }
        .toggle.off { background:#303a4d; }.toggle.off span { transform:translateX(0); }
        .focus-title { margin:26px 0 12px;display:flex;align-items:center;justify-content:space-between; }
        .kicker { color:#707b90;font-size:10px;letter-spacing:.16em;font-weight:800; }
        .focus-title button { color:#9e91ff;background:transparent;border:0;font-size:11px;cursor:pointer; }
        .signal { border:1px solid #3b356e;background:#17162b;border-radius:16px;padding:16px;display:flex;gap:13px;align-items:flex-start; }
        .signal-icon { width:34px;height:34px;display:grid;place-items:center;border-radius:11px;background:#31265d;color:#afa3ff;flex:none; }
        .signal h2 { font-size:13px;margin:0 0 5px; }.signal p { margin:0;color:#9097ad;font-size:11px;line-height:1.5; }
        .signal b { color:#c9c2ff; font-weight:700; }
        .strip { margin:13px 0 22px;display:flex;gap:9px;overflow:hidden; }
        .strip-card { min-width:112px;border-radius:13px;padding:11px;background:#111725;border:1px solid #202a3b; }
        .strip-card span { display:block;color:#748198;font-size:9px;letter-spacing:.08em; }.strip-card strong{font-size:14px;display:block;margin:6px 0 2px}.up{color:#63d2a7}.down{color:#e87d8d}
        .allocation { background:#111725;border:1px solid #202a3b;border-radius:16px;padding:15px;margin-bottom:20px; }
        .alloc-head { display:flex;justify-content:space-between;align-items:baseline;margin-bottom:14px; }.alloc-head strong{font-size:13px}.alloc-head span{color:#7a859a;font-size:10px}
        .bar { height:8px; background:#242d40;border-radius:9px;display:flex;overflow:hidden;gap:2px }.bar i{height:100%;display:block}.alloc-foot{display:flex;justify-content:space-between;margin-top:11px;color:#778297;font-size:10px}.alloc-foot b{color:#dce2ec;font-weight:600}
        .queue-head { display:flex;justify-content:space-between;align-items:center;margin-bottom:11px; }.queue-count{background:#302a61;color:#bcb1ff;border-radius:10px;padding:4px 7px;font-size:10px;font-weight:700}
        .position { display:flex;align-items:center;gap:11px;padding:13px 0;border-bottom:1px solid #1d2635; }.position:last-child{border-bottom:0}.coin{width:34px;height:34px;border-radius:11px;display:grid;place-items:center;font-size:10px;font-weight:800;color:#0b0e16}.pos-copy{flex:1}.pos-copy strong{display:block;font-size:12px}.pos-copy span{display:block;color:#707b8f;font-size:10px;margin-top:3px}.pos-value{text-align:right}.pos-value strong{display:block;font-family:'DM Mono';font-size:11px}.pos-value span{display:block;font-size:10px;margin-top:3px}
        .bottom-nav { display:flex;justify-content:space-around;gap:3px;border-top:1px solid #202a39;margin:20px -18px -8px;padding:15px 0 0; }.nav-item{border:0;background:none;color:#68748b;font-size:9px;display:grid;gap:5px;justify-items:center;cursor:pointer}.nav-item.active{color:#a99eff}.nav-item svg{width:17px;height:17px}.toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:#e9ecf6;color:#111521;padding:10px 15px;border-radius:12px;font-size:11px;font-weight:700;z-index:4;box-shadow:0 8px 25px #0008;white-space:nowrap}
      `}</style>
      <div className="orbit-content">
        <div className="topline">
          <div className="brand"><span className="brand-mark"><Crosshair size={14} /></span> ORBIT / AI</div>
          <button className="icon-ghost" onClick={() => flash("No new alerts")} aria-label="Notifications"><Bell size={16} /></button>
        </div>
        <p className="kicker">MONDAY · 09:41 UTC</p>
        <h1 className="greeting">Your portfolio<br /><span style={{ color: "#9f94f4" }}>has a pulse.</span></h1>
        <p className="subtle" style={{ marginTop: 10 }}>One screen for the decisions that matter right now.</p>

        <div className="mode-card">
          <div className="mode-row">
            <div className="mode-label"><span className="pulse" /><div><div className="mode-title">Autopilot is {running ? "active" : "paused"}</div><div className="mode-value mono">{running ? "SCANNING 4 MARKETS" : "MANUAL REVIEW"}</div></div></div>
            <button className={`toggle ${running ? "" : "off"}`} onClick={() => { setRunning(!running); flash(running ? "Autopilot paused" : "Autopilot resumed"); }} aria-label="Toggle autopilot"><span /></button>
          </div>
        </div>

        <div className="focus-title"><span className="kicker">THE FOCUS</span><button onClick={() => flash("Focus recalibrated")}>Recalibrate <ChevronRight size={12} style={{ verticalAlign: "middle" }} /></button></div>
        <div className="signal">
          <div className="signal-icon"><Sparkles size={17} /></div>
          <div><h2>Trim NVDA exposure</h2><p>Momentum cooled for 3 sessions. <b>Risk agent recommends −4%.</b></p></div>
          <button className="icon-ghost" onClick={() => { setPositions(positions.filter((p) => p.ticker !== "NVDA")); flash("NVDA moved to review"); }} aria-label="Dismiss signal"><X size={14} /></button>
        </div>
        <div className="strip">
          <div className="strip-card"><span>PORTFOLIO</span><strong>$40,188.52</strong><em className="up" style={{ fontSize: 10 }}>+3.24% today</em></div>
          <div className="strip-card"><span>BUYING POWER</span><strong>$18,403</strong><em style={{ color: "#8e9bb0", fontSize: 10 }}>61.4% liquid</em></div>
          <div className="strip-card"><span>SHARPE</span><strong>1.84</strong><em className="up" style={{ fontSize: 10 }}>+0.12 this week</em></div>
        </div>

        <div className="allocation">
          <div className="alloc-head"><strong>Capital map</strong><span>{total}% allocated · 2% reserve</span></div>
          <div className="bar">{positions.map((p) => <i key={p.ticker} style={{ width: `${p.allocation}%`, background: p.color }} />)}<i style={{ width: `${100 - total}%`, background: "#273044" }} /></div>
          <div className="alloc-foot"><span><b>Equities</b> 49%</span><span><b>Crypto</b> 24%</span><span><b>Cash</b> 27%</span></div>
        </div>

        <div className="queue-head"><span className="kicker">LIVE POSITIONS</span><span className="queue-count">{positions.length} tracked</span></div>
        <div>{positions.map((p) => <div className="position" key={p.ticker}><div className="coin" style={{ background: p.color }}>{p.ticker.slice(0, 2)}</div><div className="pos-copy"><strong>{p.ticker}</strong><span>{p.name} · {p.allocation}% weight</span></div><div className="pos-value"><strong>{p.value}</strong><span className={p.positive ? "up" : "down"}>{p.positive ? <ArrowUpRight size={11} /> : <ArrowDownRight size={11} />} {p.move}</span></div></div>)}</div>

        <div className="bottom-nav">{[["Command", Gauge], ["Strategies", TrendingUp], ["Agents", Activity], ["Wallet", Wallet]].map(([label, Icon]) => <button key={label as string} className={`nav-item ${activeTab === label ? "active" : ""}`} onClick={() => { setActiveTab(label as string); flash(`${label} view selected`); }}><Icon /><span>{label as string}</span></button>)}</div>
      </div>
      {notice && <div className="toast">{notice}</div>}
    </div>
  );
}