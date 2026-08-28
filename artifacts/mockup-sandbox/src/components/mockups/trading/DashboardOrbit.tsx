import { useState, type ReactNode } from "react";
import "./dashboard-orbit.css";

type IconName = "grid" | "pulse" | "layers" | "bot" | "clock" | "settings" | "bell" | "chevron";

function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, ReactNode> = {
    grid: <><rect x="3" y="3" width="6" height="6" rx="1" /><rect x="15" y="3" width="6" height="6" rx="1" /><rect x="3" y="15" width="6" height="6" rx="1" /><rect x="15" y="15" width="6" height="6" rx="1" /></>,
    pulse: <><path d="M3 12h4l2.2-7 4.2 14 2.2-7H21" /></>,
    layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 12 9 5 9-5" /><path d="m3 16 9 5 9-5" /></>,
    bot: <><rect x="4" y="7" width="16" height="13" rx="3" /><path d="M12 3v4M8 13h.01M16 13h.01M8 17h8" /></>,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    settings: <><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .34 1.88l.06.06-1.7 1.7-.06-.06a1.7 1.7 0 0 0-1.88-.34 1.7 1.7 0 0 0-1.03 1.56V20h-2.4v-.2a1.7 1.7 0 0 0-1.03-1.56 1.7 1.7 0 0 0-1.88.34l-.06.06-1.7-1.7.06-.06A1.7 1.7 0 0 0 8.46 15 1.7 1.7 0 0 0 6.9 14H6.7v-2.4h.2a1.7 1.7 0 0 0 1.56-1.03 1.7 1.7 0 0 0-.34-1.88l-.06-.06 1.7-1.7.06.06a1.7 1.7 0 0 0 1.88.34A1.7 1.7 0 0 0 12.73 5v-.2h2.4V5a1.7 1.7 0 0 0 1.03 1.56 1.7 1.7 0 0 0 1.88-.34l.06-.06 1.7 1.7-.06.06A1.7 1.7 0 0 0 19.4 9c.2.63.8 1.03 1.56 1.03h.2v2.4h-.2c-.76 0-1.36.4-1.56 1.03" /></>,
    bell: <><path d="M18 9a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4" /></>,
    chevron: <path d="m8 10 4 4 4-4" />,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">{paths[name]}</svg>;
}

const watch = [
  { symbol: "BTC / USD", value: "$67,284.12", change: "+2.84%", color: "#86e3c3" },
  { symbol: "ETH / USD", value: "$3,492.70", change: "+1.21%", color: "#86e3c3" },
  { symbol: "SOL / USD", value: "$148.06", change: "-0.62%", color: "#e78b83" },
  { symbol: "AVAX / USD", value: "$37.91", change: "+4.08%", color: "#86e3c3" },
];

const agents = [
  { name: "Execution", role: "timing + sizing", score: "94.8%", state: "RUNNING", tint: "#86e3c3" },
  { name: "Risk guard", role: "exposure limits", score: "99.1%", state: "WATCHING", tint: "#e1bc74" },
  { name: "Arbitrage", role: "venue spread", score: "88.6%", state: "RUNNING", tint: "#9bb9e8" },
];

export function DashboardOrbit() {
  const [activeRail, setActiveRail] = useState("pulse");
  const [orderSide, setOrderSide] = useState<"BUY" | "SELL">("BUY");
  const [orderType, setOrderType] = useState("Market");
  const [strategyOn, setStrategyOn] = useState(true);
  const [notice, setNotice] = useState("");

  const submitOrder = () => {
    setNotice(`${orderSide} order staged · BTC / USD`);
    window.setTimeout(() => setNotice(""), 2600);
  };

  return (
    <main className="orbit-shell">
      <aside className="orbit-rail">
        <div className="orbit-mark">Q<span>·</span></div>
        <div className="rail-group">
          {(["grid", "pulse", "layers", "bot", "clock"] as IconName[]).map((item) => (
            <button key={item} aria-label={item} onClick={() => setActiveRail(item)} className={`rail-button ${activeRail === item ? "selected" : ""}`}>
              <Icon name={item} />
            </button>
          ))}
        </div>
        <div className="rail-group bottom">
          <button aria-label="notifications" onClick={() => setNotice("No new alerts")} className="rail-button"><Icon name="bell" /></button>
          <button aria-label="settings" onClick={() => setNotice("Settings are read-only in preview")} className="rail-button"><Icon name="settings" /></button>
          <div className="avatar">ML</div>
        </div>
      </aside>

      <section className="orbit-main">
        <header className="orbit-header">
          <div>
            <div className="eyebrow">QUANTLYTICS / LIVE DESK</div>
            <h1>Market command <em>center</em></h1>
          </div>
          <div className="header-right">
            <div className="market-status"><i /> MARKET OPEN <span>·</span> 09:42:18 UTC</div>
            <button className="header-icon" aria-label="notifications" onClick={() => setNotice("All systems nominal")}><Icon name="bell" /></button>
          </div>
        </header>

        <div className="dashboard-grid">
          <section className="panel hero-panel">
            <div className="panel-top">
              <div>
                <div className="label">TOTAL PORTFOLIO</div>
                <div className="balance">$284,691<span>.42</span></div>
              </div>
              <div className="gain"><strong>+$6,824.19</strong><small>+2.46% today</small></div>
            </div>
            <div className="chart-wrap">
              <svg viewBox="0 0 760 260" preserveAspectRatio="none" className="line-chart" role="img" aria-label="Portfolio performance chart">
                <defs><linearGradient id="area" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#86e3c3" stopOpacity=".21" /><stop offset="1" stopColor="#86e3c3" stopOpacity="0" /></linearGradient></defs>
                {[52, 104, 156, 208].map((y) => <line key={y} x1="0" x2="760" y1={y} y2={y} stroke="#263b4b" strokeDasharray="3 7" />)}
                <path d="M0 205 C35 198 45 166 82 179 S125 142 161 157 S207 126 246 146 S288 115 319 128 S368 91 401 114 S446 120 477 84 S519 111 548 85 S590 60 620 78 S668 42 704 56 S735 29 760 25 V260 H0Z" fill="url(#area)" />
                <path d="M0 205 C35 198 45 166 82 179 S125 142 161 157 S207 126 246 146 S288 115 319 128 S368 91 401 114 S446 120 477 84 S519 111 548 85 S590 60 620 78 S668 42 704 56 S735 29 760 25" fill="none" stroke="#86e3c3" strokeWidth="2.5" />
                <circle cx="760" cy="25" r="5" fill="#86e3c3" />
              </svg>
              <div className="chart-labels"><span>01 MAY</span><span>08 MAY</span><span>15 MAY</span><span>22 MAY</span><span>29 MAY</span></div>
            </div>
            <div className="metric-strip">
              <div><span>SHARPE RATIO</span><b>2.84</b></div><div><span>WIN RATE</span><b>67.3%</b></div><div><span>MAX DRAWDOWN</span><b className="negative">−4.18%</b></div><div><span>OPEN POSITIONS</span><b>12</b></div>
            </div>
          </section>

          <section className="panel order-panel">
            <div className="panel-title"><div><div className="label">QUICK EXECUTION</div><h2>Order ticket</h2></div><span className="live-chip">PAPER</span></div>
            <div className="asset-select"><span className="coin-dot">₿</span><div><b>BTC / USD</b><small>Bitcoin · Coinbase</small></div><Icon name="chevron" size={16} /></div>
            <div className="side-toggle"><button className={orderSide === "BUY" ? "active-buy" : ""} onClick={() => setOrderSide("BUY")}>BUY</button><button className={orderSide === "SELL" ? "active-sell" : ""} onClick={() => setOrderSide("SELL")}>SELL</button></div>
            <label className="field-label">ORDER TYPE <select value={orderType} onChange={(e) => setOrderType(e.target.value)}><option>Market</option><option>Limit</option><option>Stop loss</option></select></label>
            <label className="field-label">AMOUNT <div className="input-like"><span>$</span><input defaultValue="2,500" aria-label="amount" /><small>USD</small></div></label>
            <div className="order-summary"><span>Est. price</span><b>$67,284.12</b><span>Buying power</span><b>$84,210.06</b></div>
            <button className={`execute ${orderSide === "SELL" ? "sell" : ""}`} onClick={submitOrder}>{orderSide} BTC <span>⌘ ↵</span></button>
            {notice && <div className="toast-note">{notice}</div>}
          </section>

          <section className="panel watch-panel">
            <div className="panel-title"><div><div className="label">MARKET RADAR</div><h2>Watchlist</h2></div><button className="text-button" onClick={() => setNotice("Watchlist synced")}>SYNC <span>↗</span></button></div>
            <div className="watch-list">{watch.map((item) => <button key={item.symbol} className="watch-row" onClick={() => setNotice(`${item.symbol} selected for order ticket`)}><span className="watch-symbol"><i style={{ background: item.color }} />{item.symbol}</span><b>{item.value}</b><span style={{ color: item.color }}>{item.change}</span></button>)}</div>
          </section>

          <section className="panel strategy-panel">
            <div className="panel-title"><div><div className="label">AUTONOMOUS LAYER</div><h2>Strategy pulse</h2></div><button className={`switch ${strategyOn ? "on" : ""}`} aria-label="toggle strategy" onClick={() => setStrategyOn(!strategyOn)}><i /></button></div>
            <div className="strategy-focus"><div className="orbital"><span /><span /><span /><b>Q</b></div><div><b>Multi-chain momentum</b><small>{strategyOn ? "Running across 4 venues" : "Strategy paused"}</small></div></div>
            <div className="strategy-bars"><div><span>Signal confidence</span><b>82%</b><i><em style={{ width: "82%" }} /></i></div><div><span>Capital deployed</span><b>$48.2k</b><i><em style={{ width: "58%", background: "#e1bc74" }} /></i></div></div>
          </section>
        </div>

        <section className="lower-section">
          <div className="section-heading"><div><div className="eyebrow">AGENT NETWORK</div><h2>Autonomous operators</h2></div><button className="text-button" onClick={() => setNotice("Agent telemetry refreshed")}>VIEW TELEMETRY <span>↗</span></button></div>
          <div className="agent-grid">{agents.map((agent) => <button key={agent.name} className="agent-card" onClick={() => setNotice(`${agent.name} agent selected`)}><div className="agent-top"><span className="agent-icon"><Icon name="bot" /></span><i style={{ background: agent.tint }} /></div><b>{agent.name}</b><small>{agent.role}</small><div className="agent-bottom"><span>{agent.state}</span><strong style={{ color: agent.tint }}>{agent.score}</strong></div></button>)}</div>
        </section>
        <footer><span>QUANTLYTICS TERMINAL / BUILD 2.4.8</span><span><i /> ALL SYSTEMS NOMINAL</span></footer>
      </section>
    </main>
  );
}

export default DashboardOrbit;