export default function MarketIntelligence() {
  return (
    <div className="w-screen h-screen overflow-hidden relative bg-bg text-text font-body">
      <div className="absolute inset-0 deck-grid opacity-40" />
      <div className="absolute -bottom-[15vh] left-[10vw] w-[45vw] h-[45vw] deck-glow-blue opacity-35" />

      <div className="absolute top-[7vh] left-[7vw] right-[7vw] flex items-center justify-between">
        <span className="font-mono text-[1.4vw] tracking-[0.35em] text-muted">
          QUANTLYTICS
        </span>
        <span className="font-mono text-[1.4vw] tracking-[0.35em] text-muted">
          06 / 08
        </span>
      </div>
      <div className="absolute top-[11vh] left-[7vw] right-[7vw] h-px bg-line" />

      <div className="absolute top-[17vh] left-[7vw] w-[48vw]">
        <div className="h-[0.5vh] w-[8vw] rule-gradient" />
        <h2 className="mt-[2.5vh] font-display text-[4vw] font-extrabold tracking-tighter leading-[1.02]">
          Market intelligence
        </h2>
        <p className="mt-[2.5vh] font-body text-[1.9vw] leading-snug text-muted text-pretty">
          Price action, technicals and sentiment collapse into a single
          composite score, refreshed every thirty seconds.
        </p>
      </div>

      <div className="absolute top-[47vh] left-[7vw] w-[45vw]">
        <span className="font-mono text-[1.5vw] tracking-[0.3em] text-muted">
          COMPOSITE SIGNAL
        </span>
        <div className="mt-[1vh] flex items-baseline gap-[2vw]">
          <span className="font-display text-[10vw] font-black tracking-tighter leading-[0.85] text-gradient">
            92%
          </span>
          <span className="font-display text-[2.6vw] font-bold tracking-tight text-primary">
            Strong buy
          </span>
        </div>
        <div className="mt-[3vh] h-[0.8vh] w-[38vw] bg-panel border border-line">
          <div className="h-full w-[92%] rule-gradient" />
        </div>
      </div>

      <div className="absolute top-[30vh] right-[7vw] w-[34vw]">
        <div className="flex items-baseline justify-between border-t border-line py-[2.6vh]">
          <span className="font-body text-[1.9vw] text-muted">Regime</span>
          <span className="font-mono text-[2.2vw] font-bold">Bull trend</span>
        </div>
        <div className="flex items-baseline justify-between border-t border-line py-[2.6vh]">
          <span className="font-body text-[1.9vw] text-muted">Momentum</span>
          <span className="font-mono text-[2.2vw] font-bold text-pos">
            +8.42
          </span>
        </div>
        <div className="flex items-baseline justify-between border-t border-line py-[2.6vh]">
          <span className="font-body text-[1.9vw] text-muted">Volatility</span>
          <span className="font-mono text-[2.2vw] font-bold">Medium</span>
        </div>
        <div className="flex items-baseline justify-between border-t border-b border-line py-[2.6vh]">
          <span className="font-body text-[1.9vw] text-muted">Alerts</span>
          <span className="font-mono text-[2.2vw] font-bold text-accent">
            Live
          </span>
        </div>
      </div>

      <p className="absolute bottom-[6vh] left-[7vw] font-mono text-[1.5vw] text-muted">
        Reading captured in the demo environment.
      </p>
    </div>
  );
}
