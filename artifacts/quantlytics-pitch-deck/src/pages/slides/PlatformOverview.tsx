export default function PlatformOverview() {
  return (
    <div className="w-screen h-screen overflow-hidden relative bg-bg text-text font-body">
      <div className="absolute inset-0 deck-grid opacity-40" />
      <div className="absolute -top-[25vh] right-[5vw] w-[40vw] h-[40vw] deck-glow-violet opacity-40" />

      <div className="absolute top-[7vh] left-[7vw] right-[7vw] flex items-center justify-between">
        <span className="font-mono text-[1.4vw] tracking-[0.35em] text-muted">
          QUANTLYTICS
        </span>
        <span className="font-mono text-[1.4vw] tracking-[0.35em] text-muted">
          03 / 08
        </span>
      </div>
      <div className="absolute top-[11vh] left-[7vw] right-[7vw] h-px bg-line" />

      <div className="absolute top-[16vh] left-[7vw] w-[70vw]">
        <div className="h-[0.5vh] w-[8vw] rule-gradient" />
        <h2 className="mt-[2.5vh] font-display text-[4.2vw] font-extrabold tracking-tighter leading-[1]">
          Platform overview
        </h2>
        <p className="mt-[2vh] font-body text-[2vw] leading-snug text-muted text-pretty">
          One system reads the market, votes on every trade, executes across
          chains, and holds the risk line.
        </p>
      </div>

      <div className="absolute top-[45vh] left-[7vw] right-[7vw] grid grid-cols-3 gap-[2vw]">
        <div className="bg-panel border border-line p-[2.5vw]">
          <span className="font-mono text-[1.4vw] tracking-[0.3em] text-primary">
            READ
          </span>
          <h3 className="mt-[1.5vh] font-display text-[2.2vw] font-bold tracking-tight">
            Market intelligence
          </h3>
          <p className="mt-[1.5vh] font-body text-[1.8vw] leading-snug text-muted text-pretty">
            One composite score from regime, technicals and sentiment.
          </p>
        </div>
        <div className="bg-panel border border-line p-[2.5vw]">
          <span className="font-mono text-[1.4vw] tracking-[0.3em] text-primary">
            DECIDE
          </span>
          <h3 className="mt-[1.5vh] font-display text-[2.2vw] font-bold tracking-tight">
            AI agent swarm
          </h3>
          <p className="mt-[1.5vh] font-body text-[1.8vw] leading-snug text-muted text-pretty">
            Four specialists vote before an order is sent.
          </p>
        </div>
        <div className="bg-panel border border-line p-[2.5vw]">
          <span className="font-mono text-[1.4vw] tracking-[0.3em] text-primary">
            EXECUTE
          </span>
          <h3 className="mt-[1.5vh] font-display text-[2.2vw] font-bold tracking-tight">
            Multi-chain routing
          </h3>
          <p className="mt-[1.5vh] font-body text-[1.8vw] leading-snug text-muted text-pretty">
            Base, Solana and the major L2s behind one book.
          </p>
        </div>
      </div>

      <div className="absolute bottom-[7vh] left-[7vw] right-[7vw]">
        <div className="h-px bg-line" />
        <div className="mt-[2vh] flex items-end justify-between">
          <span className="font-mono text-[1.5vw] text-muted">
            Sharpe ratio <span className="text-text">1.84</span>
          </span>
          <span className="font-mono text-[1.5vw] text-muted">
            Win rate <span className="text-text">64.3%</span>
          </span>
          <span className="font-mono text-[1.5vw] text-muted">
            Open positions <span className="text-text">7</span>
          </span>
          <span className="font-mono text-[1.5vw] text-muted">
            Demo environment figures
          </span>
        </div>
      </div>
    </div>
  );
}
