export default function AgentSwarm() {
  return (
    <div className="w-screen h-screen overflow-hidden relative bg-bg text-text font-body">
      <div className="absolute inset-0 deck-grid opacity-40" />
      <div className="absolute top-[20vh] left-[35vw] w-[35vw] h-[35vw] deck-glow-blue opacity-30" />

      <div className="absolute top-[7vh] left-[7vw] right-[7vw] flex items-center justify-between">
        <span className="font-mono text-[1.4vw] tracking-[0.35em] text-muted">
          QUANTLYTICS
        </span>
        <span className="font-mono text-[1.4vw] tracking-[0.35em] text-muted">
          04 / 08
        </span>
      </div>
      <div className="absolute top-[11vh] left-[7vw] right-[7vw] h-px bg-line" />

      <div className="absolute top-[16vh] left-[7vw] w-[44vw]">
        <div className="h-[0.5vh] w-[8vw] rule-gradient" />
        <h2 className="mt-[2.5vh] font-display text-[4.2vw] font-extrabold tracking-tighter leading-[1]">
          AI agent swarm
        </h2>
      </div>
      <p className="absolute top-[20vh] right-[7vw] w-[38vw] font-mono text-[1.7vw] leading-snug text-primary text-right">
        Consensus voting across 4 specialists
      </p>

      <div className="absolute top-[36vh] left-[7vw] right-[7vw] grid grid-cols-2 gap-x-[2vw] gap-y-[2.5vh]">
        <div className="bg-panel border border-line px-[2.2vw] py-[2.2vh]">
          <h3 className="font-display text-[2.2vw] font-bold tracking-tight">
            Execution Agent
          </h3>
          <p className="mt-[0.8vh] font-body text-[1.8vw] leading-snug text-muted">
            Optimizes trade timing and sizing
          </p>
          <div className="mt-[1.6vh] flex items-baseline gap-[2.5vw]">
            <span className="font-mono text-[2.4vw] font-bold text-primary">
              70.2%
            </span>
            <span className="font-mono text-[1.5vw] text-muted">
              1,284 decisions
            </span>
          </div>
        </div>
        <div className="bg-panel border border-line px-[2.2vw] py-[2.2vh]">
          <h3 className="font-display text-[2.2vw] font-bold tracking-tight">
            Risk Agent
          </h3>
          <p className="mt-[0.8vh] font-body text-[1.8vw] leading-snug text-muted">
            Monitors portfolio risk and enforces limits
          </p>
          <div className="mt-[1.6vh] flex items-baseline gap-[2.5vw]">
            <span className="font-mono text-[2.4vw] font-bold text-primary">
              75.5%
            </span>
            <span className="font-mono text-[1.5vw] text-muted">
              964 decisions
            </span>
          </div>
        </div>
        <div className="bg-panel border border-line px-[2.2vw] py-[2.2vh]">
          <h3 className="font-display text-[2.2vw] font-bold tracking-tight">
            Arbitrage Agent
          </h3>
          <p className="mt-[0.8vh] font-body text-[1.8vw] leading-snug text-muted">
            Finds price discrepancies and arbitrage
          </p>
          <div className="mt-[1.6vh] flex items-baseline gap-[2.5vw]">
            <span className="font-mono text-[2.4vw] font-bold text-accent">
              56.4%
            </span>
            <span className="font-mono text-[1.5vw] text-muted">
              512 decisions
            </span>
          </div>
        </div>
        <div className="bg-panel border border-line px-[2.2vw] py-[2.2vh]">
          <h3 className="font-display text-[2.2vw] font-bold tracking-tight">
            Market Making Agent
          </h3>
          <p className="mt-[0.8vh] font-body text-[1.8vw] leading-snug text-muted">
            Provides liquidity and captures spread
          </p>
          <div className="mt-[1.6vh] flex items-baseline gap-[2.5vw]">
            <span className="font-mono text-[2.4vw] font-bold text-accent">
              61.0%
            </span>
            <span className="font-mono text-[1.5vw] text-muted">
              2,140 decisions
            </span>
          </div>
        </div>
      </div>

      <p className="absolute bottom-[6vh] left-[7vw] font-mono text-[1.5vw] text-muted">
        Accuracy and decision counts from the demo environment. Every agent can
        be switched off mid-session.
      </p>
    </div>
  );
}
