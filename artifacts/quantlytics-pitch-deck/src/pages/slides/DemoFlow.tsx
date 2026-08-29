export default function DemoFlow() {
  return (
    <div className="w-screen h-screen overflow-hidden relative bg-bg text-text font-body">
      <div className="absolute inset-0 deck-grid opacity-40" />
      <div className="absolute -top-[18vh] right-[8vw] w-[36vw] h-[36vw] deck-glow-violet opacity-35" />

      <div className="absolute top-[7vh] left-[7vw] right-[7vw] flex items-center justify-between">
        <span className="font-mono text-[1.4vw] tracking-[0.35em] text-muted">
          QUANTLYTICS
        </span>
        <span className="font-mono text-[1.4vw] tracking-[0.35em] text-muted">
          07 / 08
        </span>
      </div>
      <div className="absolute top-[11vh] left-[7vw] right-[7vw] h-px bg-line" />

      <div className="absolute top-[16vh] left-[7vw] w-[50vw]">
        <div className="h-[0.5vh] w-[8vw] rule-gradient" />
        <h2 className="mt-[2.5vh] font-display text-[4.2vw] font-extrabold tracking-tighter leading-[1]">
          Live demo flow
        </h2>
      </div>
      <p className="absolute top-[20vh] right-[7vw] w-[30vw] font-body text-[1.9vw] leading-snug text-muted text-right text-pretty">
        Five minutes, one dashboard, no slides.
      </p>

      <div className="absolute top-[37vh] left-[7vw] right-[7vw]">
        <div className="flex items-baseline gap-[3vw] border-t border-line py-[2.2vh]">
          <span className="font-mono text-[1.8vw] text-primary w-[4vw]">01</span>
          <span className="font-display text-[2.1vw] font-bold tracking-tight w-[24vw]">
            Open the dashboard
          </span>
          <span className="font-body text-[1.9vw] text-muted">
            Portfolio value, Sharpe and win rate load live.
          </span>
        </div>
        <div className="flex items-baseline gap-[3vw] border-t border-line py-[2.2vh]">
          <span className="font-mono text-[1.8vw] text-primary w-[4vw]">02</span>
          <span className="font-display text-[2.1vw] font-bold tracking-tight w-[24vw]">
            Read the signal
          </span>
          <span className="font-body text-[1.9vw] text-muted">
            Composite score and regime set the posture.
          </span>
        </div>
        <div className="flex items-baseline gap-[3vw] border-t border-line py-[2.2vh]">
          <span className="font-mono text-[1.8vw] text-primary w-[4vw]">03</span>
          <span className="font-display text-[2.1vw] font-bold tracking-tight w-[24vw]">
            Start the swarm
          </span>
          <span className="font-body text-[1.9vw] text-muted">
            Four agents begin voting on every candidate trade.
          </span>
        </div>
        <div className="flex items-baseline gap-[3vw] border-t border-line py-[2.2vh]">
          <span className="font-mono text-[1.8vw] text-primary w-[4vw]">04</span>
          <span className="font-display text-[2.1vw] font-bold tracking-tight w-[24vw]">
            Watch a fill land
          </span>
          <span className="font-body text-[1.9vw] text-muted">
            Trades stream in with strategy, price and P&amp;L.
          </span>
        </div>
        <div className="flex items-baseline gap-[3vw] border-t border-b border-line py-[2.2vh]">
          <span className="font-mono text-[1.8vw] text-primary w-[4vw]">05</span>
          <span className="font-display text-[2.1vw] font-bold tracking-tight w-[24vw]">
            Cut risk on the spot
          </span>
          <span className="font-body text-[1.9vw] text-muted">
            Disable a strategy or agent and the book rebalances.
          </span>
        </div>
      </div>
    </div>
  );
}
