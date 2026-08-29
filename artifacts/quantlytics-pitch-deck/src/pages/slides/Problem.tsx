export default function Problem() {
  return (
    <div className="w-screen h-screen overflow-hidden relative bg-bg text-text font-body">
      <div className="absolute inset-0 deck-grid opacity-40" />
      <div className="absolute -top-[20vh] -left-[10vw] w-[45vw] h-[45vw] deck-glow-blue opacity-40" />

      <div className="absolute top-[7vh] left-[7vw] right-[7vw] flex items-center justify-between">
        <span className="font-mono text-[1.4vw] tracking-[0.35em] text-muted">
          QUANTLYTICS
        </span>
        <span className="font-mono text-[1.4vw] tracking-[0.35em] text-muted">
          02 / 08
        </span>
      </div>
      <div className="absolute top-[11vh] left-[7vw] right-[7vw] h-px bg-line" />

      <div className="absolute top-[16vh] left-[7vw] w-[52vw]">
        <div className="h-[0.5vh] w-[8vw] rule-gradient" />
        <h2 className="mt-[2.5vh] font-display text-[4.2vw] font-extrabold tracking-tighter leading-[1]">
          The problem
        </h2>
      </div>
      <p className="absolute top-[19vh] right-[7vw] w-[30vw] font-body text-[1.9vw] leading-snug text-muted text-pretty">
        Crypto desks lose money to speed, fragmentation and blind spots — not to
        bad ideas.
      </p>

      <div className="absolute top-[45vh] left-[7vw] right-[7vw] grid grid-cols-2 gap-x-[5vw] gap-y-[6vh]">
        <div className="border-t border-line pt-[2.5vh]">
          <span className="font-mono text-[1.5vw] text-primary">01</span>
          <h3 className="mt-[1vh] font-display text-[2.3vw] font-bold tracking-tight">
            Markets never close
          </h3>
          <p className="mt-[1vh] font-body text-[1.9vw] leading-snug text-muted text-pretty">
            Setups form and decay while the desk is asleep.
          </p>
        </div>
        <div className="border-t border-line pt-[2.5vh]">
          <span className="font-mono text-[1.5vw] text-primary">02</span>
          <h3 className="mt-[1vh] font-display text-[2.3vw] font-bold tracking-tight">
            Liquidity is fragmented
          </h3>
          <p className="mt-[1vh] font-body text-[1.9vw] leading-snug text-muted text-pretty">
            Every chain and L2 keeps its own book and its own spread.
          </p>
        </div>
        <div className="border-t border-line pt-[2.5vh]">
          <span className="font-mono text-[1.5vw] text-primary">03</span>
          <h3 className="mt-[1vh] font-display text-[2.3vw] font-bold tracking-tight">
            Signals arrive as noise
          </h3>
          <p className="mt-[1vh] font-body text-[1.9vw] leading-snug text-muted text-pretty">
            Price, technicals and sentiment sit in separate tabs.
          </p>
        </div>
        <div className="border-t border-line pt-[2.5vh]">
          <span className="font-mono text-[1.5vw] text-primary">04</span>
          <h3 className="mt-[1vh] font-display text-[2.3vw] font-bold tracking-tight">
            Risk is reviewed late
          </h3>
          <p className="mt-[1vh] font-body text-[1.9vw] leading-snug text-muted text-pretty">
            Exposure gets checked nightly instead of per trade.
          </p>
        </div>
      </div>
    </div>
  );
}
