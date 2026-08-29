export default function MultiChainExecution() {
  return (
    <div className="w-screen h-screen overflow-hidden relative bg-bg text-text font-body">
      <div className="absolute inset-0 deck-grid opacity-40" />
      <div className="absolute bottom-[5vh] -left-[8vw] w-[38vw] h-[38vw] deck-glow-violet opacity-35" />

      <div className="absolute top-[7vh] left-[7vw] right-[7vw] flex items-center justify-between">
        <span className="font-mono text-[1.4vw] tracking-[0.35em] text-muted">
          QUANTLYTICS
        </span>
        <span className="font-mono text-[1.4vw] tracking-[0.35em] text-muted">
          05 / 08
        </span>
      </div>
      <div className="absolute top-[11vh] left-[7vw] right-[7vw] h-px bg-line" />

      <div className="absolute top-[20vh] left-[7vw] w-[36vw]">
        <div className="h-[0.5vh] w-[8vw] rule-gradient" />
        <h2 className="mt-[2.5vh] font-display text-[4vw] font-extrabold tracking-tighter leading-[1.02]">
          Multi-chain execution
        </h2>
        <p className="mt-[2.5vh] font-body text-[1.9vw] leading-snug text-muted text-pretty">
          Strategies route to whichever venue holds the fill. Positions, P&amp;L
          and risk stay in one book.
        </p>
        <div className="mt-[4vh] grid grid-cols-2 gap-[1vw]">
          <span className="border border-line px-[1.2vw] py-[1.2vh] font-mono text-[1.6vw] text-text">
            BASE
          </span>
          <span className="border border-line px-[1.2vw] py-[1.2vh] font-mono text-[1.6vw] text-text">
            SOLANA
          </span>
          <span className="border border-line px-[1.2vw] py-[1.2vh] font-mono text-[1.6vw] text-text">
            ARBITRUM
          </span>
          <span className="border border-line px-[1.2vw] py-[1.2vh] font-mono text-[1.6vw] text-text">
            OPTIMISM
          </span>
        </div>
      </div>

      <div className="absolute top-[20vh] right-[7vw] w-[43vw] bg-panel border border-line">
        <div className="flex items-center justify-between border-b border-line px-[2vw] py-[2vh]">
          <span className="font-mono text-[1.5vw] tracking-[0.3em] text-muted">
            RECENT FILLS
          </span>
          <span className="font-mono text-[1.5vw] text-primary">LIVE</span>
        </div>
        <div className="flex items-baseline justify-between px-[2vw] py-[2.8vh] border-b border-line">
          <span className="font-mono text-[1.8vw] w-[11vw]">ETH/USDC</span>
          <span className="font-mono text-[1.5vw] text-primary w-[5vw]">
            BUY
          </span>
          <span className="font-body text-[1.6vw] text-muted w-[13vw]">
            Momentum
          </span>
          <span className="font-mono text-[1.7vw] text-pos w-[7vw] text-right">
            +184.32
          </span>
        </div>
        <div className="flex items-baseline justify-between px-[2vw] py-[2.8vh] border-b border-line">
          <span className="font-mono text-[1.8vw] w-[11vw]">SOL/USDC</span>
          <span className="font-mono text-[1.5vw] text-accent w-[5vw]">
            SELL
          </span>
          <span className="font-body text-[1.6vw] text-muted w-[13vw]">
            Mean reversion
          </span>
          <span className="font-mono text-[1.7vw] text-neg w-[7vw] text-right">
            -62.15
          </span>
        </div>
        <div className="flex items-baseline justify-between px-[2vw] py-[2.8vh] border-b border-line">
          <span className="font-mono text-[1.8vw] w-[11vw]">WBTC/USDC</span>
          <span className="font-mono text-[1.5vw] text-primary w-[5vw]">
            BUY
          </span>
          <span className="font-body text-[1.6vw] text-muted w-[13vw]">
            Trend following
          </span>
          <span className="font-mono text-[1.7vw] text-pos w-[7vw] text-right">
            +512.70
          </span>
        </div>
        <div className="flex items-baseline justify-between px-[2vw] py-[2.8vh] border-b border-line">
          <span className="font-mono text-[1.8vw] w-[11vw]">ARB/USDC</span>
          <span className="font-mono text-[1.5vw] text-accent w-[5vw]">
            SELL
          </span>
          <span className="font-body text-[1.6vw] text-muted w-[13vw]">
            Arbitrage
          </span>
          <span className="font-mono text-[1.7vw] text-pos w-[7vw] text-right">
            +97.44
          </span>
        </div>
        <div className="flex items-baseline justify-between px-[2vw] py-[2.8vh]">
          <span className="font-mono text-[1.8vw] w-[11vw]">BASE/USDC</span>
          <span className="font-mono text-[1.5vw] text-primary w-[5vw]">
            BUY
          </span>
          <span className="font-body text-[1.6vw] text-muted w-[13vw]">
            Market making
          </span>
          <span className="font-mono text-[1.7vw] text-pos w-[7vw] text-right">
            +43.06
          </span>
        </div>
      </div>

      <p className="absolute bottom-[6vh] right-[7vw] font-mono text-[1.5vw] text-muted">
        Fills captured in the demo environment.
      </p>
    </div>
  );
}
