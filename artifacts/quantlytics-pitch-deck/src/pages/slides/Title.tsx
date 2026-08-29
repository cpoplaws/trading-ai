const base = import.meta.env.BASE_URL;

export default function Title() {
  return (
    <div className="w-screen h-screen overflow-hidden relative bg-bg text-text font-body">
      <img
        src={`${base}hero-lattice.jpg`}
        crossOrigin="anonymous"
        alt="Abstract lattice of blue and violet light filaments on deep navy"
        className="absolute inset-0 w-full h-full object-cover"
      />
      <div className="absolute inset-0 bg-gradient-to-r from-bg via-bg/90 to-bg/25" />
      <div className="absolute inset-0 bg-gradient-to-t from-bg via-transparent to-bg/50" />
      <div className="absolute inset-0 deck-grid opacity-30" />

      <div className="absolute top-[7vh] left-[7vw] right-[7vw] flex items-center justify-between">
        <span className="font-mono text-[1.5vw] tracking-[0.4em] text-muted">
          QUANTLYTICS
        </span>
        <span className="font-mono text-[1.5vw] tracking-[0.4em] text-muted">
          2026
        </span>
      </div>

      <div className="absolute left-[7vw] top-[30vh] w-[62vw]">
        <p className="font-mono text-[1.5vw] tracking-[0.35em] text-primary">
          AI TRADING PLATFORM
        </p>
        <h1 className="mt-[2vh] font-display text-[8vw] font-black tracking-tighter leading-[0.9]">
          Quantlytics
        </h1>
        <div className="mt-[3vh] h-[0.5vh] w-[20vw] rule-gradient" />
        <p className="mt-[3vh] font-display text-[3vw] font-semibold tracking-tight text-gradient">
          The intelligent edge in crypto.
        </p>
        <p className="mt-[2vh] font-mono text-[1.6vw] text-muted">
          Multi-chain trading · Base · Solana · L2s
        </p>
      </div>

      <div className="absolute bottom-[7vh] left-[7vw] right-[7vw] flex items-end justify-between">
        <span className="font-mono text-[1.5vw] text-muted">
          Platform overview and live demo
        </span>
        <span className="font-mono text-[1.5vw] text-muted">01 / 08</span>
      </div>
    </div>
  );
}
