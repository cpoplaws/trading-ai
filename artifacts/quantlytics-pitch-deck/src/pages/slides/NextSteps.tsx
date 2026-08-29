const base = import.meta.env.BASE_URL;

export default function NextSteps() {
  return (
    <div className="w-screen h-screen overflow-hidden relative bg-bg text-text font-body">
      <img
        src={`${base}closing-horizon.jpg`}
        crossOrigin="anonymous"
        alt="Dark navy horizon of layered glass planes lit by a blue and violet glow"
        className="absolute inset-0 w-full h-full object-cover"
      />
      <div className="absolute inset-0 bg-gradient-to-r from-bg via-bg/80 to-bg/15" />
      <div className="absolute inset-0 bg-gradient-to-t from-bg via-transparent to-bg/50" />
      <div className="absolute inset-0 deck-grid opacity-25" />

      <div className="absolute top-[7vh] left-[7vw] right-[7vw] flex items-center justify-between">
        <span className="font-mono text-[1.4vw] tracking-[0.35em] text-muted">
          QUANTLYTICS
        </span>
        <span className="font-mono text-[1.4vw] tracking-[0.35em] text-muted">
          08 / 08
        </span>
      </div>
      <div className="absolute top-[11vh] left-[7vw] right-[7vw] h-px bg-line" />

      <div className="absolute top-[17vh] left-[7vw] w-[55vw]">
        <div className="h-[0.5vh] w-[8vw] rule-gradient" />
        <h2 className="mt-[2.5vh] font-display text-[4.2vw] font-extrabold tracking-tighter leading-[1]">
          Next steps
        </h2>
      </div>

      <div className="absolute top-[38vh] left-[7vw] w-[74vw]">
        <div className="flex items-baseline gap-[2.5vw] border-t border-line py-[2.4vh]">
          <span className="font-mono text-[1.8vw] text-primary w-[4vw]">01</span>
          <span className="font-display text-[2.2vw] font-bold tracking-tight w-[26vw]">
            Connect a live venue
          </span>
          <span className="font-body text-[1.9vw] text-muted">
            Swap the demo feed for funded keys.
          </span>
        </div>
        <div className="flex items-baseline gap-[2.5vw] border-t border-line py-[2.4vh]">
          <span className="font-mono text-[1.8vw] text-primary w-[4vw]">02</span>
          <span className="font-display text-[2.2vw] font-bold tracking-tight w-[26vw]">
            Run a shadow book
          </span>
          <span className="font-body text-[1.9vw] text-muted">
            Trade alongside the desk for thirty days.
          </span>
        </div>
        <div className="flex items-baseline gap-[2.5vw] border-t border-b border-line py-[2.4vh]">
          <span className="font-mono text-[1.8vw] text-primary w-[4vw]">03</span>
          <span className="font-display text-[2.2vw] font-bold tracking-tight w-[26vw]">
            Set the risk envelope
          </span>
          <span className="font-body text-[1.9vw] text-muted">
            Per-agent limits before capital is committed.
          </span>
        </div>
      </div>

      <div className="absolute bottom-[7vh] left-[7vw] right-[7vw] flex items-end justify-between">
        <span className="font-display text-[2.4vw] font-bold tracking-tight text-gradient">
          The intelligent edge in crypto.
        </span>
        <span className="font-mono text-[1.5vw] text-muted">
          quantlytics · 2026
        </span>
      </div>
    </div>
  );
}
