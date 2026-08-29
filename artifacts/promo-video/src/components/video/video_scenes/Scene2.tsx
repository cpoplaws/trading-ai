import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

const agents = [
  { name: 'Execution Agent', desc: 'Optimizes Trade Timing', color: 'text-blue-400', border: 'border-blue-500/50', bg: 'bg-blue-500/10' },
  { name: 'Risk Agent', desc: 'Monitors Limits', color: 'text-amber-400', border: 'border-amber-500/50', bg: 'bg-amber-500/10' },
  { name: 'Arbitrage Agent', desc: 'Captures Discrepancies', color: 'text-purple-400', border: 'border-purple-500/50', bg: 'bg-purple-500/10' },
  { name: 'Market Making', desc: 'Provides Liquidity', color: 'text-green-400', border: 'border-green-500/50', bg: 'bg-green-500/10' },
];

export const Scene2: React.FC = () => {
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const timers = [
      setTimeout(() => setPhase(1), 400),
      setTimeout(() => setPhase(2), 1000),
      setTimeout(() => setPhase(3), 2000),
      setTimeout(() => setPhase(4), 3000),
    ];
    return () => timers.forEach(t => clearTimeout(t));
  }, []);

  return (
    <motion.div 
      className="absolute inset-0 bg-bg-dark overflow-hidden flex flex-col justify-center items-center"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, x: "-100%" }}
      transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
    >
      <motion.div 
        className="absolute top-[8vh] left-[6vw]"
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: phase >= 1 ? 1 : 0, x: phase >= 1 ? 0 : -50 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        <h2 className="text-[3vw] font-bold text-white mb-[1vh]">AI Agent Swarm</h2>
        <p className="text-[1.8vw] text-text-secondary">Four specialized models cooperating in real-time</p>
      </motion.div>

      {/* Network Container */}
      <div className="relative w-[60vw] h-[60vw] mt-[10vh] max-h-[70vh]">
        {/* Central Hub */}
        <motion.div 
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[12vw] h-[12vw] rounded-full border-[0.2vw] border-primary/30 bg-primary/10 flex items-center justify-center glow-primary z-20"
          initial={{ scale: 0 }}
          animate={{ scale: phase >= 2 ? 1 : 0 }}
          transition={{ type: "spring", bounce: 0.5 }}
        >
          <div className="text-center">
            <div className="text-[1.2vw] font-mono text-primary animate-pulse">99.8%</div>
            <div className="text-[1vw] text-white">Consensus</div>
          </div>
        </motion.div>

        {/* Central pulsing ring */}
        <motion.div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[18vw] h-[18vw] rounded-full border border-primary/20 z-10"
          initial={{ scale: 0, opacity: 0 }}
          animate={phase >= 2 ? { scale: [1, 1.5], opacity: [0.5, 0] } : {}}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
        />

        {/* Agents */}
        {agents.map((agent, i) => {
          const angleRad = (i * Math.PI) / 2 - Math.PI / 4;
          const angleDeg = i * 90 - 45;
          const radius = 22; // vw
          const x = Math.cos(angleRad) * radius;
          const y = Math.sin(angleRad) * radius;
          
          const showPhase = phase >= 3;
          
          return (
            <React.Fragment key={agent.name}>
              {/* Connection Line */}
              <motion.div 
                className="absolute top-1/2 left-1/2 h-[0.2vw] bg-gradient-to-r from-transparent to-primary/50 origin-left z-0"
                style={{ width: `${radius}vw`, rotate: angleDeg }}
                initial={{ scaleX: 0 }}
                animate={{ scaleX: showPhase ? 1 : 0 }}
                transition={{ duration: 0.6, delay: i * 0.1 }}
              />
              
              {/* Agent Node */}
              <motion.div 
                className={`absolute w-[16vw] p-[1.5vw] rounded-xl border-[0.1vw] ${agent.border} ${agent.bg} backdrop-blur-md z-30`}
                style={{ 
                  left: `calc(50% + ${x}vw)`, 
                  top: `calc(50% + ${y}vw)`,
                  x: "-50%",
                  y: "-50%"
                }}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: showPhase ? 1 : 0, opacity: showPhase ? 1 : 0 }}
                transition={{ type: "spring", bounce: 0.4, delay: i * 0.15 + 0.2 }}
              >
                <div className={`text-[1.2vw] font-bold ${agent.color} mb-[0.5vh]`}>{agent.name}</div>
                <div className="text-[1vw] text-white/70">{agent.desc}</div>
                
                {/* Decision Output Simulation */}
                <motion.div 
                  className="mt-[1vh] font-mono text-[0.8vw] bg-black/50 p-[0.5vw] rounded"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: phase >= 4 ? 1 : 0 }}
                  transition={{ duration: 0.3, delay: i * 0.1 + 1.5 }}
                >
                  <span className="text-green-400">Signal:</span> {Math.floor(Math.random() * 20 + 80)}% CONF
                </motion.div>
              </motion.div>
            </React.Fragment>
          );
        })}
      </div>
    </motion.div>
  );
};
