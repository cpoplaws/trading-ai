import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

export const Scene3: React.FC = () => {
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const timers = [
      setTimeout(() => setPhase(1), 300),
      setTimeout(() => setPhase(2), 800),
      setTimeout(() => setPhase(3), 1600),
      setTimeout(() => setPhase(4), 2200),
    ];
    return () => timers.forEach(t => clearTimeout(t));
  }, []);

  return (
    <motion.div 
      className="absolute inset-0 bg-bg-dark p-[6vw] flex gap-[4vw]"
      initial={{ opacity: 0, x: "100%" }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, scale: 0.9, y: "-10%" }}
      transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
    >
      {/* Left Column: Market Intelligence */}
      <div className="flex-1 flex flex-col justify-center">
        <motion.h2 
          className="text-[3vw] font-bold text-white mb-[4vh] flex items-center gap-[1vw]"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: phase >= 1 ? 1 : 0, y: phase >= 1 ? 0 : 20 }}
        >
          <svg className="w-[3vw] h-[3vw] text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
          </svg>
          Market Intelligence
        </motion.h2>

        <motion.div 
          className="bg-bg-panel border border-white/10 rounded-2xl p-[3vw] relative overflow-hidden"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: phase >= 2 ? 1 : 0, scale: phase >= 2 ? 1 : 0.9 }}
          transition={{ type: "spring", bounce: 0.3 }}
        >
          {/* Animated Background Gradient */}
          <motion.div 
            className="absolute -right-[10vw] -top-[10vw] w-[30vw] h-[30vw] bg-success/20 rounded-full blur-[80px]"
            animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0.8, 0.5] }}
            transition={{ duration: 4, repeat: Infinity }}
          />

          <div className="relative z-10">
            <div className="text-[1.2vw] text-text-secondary uppercase tracking-widest mb-[1vh]">Current Regime</div>
            <div className="text-[4vw] font-bold text-success mb-[2vh] flex items-center gap-[1vw]">
              Bull Trend
              <svg className="w-[3vw] h-[3vw] text-success" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
            
            <div className="space-y-[2vh]">
              <div className="flex justify-between items-center bg-black/30 p-[1.5vw] rounded-lg border border-white/5">
                <span className="text-[1.5vw] text-text-secondary">Momentum</span>
                <span className="text-[1.8vw] font-bold text-success">+8.42</span>
              </div>
              <div className="flex justify-between items-center bg-black/30 p-[1.5vw] rounded-lg border border-white/5">
                <span className="text-[1.5vw] text-text-secondary">Volatility</span>
                <span className="text-[1.8vw] font-bold text-warning">Medium</span>
              </div>
              <div className="flex justify-between items-center bg-black/30 p-[1.5vw] rounded-lg border border-white/5">
                <span className="text-[1.5vw] text-text-secondary">Composite Signal</span>
                <span className="text-[1.8vw] font-bold text-primary">Strong Buy (92%)</span>
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Right Column: Strategy Grid */}
      <div className="flex-1 flex flex-col justify-center">
        <motion.h2 
          className="text-[3vw] font-bold text-white mb-[4vh]"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: phase >= 3 ? 1 : 0, y: phase >= 3 ? 0 : 20 }}
        >
          Active Strategies
        </motion.h2>

        <div className="grid grid-cols-2 gap-[2vw]">
          {[
            { name: 'Trend Following', win: '68%', trades: 142, status: 'Active', color: 'text-success' },
            { name: 'Mean Reversion', win: '54%', trades: 89, status: 'Active', color: 'text-success' },
            { name: 'Stat Arb', win: '72%', trades: 412, status: 'Active', color: 'text-success' },
            { name: 'Momentum Scalp', win: '61%', trades: 305, status: 'Paused', color: 'text-warning' }
          ].map((strat, i) => (
            <motion.div 
              key={strat.name}
              className="bg-bg-panel border border-white/10 rounded-xl p-[2vw]"
              initial={{ opacity: 0, scale: 0.8, y: 30 }}
              animate={{ 
                opacity: phase >= 4 ? 1 : 0, 
                scale: phase >= 4 ? 1 : 0.8,
                y: phase >= 4 ? 0 : 30 
              }}
              transition={{ type: "spring", bounce: 0.4, delay: i * 0.1 }}
            >
              <div className="flex justify-between items-start mb-[2vh]">
                <div className="text-[1.4vw] font-bold text-white">{strat.name}</div>
                <div className={`text-[1vw] px-[0.8vw] py-[0.2vh] rounded bg-white/5 border border-white/10 ${strat.color}`}>
                  {strat.status}
                </div>
              </div>
              <div className="flex justify-between items-end">
                <div>
                  <div className="text-[1vw] text-text-secondary">Win Rate</div>
                  <div className="text-[2vw] font-mono text-white">{strat.win}</div>
                </div>
                <div className="text-right">
                  <div className="text-[1vw] text-text-secondary">Trades</div>
                  <div className="text-[1.5vw] font-mono text-white">{strat.trades}</div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </motion.div>
  );
};
