import React, { useEffect, useState } from 'react';
import { motion, useMotionValue, useTransform, animate } from 'framer-motion';

const AnimatedCounter = ({ from, to, prefix = "", suffix = "", decimals = 0 }: any) => {
  const count = useMotionValue(from);
  const rounded = useTransform(count, (latest) => {
    return prefix + latest.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) + suffix;
  });

  useEffect(() => {
    const controls = animate(count, to, { duration: 2, ease: "easeOut" });
    return controls.stop;
  }, [count, to]);

  return <motion.span>{rounded}</motion.span>;
};

export const Scene4: React.FC = () => {
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const timers = [
      setTimeout(() => setPhase(1), 300),
      setTimeout(() => setPhase(2), 1000),
      setTimeout(() => setPhase(3), 1500),
    ];
    return () => timers.forEach(t => clearTimeout(t));
  }, []);

  return (
    <motion.div 
      className="absolute inset-0 bg-bg-dark flex flex-col items-center justify-center p-[6vw] bg-grid-pattern"
      initial={{ opacity: 0, y: "20%" }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, filter: "blur(20px)" }}
      transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
    >
      <motion.div
        className="w-full max-w-[80vw]"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: phase >= 1 ? 1 : 0, scale: phase >= 1 ? 1 : 0.9 }}
        transition={{ duration: 0.6 }}
      >
        <div className="text-center mb-[6vh]">
          <div className="text-[2vw] text-text-secondary uppercase tracking-widest mb-[2vh]">Total Portfolio Value</div>
          <div className="text-[10vw] font-bold text-white font-mono leading-none flex justify-center items-center gap-[2vw]">
            {phase >= 2 ? <AnimatedCounter from={1450000} to={1524382.45} prefix="$" decimals={2} /> : "$1,450,000.00"}
          </div>
          <div className="text-[3vw] text-success font-mono mt-[2vh] flex items-center justify-center gap-[1vw]">
            <span className="text-[2vw]">↑</span>
            {phase >= 2 ? <AnimatedCounter from={12000} to={24382.45} prefix="+$" decimals={2} /> : "+$12,000.00"}
            <span className="text-text-secondary">
              ({phase >= 2 ? <AnimatedCounter from={0.8} to={1.62} prefix="+" suffix="%" decimals={2} /> : "+0.8%"})
            </span>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-[2vw] mb-[6vh]">
          {[
            { label: 'Sharpe Ratio', from: 1.2, to: 2.85, decimals: 2 },
            { label: 'Win Rate', from: 45.0, to: 68.4, suffix: '%', decimals: 1 },
            { label: 'Live Positions', from: 5, to: 14, decimals: 0 },
          ].map((stat, i) => (
            <motion.div 
              key={stat.label}
              className="bg-bg-panel border border-white/10 rounded-2xl p-[2vw] text-center relative overflow-hidden"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: phase >= 3 ? 1 : 0, y: phase >= 3 ? 0 : 30 }}
              transition={{ duration: 0.6, delay: i * 0.1 }}
            >
              {/* Subtle bottom glow */}
              <div className="absolute -bottom-[20%] left-1/2 -translate-x-1/2 w-full h-[50%] bg-primary/10 blur-[30px]" />
              
              <div className="text-[1.2vw] text-text-secondary mb-[1vh] relative z-10">{stat.label}</div>
              <div className="text-[3vw] font-bold text-white font-mono relative z-10">
                {phase >= 3 ? <AnimatedCounter from={stat.from} to={stat.to} suffix={stat.suffix} decimals={stat.decimals} /> : stat.from}
              </div>
            </motion.div>
          ))}
        </div>

        {/* Recent Trades */}
        <motion.div
          className="bg-bg-panel border border-white/10 rounded-2xl overflow-hidden"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: phase >= 3 ? 1 : 0, y: phase >= 3 ? 0 : 30 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <div className="px-[2vw] py-[1.5vh] border-b border-white/10 bg-black/20 flex justify-between items-center">
            <span className="text-[1.2vw] font-bold text-white">Recent Trades</span>
            <span className="text-[1vw] text-text-secondary animate-pulse">● Live execution</span>
          </div>
          <div className="p-[2vw] space-y-[1.5vh]">
            {[
              { pair: 'BTC/USD', side: 'BUY', size: '2.5', price: '$64,230', pnl: '+$450', color: 'text-success' },
              { pair: 'SOL/USD', side: 'SELL', size: '145', price: '$142.50', pnl: '+$210', color: 'text-success' },
              { pair: 'ETH/USD', side: 'BUY', size: '18.4', price: '$3,105', pnl: '-$45', color: 'text-error' },
            ].map((trade, i) => (
              <motion.div 
                key={i} 
                className="flex justify-between items-center bg-black/40 p-[1vw] rounded-lg"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: phase >= 3 ? 1 : 0, x: phase >= 3 ? 0 : -20 }}
                transition={{ duration: 0.4, delay: 0.6 + i * 0.15 }}
              >
                <div className="flex gap-[2vw] items-center w-1/3">
                  <span className="text-[1.2vw] font-bold text-white font-mono">{trade.pair}</span>
                  <span className={`text-[1vw] font-bold ${trade.side === 'BUY' ? 'text-success' : 'text-error'}`}>{trade.side}</span>
                </div>
                <div className="flex justify-between items-center w-2/3">
                  <span className="text-[1.2vw] text-text-secondary font-mono">{trade.size}</span>
                  <span className="text-[1.2vw] text-white font-mono">{trade.price}</span>
                  <span className={`text-[1.2vw] font-bold font-mono ${trade.color}`}>{trade.pnl}</span>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </motion.div>
    </motion.div>
  );
};
