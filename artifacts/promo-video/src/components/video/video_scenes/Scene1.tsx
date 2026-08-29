import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

export const Scene1: React.FC = () => {
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const timers = [
      setTimeout(() => setPhase(1), 500),
      setTimeout(() => setPhase(2), 1500),
      setTimeout(() => setPhase(3), 2800),
    ];
    return () => timers.forEach(t => clearTimeout(t));
  }, []);

  return (
    <motion.div 
      className="absolute inset-0 flex flex-col items-center justify-center bg-grid-pattern"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, scale: 1.1, filter: "blur(10px)" }}
      transition={{ duration: 0.8 }}
    >
      {/* Background glow */}
      <motion.div 
        className="absolute w-[40vw] h-[40vw] rounded-full bg-primary/20 blur-[100px]"
        initial={{ scale: 0.5, opacity: 0 }}
        animate={{ scale: 1.5, opacity: 0.8 }}
        transition={{ duration: 3, ease: "easeOut" }}
      />
      
      <div className="relative z-10 flex flex-col items-center">
        <motion.div
          className="w-[8vw] h-[8vw] rounded-2xl bg-gradient-primary flex items-center justify-center mb-[3vh] glow-primary"
          initial={{ scale: 0, rotate: -45 }}
          animate={{ scale: phase >= 1 ? 1 : 0, rotate: phase >= 1 ? 0 : -45 }}
          transition={{ type: "spring", stiffness: 200, damping: 20 }}
        >
          <svg className="w-[4vw] h-[4vw] text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </motion.div>

        <div className="overflow-hidden">
          <motion.h1 
            className="text-[6vw] font-bold tracking-tight text-white m-0 leading-none"
            initial={{ y: "100%" }}
            animate={{ y: phase >= 1 ? "0%" : "100%" }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          >
            Quantlytics
          </motion.h1>
        </div>

        <div className="overflow-hidden mt-[1vh]">
          <motion.h2 
            className="text-[3vw] font-semibold text-gradient m-0 leading-tight"
            initial={{ y: "100%" }}
            animate={{ y: phase >= 2 ? "0%" : "100%" }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          >
            Crypto AI Trading Dashboard
          </motion.h2>
        </div>

        <motion.div 
          className="mt-[4vh] px-[2vw] py-[1vh] rounded-full border border-white/10 bg-white/5 backdrop-blur-md"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: phase >= 3 ? 1 : 0, y: phase >= 3 ? 0 : 20 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <p className="text-[1.5vw] font-medium text-text-secondary m-0 tracking-wide uppercase">
            Multi-Chain Trading • Base • Solana • L2s
          </p>
        </motion.div>
      </div>
    </motion.div>
  );
};
