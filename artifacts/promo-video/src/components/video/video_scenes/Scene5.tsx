import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

export const Scene5: React.FC = () => {
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const timers = [
      setTimeout(() => setPhase(1), 500),
      setTimeout(() => setPhase(2), 1200),
    ];
    return () => timers.forEach(t => clearTimeout(t));
  }, []);

  return (
    <motion.div 
      className="absolute inset-0 bg-bg-dark flex flex-col items-center justify-center"
      initial={{ opacity: 0, scale: 1.5 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, filter: "blur(20px)" }}
      transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
    >
      <motion.div 
        className="absolute inset-0 bg-gradient-primary opacity-20 blur-[150px]"
        initial={{ scale: 0.5, opacity: 0 }}
        animate={{ scale: phase >= 1 ? 1 : 0.5, opacity: phase >= 1 ? 0.2 : 0 }}
        transition={{ duration: 2 }}
      />

      <div className="relative z-10 flex flex-col items-center">
        <motion.div
          className="w-[12vw] h-[12vw] rounded-[3vw] bg-gradient-primary flex items-center justify-center mb-[4vh] glow-primary"
          initial={{ scale: 0, rotate: 180 }}
          animate={{ scale: phase >= 1 ? 1 : 0, rotate: phase >= 1 ? 0 : 180 }}
          transition={{ type: "spring", stiffness: 150, damping: 15 }}
        >
          <svg className="w-[6vw] h-[6vw] text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </motion.div>

        <div className="overflow-hidden">
          <motion.h1 
            className="text-[8vw] font-bold tracking-tight text-white m-0 leading-none"
            initial={{ y: "100%" }}
            animate={{ y: phase >= 1 ? "0%" : "100%" }}
            transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1], delay: 0.2 }}
          >
            Quantlytics
          </motion.h1>
        </div>

        <motion.div 
          className="mt-[4vh]"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: phase >= 2 ? 1 : 0, y: phase >= 2 ? 0 : 20 }}
          transition={{ duration: 0.6 }}
        >
          <p className="text-[2.5vw] font-medium text-text-secondary m-0">
            The intelligent edge in crypto.
          </p>
        </motion.div>
      </div>
    </motion.div>
  );
};
