import { createRoot } from 'react-dom/client';

import App from './App';

import './index.css';

// Apply dark theme to root HTML element (app is dark-only)
document.documentElement.classList.add('dark');

createRoot(document.getElementById('root')!).render(<App />);
