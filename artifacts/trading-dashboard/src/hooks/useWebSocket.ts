import { useState, useEffect, useRef } from 'react'
import { io, Socket } from 'socket.io-client'

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'

export function useWebSocket() {
  const [connected, setConnected] = useState(false)
  const [data, setData] = useState<any>(null)
  const socketRef = useRef<Socket | null>(null)

  useEffect(() => {
    socketRef.current = io(WS_URL, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5,
    })

    const socket = socketRef.current

    socket.on('connect', () => {
      setConnected(true)
    })

    socket.on('disconnect', () => {
      setConnected(false)
    })

    socket.on('portfolio_update', (update: any) => {
      setData((prev: any) => ({ ...prev, portfolio: update }))
    })

    socket.on('trade_update', (trade: any) => {
      setData((prev: any) => ({ ...prev, latest_trade: trade }))
    })

    socket.on('strategy_update', (strategy: any) => {
      setData((prev: any) => ({ ...prev, strategies: strategy }))
    })

    return () => {
      socket.disconnect()
    }
  }, [])

  return { connected, data }
}
