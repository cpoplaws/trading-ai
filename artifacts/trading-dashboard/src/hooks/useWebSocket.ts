import { useState, useEffect, useRef } from 'react'
import { io, Socket } from 'socket.io-client'

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'

/**
 * Socket.IO always connects to `<origin>` + its `path` option, and treats any
 * path in the URL it is given as a namespace instead. When `VITE_WS_URL` points
 * at a backend mounted under a path prefix — the staging mirror at
 * `/api/staging-trading`, or any reverse-proxied deployment — that prefix has
 * to be handed over as `path`.
 *
 * For a root-mounted backend (`ws://localhost:8000`, the Python engine) this
 * resolves to the default `/socket.io` and behaviour is unchanged.
 */
function resolveSocketTarget(url: string): { origin: string; path: string } {
  try {
    const parsed = new URL(url)
    const prefix = parsed.pathname.replace(/\/+$/, '')
    return {
      origin: `${parsed.protocol}//${parsed.host}`,
      path: `${prefix}/socket.io`,
    }
  } catch {
    return { origin: url, path: '/socket.io' }
  }
}

export function useWebSocket() {
  const [connected, setConnected] = useState(false)
  const [data, setData] = useState<any>(null)
  const socketRef = useRef<Socket | null>(null)

  useEffect(() => {
    const { origin, path } = resolveSocketTarget(WS_URL)

    socketRef.current = io(origin, {
      path,
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
