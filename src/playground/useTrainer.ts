import { useCallback, useEffect, useRef, useState } from 'react'
import type { Command, Report, Snapshot } from './protocol'

export type TrainerStatus =
  | { kind: 'loading' }
  | { kind: 'error'; message: string }
  | { kind: 'ready'; state: Snapshot; sample7: Float32Array }

/** Starts the training worker and keeps the latest snapshot of its state. */
export function useTrainer() {
  const worker = useRef<Worker | null>(null)
  const sample7 = useRef<Float32Array | null>(null)
  const [status, setStatus] = useState<TrainerStatus>({ kind: 'loading' })

  useEffect(() => {
    const w = new Worker(new URL('./trainer.worker.ts', import.meta.url), { type: 'module' })
    worker.current = w
    w.onmessage = (event: MessageEvent<Report>) => {
      const report = event.data
      if (report.type === 'ready') sample7.current = report.sample7
      else if (report.type === 'error') setStatus({ kind: 'error', message: report.message })
      else if (report.type === 'state' && sample7.current) {
        setStatus({ kind: 'ready', state: report.state, sample7: sample7.current })
      }
    }
    w.onerror = (event) => setStatus({ kind: 'error', message: event.message || 'The training worker failed' })
    return () => {
      w.terminate()
      worker.current = null
    }
  }, [])

  const send = useCallback((command: Command) => worker.current?.postMessage(command), [])
  return { status, send }
}
