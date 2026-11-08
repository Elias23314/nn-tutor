import type { Point } from '../engine/trainer'
import { thousands } from '../playground/connection'
import { THRESHOLD } from './story'

const W = 630
const H = 270
const L = 58
const R = 610
const T = 16
const B = 222

interface Props {
  overall: Point[]
  digit: number
  digitPoints: Point[]
  roundsPerPass: number
}

const line = (points: Point[], x: (r: number) => number, y: (v: number) => number) =>
  points.map((p) => `${x(p.round).toFixed(1)},${y(p.value).toFixed(1)}`).join(' ')

/** Percent of test digits right as it practices: all digits (solid) and one digit (dashed). */
export function PracticeChart({ overall, digit, digitPoints, roundsPerPass }: Props) {
  const lastRound = overall.at(-1)?.round ?? 0
  const xMax = Math.max(roundsPerPass, lastRound)
  const x = (r: number) => L + ((R - L) * r) / xMax
  const y = (v: number) => B - (B - T) * v
  const xTicks = [0, Math.round(xMax / 200) * 100, xMax]
  const latest = overall.at(-1)
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: '100%', height: 'auto', display: 'block' }}
      role="img"
      aria-label={`Percent of test digits right over practice rounds${latest ? `: ${(latest.value * 100).toFixed(1)}% at round ${thousands(latest.round)}` : ''}`}
    >
      {[0, 0.5, 1].map((t) => (
        <g key={t}>
          <line x1={L} y1={y(t)} x2={R} y2={y(t)} stroke="var(--track)" />
          <text x={48} y={y(t) + 4} fontSize={12} fill="var(--muted)" textAnchor="end">
            {t * 100}
          </text>
        </g>
      ))}
      <line x1={L} y1={y(THRESHOLD)} x2={R} y2={y(THRESHOLD)} stroke="#B9BCC1" strokeDasharray="2 3" />
      <text x={R} y={y(THRESHOLD) - 5} fontSize={11.5} fill="var(--muted)" textAnchor="end">
        80%: counts as right
      </text>
      {xTicks.map((t) => (
        <text key={t} x={x(t)} y={242} fontSize={12} fill="var(--muted)" textAnchor="middle">
          {thousands(t)}
        </text>
      ))}
      {digitPoints.length > 1 && (
        <polyline points={line(digitPoints, x, y)} fill="none" stroke="var(--bar)" strokeWidth={2} strokeDasharray="6 4" />
      )}
      {overall.length > 1 && <polyline points={line(overall, x, y)} fill="none" stroke="var(--ink)" strokeWidth={2.3} />}
      {overall.length === 1 && <circle cx={x(overall[0].round)} cy={y(overall[0].value)} r={3.5} fill="var(--ink)" />}
      {digitPoints.length > 1 && (
        <text x={x(digitPoints.at(-1)!.round) + 4} y={y(digitPoints.at(-1)!.value) + 14} fontSize={12} fill="var(--muted)">
          {digit}s
        </text>
      )}
      <text x={(L + R) / 2} y={262} fontSize={12} fill="var(--muted)" textAnchor="middle">
        Practice rounds
      </text>
      <text x={14} y={(T + B) / 2} fontSize={12} fill="var(--muted)" textAnchor="middle" transform={`rotate(-90 14 ${(T + B) / 2})`}>
        Percent right
      </text>
    </svg>
  )
}
