import { useState } from 'react'
import type { Point } from '../engine/trainer'
import { thousands } from './connection'

const W = 332
const H = 220
const L = 58
const R = 308
const T = 18
const B = 176

interface LineChartProps {
  points: Point[]
  yTicks: number[]
  yLabel: string
  formatY: (v: number) => string
  /** Map a value to 0–1 on the y axis. */
  scaleY: (v: number) => number
  roundsPerPass: number
  ariaLabel: string
}

function niceRoundTicks(maxRound: number): number[] {
  const steps = [100, 300, 900, 1_800, 3_600, 9_000, 18_000]
  const end = steps.find((s) => s >= maxRound) ?? maxRound
  return [0, end / 2, end]
}

function LineChart({ points, yTicks, yLabel, formatY, scaleY, roundsPerPass, ariaLabel }: LineChartProps) {
  const last = points.at(-1)?.round ?? 0
  const xTicks = niceRoundTicks(Math.max(last, Math.min(roundsPerPass, 100)))
  const xMax = xTicks[2]
  const x = (round: number) => L + ((R - L) * round) / xMax
  const y = (v: number) => B - (B - T) * Math.min(1, Math.max(0, scaleY(v)))
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={ariaLabel}>
      {yTicks.map((t) => (
        <g key={t}>
          <line x1={L} y1={y(t)} x2={R} y2={y(t)} stroke="var(--track)" />
          <text x={48} y={y(t) + 4} fontSize={12} fill="var(--muted)" textAnchor="end">
            {formatY(t)}
          </text>
        </g>
      ))}
      {xTicks.map((t) => (
        <text key={t} x={x(t)} y={196} fontSize={12} fill="var(--muted)" textAnchor="middle">
          {thousands(t)}
        </text>
      ))}
      {points.length > 1 && (
        <polyline
          points={points.map((p) => `${x(p.round).toFixed(1)},${y(p.value).toFixed(1)}`).join(' ')}
          fill="none"
          stroke="var(--ink)"
          strokeWidth={2.3}
          strokeLinejoin="round"
        />
      )}
      {points.length === 1 && <circle cx={x(points[0].round)} cy={y(points[0].value)} r={3} fill="var(--ink)" />}
      <text x={183} y={214} fontSize={12} fill="var(--muted)" textAnchor="middle">
        Practice rounds
      </text>
      <text x={14} y={97} fontSize={12} fill="var(--muted)" textAnchor="middle" transform="rotate(-90 14 97)">
        {yLabel}
      </text>
    </svg>
  )
}

export function AccuracyChart({ points, roundsPerPass }: { points: Point[]; roundsPerPass: number }) {
  const latest = points.at(-1)
  return (
    <div className="panel chart">
      <div className="ptitle">
        <span className="h3">Test digits right</span>
      </div>
      <LineChart
        points={points}
        yTicks={[0, 0.5, 1]}
        yLabel="Percent right"
        formatY={(v) => String(v * 100)}
        scaleY={(v) => v}
        roundsPerPass={roundsPerPass}
        ariaLabel={`Percent of test digits right over practice rounds${latest ? `, now ${(latest.value * 100).toFixed(1)}%` : ''}`}
      />
    </div>
  )
}

export function LossChart({ points, roundsPerPass }: { points: Point[]; roundsPerPass: number }) {
  const [log, setLog] = useState(false)
  const lo = Math.log10(0.003)
  const hi = Math.log10(0.3)
  return (
    <div className="panel chart">
      <div className="ptitle">
        <span className="h3">Mistake</span>
        <button className="tog" role="switch" aria-checked={log} onClick={() => setLog(!log)}>
          <span className="sw">
            <i />
          </span>
          Log
        </button>
      </div>
      <LineChart
        points={points}
        yTicks={log ? [0.003, 0.03, 0.3] : [0, 0.1, 0.2, 0.3]}
        yLabel="Average of ten squared gaps"
        formatY={(v) => String(v)}
        scaleY={log ? (v) => (Math.log10(Math.max(v, 1e-4)) - lo) / (hi - lo) : (v) => v / 0.3}
        roundsPerPass={roundsPerPass}
        ariaLabel="The mistake over practice rounds"
      />
    </div>
  )
}

export function ConfusionChart({ confusion }: { confusion: number[][] }) {
  const cell = 29
  const x0 = 46
  const y0 = 40
  return (
    <div className="panel chart">
      <div className="ptitle">
        <span className="h3">What it confuses</span>
        <span className="small">10,000 test digits</span>
      </div>
      <svg viewBox="0 0 354 346" role="img" aria-label="Table of which digits it reads as which">
        <text x={x0 + 5 * (cell + 1)} y={12} fontSize={12} fill="var(--muted)" textAnchor="middle">
          It said
        </text>
        <text x={10} y={y0 + 5 * (cell + 1)} fontSize={12} fill="var(--muted)" textAnchor="middle" transform={`rotate(-90 10 ${y0 + 5 * (cell + 1)})`}>
          Actual digit
        </text>
        {confusion.map((row, actual) => {
          const total = Math.max(1, row.reduce((a, b) => a + b, 0))
          return row.map((count, said) => {
            const share = count / total
            // Right answers are ink; mistakes are shaded by how common they are in that row.
            const fill =
              actual === said
                ? count === 0
                  ? '#fff'
                  : 'var(--ink)'
                : count === 0
                  ? '#fff'
                  : `rgb(${[255, 255, 255].map((c, i) => Math.round(c - (c - [0x8e, 0x94, 0x9c][i]) * Math.min(1, share * 6))).join(',')})`
            const cx = x0 + said * (cell + 1)
            const cy = y0 + actual * (cell + 1)
            return (
              <g key={`${actual}-${said}`}>
                <rect x={cx} y={cy} width={cell} height={cell} fill={fill} stroke="var(--line-2)" strokeWidth={0.5}>
                  <title>{`${thousands(count)} ${actual}s read as ${said}`}</title>
                </rect>
                {count > 0 && (
                  <text
                    x={cx + cell / 2}
                    y={cy + 18.5}
                    fontFamily="var(--mono)"
                    fontSize={count >= 1000 ? 8.5 : count >= 100 ? 9 : 10}
                    fontWeight={500}
                    fill={actual === said || share > 0.1 ? '#fff' : 'var(--ink)'}
                    textAnchor="middle"
                    pointerEvents="none"
                  >
                    {count}
                  </text>
                )}
              </g>
            )
          })
        })}
        {Array.from({ length: 10 }, (_, d) => (
          <g key={d}>
            <text x={38} y={y0 + d * (cell + 1) + 19} fontFamily="var(--mono)" fontSize={12} fontWeight={500} textAnchor="end">
              {d}
            </text>
            <text x={x0 + d * (cell + 1) + 15} y={32} fontFamily="var(--mono)" fontSize={12} fontWeight={500} textAnchor="middle">
              {d}
            </text>
          </g>
        ))}
      </svg>
    </div>
  )
}
