import { predict } from '../engine/network'
import { fmt } from './connection'

/** The ten outputs for the drawn digit, with its guess marked. */
export function OutputBars({ yHat }: { yHat: ArrayLike<number> | null }) {
  const guess = yHat ? predict(yHat) : -1
  return (
    <div className="bars" role="list" aria-label="Its ten outputs">
      {Array.from({ length: 10 }, (_, d) => {
        const v = yHat ? yHat[d] : 0
        return (
          <div className={`brow${d === guess ? ' top' : ''}`} key={d} role="listitem">
            <span className="dg">{d}</span>
            <div className="tr">
              <i style={{ width: `${(v * 100).toFixed(1)}%` }} />
            </div>
            <span className="vv">{yHat ? fmt(v, 2) : '–'}</span>
            <span className="nt">{d === guess ? 'its guess' : ''}</span>
          </div>
        )
      })}
    </div>
  )
}
