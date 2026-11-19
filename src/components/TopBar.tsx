import { href, type Route } from '../routes'

const RUNGS: { title: string; meta: string; n?: number; route?: Route }[] = [
  { title: 'Start here', meta: 'Why it matters', route: 'start' },
  { n: 1, title: 'What it does', meta: 'No math', route: 'layer-1' },
  { n: 2, title: 'How it works', meta: 'No math', route: 'layer-2' },
  { n: 3, title: 'With numbers', meta: 'Arithmetic', route: 'layer-3' },
  { n: 4, title: 'The math', meta: 'Calculus', route: 'layer-4' },
  { title: 'Playground', meta: 'Any depth', route: 'playground' },
]

/** The depth ladder from the mockups. Rungs without a page yet are shown but not linked. */
export function TopBar({ current }: { current: Route }) {
  return (
    <header className="topbar">
      <a className="brand" href={href('start')}>
        <b>How a machine learns</b>
        <span>A neural network tutor</span>
      </a>
      <nav className="ladder" aria-label="Depth">
        {RUNGS.map((r, i) => {
          const isCurrent = r.route === current
          const body = (
            <>
              <div className="rt">
                {r.n && <span className="n">{r.n}</span>}
                {r.title}
              </div>
              <div className="m">{r.meta}</div>
            </>
          )
          return (
            <div key={r.title} style={{ display: 'flex', alignItems: 'center' }}>
              {i > 0 && <div className="rail" />}
              {r.route ? (
                <a className={`rung${isCurrent ? ' cur' : ''}`} href={href(r.route)} aria-current={isCurrent ? 'page' : undefined}>
                  {body}
                </a>
              ) : (
                <div className="rung soon" title="Not built yet">
                  {body}
                </div>
              )}
            </div>
          )
        })}
      </nav>
    </header>
  )
}
