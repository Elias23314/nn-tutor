const RUNGS = [
  { title: 'Start here', meta: 'Why it matters' },
  { n: 1, title: 'What it does', meta: 'No math' },
  { n: 2, title: 'How it works', meta: 'No math' },
  { n: 3, title: 'With numbers', meta: 'Arithmetic' },
  { n: 4, title: 'The math', meta: 'Calc I' },
  { title: 'Playground', meta: 'Any depth' },
]

/** The depth ladder from the mockups. Only the playground is built so far. */
export function TopBar({ current }: { current: string }) {
  return (
    <header className="topbar">
      <div className="brand">
        <b>How a machine learns</b>
        <span>A neural network tutor</span>
      </div>
      <nav className="ladder" aria-label="Depth">
        {RUNGS.map((r, i) => {
          const isCurrent = r.title === current
          return (
            <div key={r.title} style={{ display: 'flex', alignItems: 'center' }}>
              {i > 0 && <div className="rail" />}
              <div
                className={`rung${isCurrent ? ' cur' : ' soon'}`}
                aria-current={isCurrent ? 'page' : undefined}
                title={isCurrent ? undefined : 'Not built yet'}
              >
                <div className="rt">
                  {r.n && <span className="n">{r.n}</span>}
                  {r.title}
                </div>
                <div className="m">{r.meta}</div>
              </div>
            </div>
          )
        })}
      </nav>
    </header>
  )
}
