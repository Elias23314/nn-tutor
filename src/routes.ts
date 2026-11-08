import { useEffect, useState } from 'react'

/** Pages that exist so far, by their hash (#/layer-1). */
export type Route = 'layer-1' | 'layer-2' | 'playground'

export const ROUTES: Route[] = ['layer-1', 'layer-2', 'playground']
export const DEFAULT_ROUTE: Route = 'layer-1'

export const href = (route: Route) => `#/${route}`

function parse(hash: string): Route {
  const name = hash.replace(/^#\/?/, '')
  return (ROUTES as string[]).includes(name) ? (name as Route) : DEFAULT_ROUTE
}

/** The current page, following the URL hash. */
export function useRoute(): Route {
  const [route, setRoute] = useState(() => parse(location.hash))
  useEffect(() => {
    const onChange = () => {
      setRoute(parse(location.hash))
      window.scrollTo(0, 0)
    }
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}
