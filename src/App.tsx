import { TopBar } from './components/TopBar'
import { Playground } from './playground/Playground'

function App() {
  return (
    <>
      <TopBar current="Playground" />
      <Playground />
    </>
  )
}

export default App
