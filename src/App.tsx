import { TopBar } from './components/TopBar'
import { Layer1 } from './layer1/Layer1'
import { Playground } from './playground/Playground'
import { useRoute } from './routes'

function App() {
  const route = useRoute()
  return (
    <>
      <TopBar current={route} />
      {route === 'layer-1' && <Layer1 />}
      {route === 'playground' && <Playground />}
    </>
  )
}

export default App
