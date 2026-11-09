import { TopBar } from './components/TopBar'
import { Layer1 } from './layer1/Layer1'
import { Layer2 } from './layer2/Layer2'
import { Layer3 } from './layer3/Layer3'
import { Playground } from './playground/Playground'
import { useRoute } from './routes'
import { Start } from './start/Start'

function App() {
  const route = useRoute()
  return (
    <>
      <TopBar current={route} />
      {route === 'start' && <Start />}
      {route === 'layer-1' && <Layer1 />}
      {route === 'layer-2' && <Layer2 />}
      {route === 'layer-3' && <Layer3 />}
      {route === 'playground' && <Playground />}
    </>
  )
}

export default App
