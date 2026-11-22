import { useEffect, useRef } from 'react'
import { prepareDrawing } from '../engine/drawing'
import { IMAGE_SIDE } from '../engine/network'
import { PixelImage } from './PixelImage'

const SIZE = 280
const BRUSH = 20

interface Props {
  /** The 28 × 28 image the network sees. */
  image: Float32Array
  /** A 28 × 28 image to paint onto the pad: the sample 7 on arrival, a blank image on Clear. */
  seed: Float32Array | null
  upsideDown: boolean
  onChange: (image: Float32Array) => void
}

/** Paints a 28 × 28 image onto the big pad, smoothly scaled. */
function paintImage(ctx: CanvasRenderingContext2D, image: Float32Array) {
  const small = document.createElement('canvas')
  small.width = small.height = IMAGE_SIDE
  const sctx = small.getContext('2d')!
  const data = sctx.createImageData(IMAGE_SIDE, IMAGE_SIDE)
  image.forEach((v, k) => {
    const shade = 255 - Math.round(v * 255)
    data.data.set([shade, shade, shade, 255], k * 4)
  })
  sctx.putImageData(data, 0, 0)
  ctx.fillStyle = '#fff'
  ctx.fillRect(0, 0, SIZE, SIZE)
  ctx.imageSmoothingEnabled = true
  ctx.drawImage(small, 0, 0, SIZE, SIZE)
}

/** Reads the pad as ink in [0, 1] (black ink on white). */
function readInk(ctx: CanvasRenderingContext2D): Float32Array {
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE)
  const ink = new Float32Array(SIZE * SIZE)
  for (let k = 0; k < ink.length; k++) ink[k] = 1 - data[k * 4] / 255
  return ink
}

export function DrawPad({ image, seed, upsideDown, onChange }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const drawing = useRef(false)
  const last = useRef<[number, number] | null>(null)

  useEffect(() => {
    const ctx = canvas.current?.getContext('2d', { willReadFrequently: true })
    if (!ctx || !seed) return
    paintImage(ctx, seed)
  }, [seed])

  const point = (e: React.PointerEvent): [number, number] => {
    const rect = canvas.current!.getBoundingClientRect()
    return [((e.clientX - rect.left) / rect.width) * SIZE, ((e.clientY - rect.top) / rect.height) * SIZE]
  }

  const stroke = (to: [number, number]) => {
    const ctx = canvas.current!.getContext('2d', { willReadFrequently: true })!
    ctx.strokeStyle = '#000'
    ctx.lineWidth = BRUSH
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.beginPath()
    const from = last.current ?? to
    ctx.moveTo(from[0], from[1])
    ctx.lineTo(to[0] + 0.01, to[1])
    ctx.stroke()
    last.current = to
  }

  const finish = () => {
    if (!drawing.current) return
    drawing.current = false
    last.current = null
    const ctx = canvas.current!.getContext('2d', { willReadFrequently: true })!
    onChange(prepareDrawing(readInk(ctx), SIZE, SIZE))
  }

  return (
    <div className="pad">
      <div className="pad-row">
        <canvas
          ref={canvas}
          className="draw"
          width={SIZE}
          height={SIZE}
          aria-label="Drawing pad. Draw a digit with the mouse, a finger or a pen."
          onPointerDown={(e) => {
            canvas.current!.setPointerCapture(e.pointerId)
            drawing.current = true
            last.current = null
            stroke(point(e))
          }}
          onPointerMove={(e) => drawing.current && stroke(point(e))}
          onPointerUp={finish}
          onPointerCancel={finish}
        />
        <div className="seen">
          <PixelImage image={image} size={52} label="What the network sees" />
          <span>{upsideDown ? 'upside down' : 'it sees'}</span>
        </div>
      </div>
      <span className="small">Centered and shrunk to 28 × 28, like its training digits</span>
    </div>
  )
}
