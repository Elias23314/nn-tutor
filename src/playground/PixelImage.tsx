import { useEffect, useRef } from 'react'
import { IMAGE_SIDE } from '../engine/network'

/** A 28 × 28 grayscale image (ink in [0, 1]) drawn with crisp pixels. */
export function PixelImage({ image, size, label }: { image: ArrayLike<number>; size: number; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const ctx = ref.current?.getContext('2d')
    if (!ctx) return
    const data = ctx.createImageData(IMAGE_SIDE, IMAGE_SIDE)
    for (let k = 0; k < IMAGE_SIDE * IMAGE_SIDE; k++) {
      const shade = 255 - Math.round(image[k] * 255)
      data.data.set([shade, shade, shade, 255], k * 4)
    }
    ctx.putImageData(data, 0, 0)
  }, [image])
  return (
    <canvas
      ref={ref}
      className="pix"
      width={IMAGE_SIDE}
      height={IMAGE_SIDE}
      style={{ width: size, height: size }}
      role="img"
      aria-label={label}
    />
  )
}
