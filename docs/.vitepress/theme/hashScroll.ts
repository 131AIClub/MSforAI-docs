import { nextTick, onBeforeUnmount } from 'vue'
import { onContentUpdated } from 'vitepress'

const SCROLL_OFFSET = 134
const IMAGE_WAIT_TIMEOUT = 3000
const SETTLE_ATTEMPTS = 3
const SETTLE_INTERVAL = 150

let pendingCorrection: Promise<void> | null = null

function waitForArticleImages() {
  const images = Array.from(document.querySelectorAll<HTMLImageElement>('.vp-doc img'))
  const pendingImages = images.filter((img) => !img.complete)
  if (pendingImages.length === 0) return Promise.resolve()
  return Promise.race([
    Promise.all(
      pendingImages.map(
        (img) =>
          new Promise<void>((resolve) => {
            const done = () => resolve()
            img.addEventListener('load', done, { once: true })
            img.addEventListener('error', done, { once: true })
          })
      )
    ),
    new Promise<void>((resolve) => setTimeout(resolve, IMAGE_WAIT_TIMEOUT))
  ])
}

async function settleHashScroll(id: string) {
  const targetTop = () => {
    const el = document.getElementById(id)
    if (!el) return null
    return Math.max(0, el.getBoundingClientRect().top + window.scrollY - SCROLL_OFFSET)
  }

  for (let attempt = 0; attempt < SETTLE_ATTEMPTS; attempt++) {
    const top = targetTop()
    if (top == null) return
    if (Math.abs(window.scrollY - top) > 2) {
      window.scrollTo({ top, behavior: 'instant' })
    }
    await new Promise((resolve) => setTimeout(resolve, SETTLE_INTERVAL))
    const settledTop = targetTop()
    if (settledTop != null && Math.abs(window.scrollY - settledTop) <= 2) return
  }
}

function scrollToHashAfterImages() {
  const hash = window.location.hash
  if (!hash) return
  let id: string
  try {
    id = decodeURIComponent(hash).slice(1)
  } catch {
    return
  }
  if (!document.getElementById(id) || pendingCorrection) return

  const navigationKey = `${window.location.pathname}${window.location.search}`
  pendingCorrection = waitForArticleImages()
    .then(async () => {
      if (`${window.location.pathname}${window.location.search}` !== navigationKey) return
      await nextTick()
      await settleHashScroll(id)
    })
    .finally(() => {
      pendingCorrection = null
    })
}

export function useHashScroll() {
  const off = onContentUpdated(scrollToHashAfterImages)
  onBeforeUnmount(off)
}
