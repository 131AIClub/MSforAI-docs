import { nextTick, onBeforeUnmount, onMounted, watch } from 'vue'
import { onContentUpdated, useData } from 'vitepress'

const OUTLINE_CONTAINER_SELECTOR = '.chapter-reading-layout .aside-container'
const FOLLOW_GAP = 12

export function useOutlineFollow() {
  const { page } = useData()
  let container: HTMLElement | null = null
  let observer: MutationObserver | null = null
  let followFrame = 0

  function pinnedControlHeight() {
    const controls = container?.querySelector<HTMLElement>('.reader-controls')
    return controls?.offsetHeight ?? 0
  }

  function followActiveLink() {
    followFrame = 0
    if (!container || container.scrollHeight <= container.clientHeight + 1) return

    const active = container.querySelector<HTMLElement>('.outline-link.active')
    if (!active) return

    const containerTop = container.getBoundingClientRect().top
    const containerBottom = container.getBoundingClientRect().bottom
    const visibleTop = containerTop + pinnedControlHeight()
    const linkTop = active.getBoundingClientRect().top
    const linkBottom = active.getBoundingClientRect().bottom
    if (linkTop >= visibleTop && linkBottom <= containerBottom) return

    const target = container.scrollTop + linkTop - visibleTop - FOLLOW_GAP
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    container.scrollTo({ top: Math.max(target, 0), behavior: reduceMotion ? 'auto' : 'smooth' })
  }

  function scheduleFollow() {
    if (!followFrame) followFrame = requestAnimationFrame(followActiveLink)
  }

  function ensureAttached() {
    const el = document.querySelector<HTMLElement>(OUTLINE_CONTAINER_SELECTOR)
    if (!el) return
    if (container === el && observer) return

    container = el
    observer?.disconnect()
    observer = new MutationObserver(scheduleFollow)
    observer.observe(el, { subtree: true, attributes: true, attributeFilter: ['class'] })
    scheduleFollow()
  }

  onMounted(ensureAttached)
  onContentUpdated(scheduleFollow)

  watch(
    () => page.value.relativePath,
    async () => {
      await nextTick()
      ensureAttached()
      scheduleFollow()
    }
  )

  onBeforeUnmount(() => {
    observer?.disconnect()
    if (followFrame) cancelAnimationFrame(followFrame)
  })
}
