<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { Maximize2, Minus, Plus, X } from '@lucide/vue'
import { useData } from 'vitepress'

const MIN_SCALE = 1
const MAX_SCALE = 5
const SCALE_STEP = 0.25
const IMAGE_SELECTOR = '.VPDoc .vp-doc img'
const INTERACTIVE_ANCESTOR_SELECTOR = 'a, button, [role="button"]'

const { page } = useData()
const open = ref(false)
const source = ref('')
const description = ref('')
const scale = ref(MIN_SCALE)
const offsetX = ref(0)
const offsetY = ref(0)
const stage = ref<HTMLElement | null>(null)
const preview = ref<HTMLImageElement | null>(null)
const closeButton = ref<HTMLButtonElement | null>(null)
const enhancedImages = new Set<HTMLImageElement>()
const originalAttributes = new WeakMap<HTMLImageElement, Map<string, string | null>>()
const pointers = new Map<number, { x: number; y: number }>()

let imageObserver: MutationObserver | null = null
let resizeObserver: ResizeObserver | null = null
let triggerImage: HTMLImageElement | null = null
let previousHtmlOverflow = ''
let previousBodyOverflow = ''
let previousBodyPaddingRight = ''
let panOrigin: { x: number; y: number; offsetX: number; offsetY: number } | null = null
let pinchOrigin: { distance: number; scale: number; centerX: number; centerY: number; offsetX: number; offsetY: number } | null = null
let pointerStart: { id: number; x: number; y: number; moved: boolean } | null = null
let lastTap: { time: number; x: number; y: number } | null = null
let suppressStageClick = false

const zoomPercent = computed(() => `${Math.round(scale.value * 100)}%`)
const dialogLabel = computed(() => description.value ? `图片预览：${description.value}` : '图片预览')
const imageStyle = computed(() => ({
  transform: `translate3d(${offsetX.value}px, ${offsetY.value}px, 0) scale(${scale.value})`
}))
const canPan = computed(() => scale.value > MIN_SCALE)

function isEligibleImage(target: EventTarget | null): target is HTMLImageElement {
  return target instanceof HTMLImageElement
    && target.matches(IMAGE_SELECTOR)
    && !target.parentElement?.closest(INTERACTIVE_ANCESTOR_SELECTOR)
}

function enhanceImage(image: HTMLImageElement) {
  if (!isEligibleImage(image) || enhancedImages.has(image)) return

  enhancedImages.add(image)
  originalAttributes.set(image, new Map(
    ['tabindex', 'role', 'aria-haspopup', 'aria-label', 'aria-keyshortcuts']
      .map((name) => [name, image.getAttribute(name)])
  ))
  image.classList.add('ms-zoomable-image')
  image.tabIndex = 0
  image.setAttribute('role', 'button')
  image.setAttribute('aria-haspopup', 'dialog')
  image.setAttribute('aria-label', image.alt ? `放大查看图片：${image.alt}` : '放大查看图片')
  image.setAttribute('aria-keyshortcuts', 'Enter Space')
}

function enhanceImages(root: ParentNode = document) {
  root.querySelectorAll<HTMLImageElement>(IMAGE_SELECTOR).forEach(enhanceImage)
}

function restoreEnhancedImages() {
  for (const image of enhancedImages) {
    image.classList.remove('ms-zoomable-image')
    for (const [name, value] of originalAttributes.get(image) ?? []) {
      if (value === null) image.removeAttribute(name)
      else image.setAttribute(name, value)
    }
  }
  enhancedImages.clear()
}

function lockPageScroll() {
  const scrollbarWidth = window.innerWidth - document.documentElement.clientWidth
  previousHtmlOverflow = document.documentElement.style.overflow
  previousBodyOverflow = document.body.style.overflow
  previousBodyPaddingRight = document.body.style.paddingRight
  document.documentElement.style.overflow = 'hidden'
  document.body.style.overflow = 'hidden'
  if (scrollbarWidth > 0) document.body.style.paddingRight = `${scrollbarWidth}px`
}

function unlockPageScroll() {
  document.documentElement.style.overflow = previousHtmlOverflow
  document.body.style.overflow = previousBodyOverflow
  document.body.style.paddingRight = previousBodyPaddingRight
}

function resetView() {
  scale.value = MIN_SCALE
  offsetX.value = 0
  offsetY.value = 0
  pointers.clear()
  panOrigin = null
  pinchOrigin = null
  pointerStart = null
  lastTap = null
}

function openImage(image: HTMLImageElement) {
  triggerImage = image
  source.value = image.currentSrc || image.src
  description.value = image.alt.trim()
  resetView()
  open.value = true
  lockPageScroll()
  nextTick(() => {
    closeButton.value?.focus()
    constrainOffset()
  })
}

function closeImage(restoreFocus = true) {
  if (!open.value) return

  const imageToFocus = triggerImage
  open.value = false
  unlockPageScroll()
  resetView()
  if (restoreFocus) nextTick(() => imageToFocus?.focus())
  triggerImage = null
}

function offsetLimits() {
  if (!stage.value || !preview.value) return { x: 0, y: 0 }
  const stageRect = stage.value.getBoundingClientRect()
  const imageWidth = preview.value.offsetWidth * scale.value
  const imageHeight = preview.value.offsetHeight * scale.value
  return {
    x: Math.max(0, (imageWidth - stageRect.width) / 2),
    y: Math.max(0, (imageHeight - stageRect.height) / 2)
  }
}

function constrainOffset() {
  const limits = offsetLimits()
  offsetX.value = Math.min(limits.x, Math.max(-limits.x, offsetX.value))
  offsetY.value = Math.min(limits.y, Math.max(-limits.y, offsetY.value))
}

function setScale(nextScale: number, clientX?: number, clientY?: number) {
  const oldScale = scale.value
  const clampedScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, nextScale))
  if (clampedScale === oldScale) return

  if (stage.value && clientX !== undefined && clientY !== undefined) {
    const rect = stage.value.getBoundingClientRect()
    const pointerX = clientX - (rect.left + rect.width / 2)
    const pointerY = clientY - (rect.top + rect.height / 2)
    const ratio = clampedScale / oldScale
    offsetX.value = pointerX - (pointerX - offsetX.value) * ratio
    offsetY.value = pointerY - (pointerY - offsetY.value) * ratio
  }

  scale.value = clampedScale
  if (clampedScale === MIN_SCALE) {
    offsetX.value = 0
    offsetY.value = 0
  }
  nextTick(constrainOffset)
}

function zoomIn() {
  setScale(scale.value + SCALE_STEP)
}

function zoomOut() {
  setScale(scale.value - SCALE_STEP)
}

function handleWheel(event: WheelEvent) {
  const factor = Math.exp(-event.deltaY * 0.002)
  setScale(scale.value * factor, event.clientX, event.clientY)
}

function distanceBetween(first: { x: number; y: number }, second: { x: number; y: number }) {
  return Math.hypot(second.x - first.x, second.y - first.y)
}

function midpoint(first: { x: number; y: number }, second: { x: number; y: number }) {
  return { x: (first.x + second.x) / 2, y: (first.y + second.y) / 2 }
}

function beginPinch() {
  const [first, second] = [...pointers.values()]
  if (!first || !second) return
  const center = midpoint(first, second)
  pinchOrigin = {
    distance: distanceBetween(first, second),
    scale: scale.value,
    centerX: center.x,
    centerY: center.y,
    offsetX: offsetX.value,
    offsetY: offsetY.value
  }
  panOrigin = null
}

function handlePointerDown(event: PointerEvent) {
  if (event.pointerType === 'mouse' && event.button !== 0) return
  if (!pointers.size) suppressStageClick = false
  if (event.pointerType !== 'mouse') stage.value?.setPointerCapture(event.pointerId)
  pointers.set(event.pointerId, { x: event.clientX, y: event.clientY })
  if (pointers.size === 1) {
    pointerStart = { id: event.pointerId, x: event.clientX, y: event.clientY, moved: false }
  }

  if (pointers.size === 2) {
    beginPinch()
  } else if (canPan.value) {
    panOrigin = {
      x: event.clientX,
      y: event.clientY,
      offsetX: offsetX.value,
      offsetY: offsetY.value
    }
  }
}

function handlePointerMove(event: PointerEvent) {
  if (!pointers.has(event.pointerId)) return
  pointers.set(event.pointerId, { x: event.clientX, y: event.clientY })
  if (
    pointerStart?.id === event.pointerId
    && Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 8
  ) {
    pointerStart.moved = true
    suppressStageClick = true
    if (!stage.value?.hasPointerCapture(event.pointerId)) {
      stage.value?.setPointerCapture(event.pointerId)
    }
  }

  if (pointers.size >= 2 && pinchOrigin) {
    suppressStageClick = true
    const [first, second] = [...pointers.values()]
    const center = midpoint(first, second)
    const nextScale = Math.min(
      MAX_SCALE,
      Math.max(MIN_SCALE, pinchOrigin.scale * distanceBetween(first, second) / pinchOrigin.distance)
    )
    const ratio = nextScale / pinchOrigin.scale
    scale.value = nextScale
    offsetX.value = center.x - pinchOrigin.centerX + pinchOrigin.offsetX * ratio
    offsetY.value = center.y - pinchOrigin.centerY + pinchOrigin.offsetY * ratio
    constrainOffset()
  } else if (panOrigin && canPan.value) {
    offsetX.value = panOrigin.offsetX + event.clientX - panOrigin.x
    offsetY.value = panOrigin.offsetY + event.clientY - panOrigin.y
    constrainOffset()
  }
}

function handlePointerEnd(event: PointerEvent) {
  const completedTap = pointerStart?.id === event.pointerId && !pointerStart.moved && pointers.size === 1
  pointers.delete(event.pointerId)
  if (stage.value?.hasPointerCapture(event.pointerId)) stage.value.releasePointerCapture(event.pointerId)

  pinchOrigin = null
  panOrigin = null
  pointerStart = null
  if (pointers.size === 2) {
    beginPinch()
  } else if (pointers.size === 1 && canPan.value) {
    const [remainingPointer] = pointers.values()
    panOrigin = {
      x: remainingPointer.x,
      y: remainingPointer.y,
      offsetX: offsetX.value,
      offsetY: offsetY.value
    }
  }
  if (completedTap) {
    const now = performance.now()
    if (
      lastTap
      && now - lastTap.time < 320
      && Math.hypot(event.clientX - lastTap.x, event.clientY - lastTap.y) < 24
    ) {
      resetView()
    } else {
      lastTap = { time: now, x: event.clientX, y: event.clientY }
    }
  } else {
    lastTap = null
  }
  constrainOffset()
}

function handleStageClick(event: MouseEvent) {
  if (event.target !== event.currentTarget) return
  if (suppressStageClick) {
    suppressStageClick = false
    return
  }
  closeImage()
}

function handleDocumentClick(event: MouseEvent) {
  if (isEligibleImage(event.target)) openImage(event.target)
}

function focusableControls() {
  return stage.value?.closest<HTMLElement>('.image-lightbox')
    ?.querySelectorAll<HTMLElement>('button:not(:disabled), [tabindex]:not([tabindex="-1"])') ?? []
}

function trapFocus(event: KeyboardEvent) {
  const controls = [...focusableControls()]
  if (!controls.length) return
  const first = controls[0]
  const last = controls[controls.length - 1]
  const activeIndex = controls.indexOf(document.activeElement as HTMLElement)
  if (activeIndex === -1) {
    event.preventDefault()
    ;(event.shiftKey ? last : first).focus()
  } else if (event.shiftKey && document.activeElement === first) {
    event.preventDefault()
    last.focus()
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault()
    first.focus()
  }
}

function handleDocumentKeydown(event: KeyboardEvent) {
  if (!open.value) {
    if ((event.key === 'Enter' || event.key === ' ') && isEligibleImage(event.target)) {
      event.preventDefault()
      openImage(event.target)
    }
    return
  }

  if (event.key === 'Escape') {
    event.preventDefault()
    closeImage()
  } else if (event.key === 'Tab') {
    trapFocus(event)
  } else if (event.key === '+' || event.key === '=') {
    event.preventDefault()
    zoomIn()
  } else if (event.key === '-' || event.key === '_') {
    event.preventDefault()
    zoomOut()
  } else if (event.key === '0') {
    event.preventDefault()
    resetView()
  }
}

onMounted(() => {
  enhanceImages()
  imageObserver = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (node instanceof HTMLImageElement) enhanceImage(node)
        else if (node instanceof Element) enhanceImages(node)
      }
    }
  })
  imageObserver.observe(document.body, { childList: true, subtree: true })
  resizeObserver = new ResizeObserver(constrainOffset)
  if (stage.value) resizeObserver.observe(stage.value)
  document.addEventListener('click', handleDocumentClick)
  document.addEventListener('keydown', handleDocumentKeydown)
})

onBeforeUnmount(() => {
  closeImage(false)
  imageObserver?.disconnect()
  resizeObserver?.disconnect()
  document.removeEventListener('click', handleDocumentClick)
  document.removeEventListener('keydown', handleDocumentKeydown)
  restoreEnhancedImages()
})

watch(() => page.value.relativePath, () => {
  closeImage(false)
  nextTick(enhanceImages)
})

watch(stage, (element, previous) => {
  if (previous) resizeObserver?.unobserve(previous)
  if (element) resizeObserver?.observe(element)
})
</script>

<template>
  <Teleport to="body">
    <Transition name="image-lightbox">
      <div
        v-if="open"
        class="image-lightbox"
        role="dialog"
        aria-modal="true"
        :aria-label="dialogLabel"
        @click.self="closeImage()"
      >
        <button
          ref="closeButton"
          class="image-lightbox__close"
          type="button"
          aria-label="关闭图片预览"
          @click="closeImage()"
        >
          <X :size="20" aria-hidden="true" />
        </button>

        <div
          ref="stage"
          class="image-lightbox__stage"
          :class="{ 'is-pannable': canPan }"
          @click="handleStageClick"
          @wheel.prevent="handleWheel"
          @pointerdown="handlePointerDown"
          @pointermove="handlePointerMove"
          @pointerup="handlePointerEnd"
          @pointercancel="handlePointerEnd"
        >
          <img
            ref="preview"
            class="image-lightbox__image"
            :src="source"
            :alt="description"
            :style="imageStyle"
            draggable="false"
            @load="constrainOffset"
            @dblclick.stop.prevent="resetView"
          />
        </div>

        <p v-if="description" class="image-lightbox__caption">{{ description }}</p>

        <div class="image-lightbox__toolbar" aria-label="图片缩放控制">
          <button type="button" :disabled="scale <= MIN_SCALE" aria-label="缩小图片" @click="zoomOut">
            <Minus :size="18" aria-hidden="true" />
          </button>
          <output aria-live="polite">{{ zoomPercent }}</output>
          <button type="button" :disabled="scale >= MAX_SCALE" aria-label="放大图片" @click="zoomIn">
            <Plus :size="18" aria-hidden="true" />
          </button>
          <span class="image-lightbox__divider" aria-hidden="true" />
          <button type="button" :disabled="scale === MIN_SCALE" aria-label="重置图片大小和位置" @click="resetView">
            <Maximize2 :size="17" aria-hidden="true" />
          </button>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>
