<script setup lang="ts">
import { onBeforeUnmount } from 'vue'
import ChapterNavigationList from './ChapterNavigationList.vue'
import { chapterNavigation } from './chapterNavigation'

const props = defineProps<{
  open: boolean
  overlay: boolean
  width: number
}>()

const emit = defineEmits<{
  close: []
  navigate: []
  'update:width': [width: number]
  'resize-end': [width: number]
}>()

const MIN_WIDTH = 220
const MAX_WIDTH = 360
let startX = 0
let startWidth = 0
let pendingWidth = 0
let resizeFrame = 0
let resizing = false

function clampWidth(width: number) {
  return Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, Math.round(width)))
}

function flushWidth() {
  resizeFrame = 0
  emit('update:width', pendingWidth)
}

function handlePointerMove(event: PointerEvent) {
  pendingWidth = clampWidth(startWidth + event.clientX - startX)
  if (!resizeFrame) resizeFrame = requestAnimationFrame(flushWidth)
}

function stopResize() {
  if (!resizing) return
  resizing = false
  window.removeEventListener('pointermove', handlePointerMove)
  window.removeEventListener('pointerup', stopResize)
  window.removeEventListener('pointercancel', stopResize)
  document.documentElement.classList.remove('chapter-sidebar-resizing')
  if (resizeFrame) {
    cancelAnimationFrame(resizeFrame)
    flushWidth()
  }
  emit('resize-end', pendingWidth)
}

function startResize(event: PointerEvent) {
  if (event.button !== 0) return
  event.preventDefault()
  resizing = true
  startX = event.clientX
  startWidth = props.width
  pendingWidth = props.width
  document.documentElement.classList.add('chapter-sidebar-resizing')
  window.addEventListener('pointermove', handlePointerMove)
  window.addEventListener('pointerup', stopResize)
  window.addEventListener('pointercancel', stopResize)
}

function resizeWithKeyboard(event: KeyboardEvent) {
  const direction = event.key === 'ArrowLeft' ? -1 : event.key === 'ArrowRight' ? 1 : 0
  if (!direction) return
  event.preventDefault()
  const width = clampWidth(props.width + direction * 8)
  emit('update:width', width)
  emit('resize-end', width)
}

onBeforeUnmount(() => {
  stopResize()
  if (resizeFrame) cancelAnimationFrame(resizeFrame)
})
</script>

<template>
  <button
    v-if="overlay && open"
    class="chapter-sidebar-backdrop"
    type="button"
    aria-label="关闭讲义章节"
    @click="emit('close')"
  />

  <aside
    id="chapter-sidebar"
    class="chapter-sidebar"
    :class="{ 'is-open': open, 'is-overlay': overlay }"
    :style="{ width: `${width}px` }"
    :aria-hidden="!open"
    :inert="open ? undefined : ''"
  >
    <div class="chapter-sidebar__header">
      <span>课程讲义</span>
      <span class="chapter-sidebar__count">{{ chapterNavigation.length }} 章</span>
    </div>

    <div class="chapter-sidebar__scroller">
      <ChapterNavigationList @navigate="emit('navigate')" />
    </div>

    <div
      class="chapter-sidebar__resizer"
      role="separator"
      aria-label="调整章节侧栏宽度"
      aria-orientation="vertical"
      :aria-valuemin="MIN_WIDTH"
      :aria-valuemax="MAX_WIDTH"
      :aria-valuenow="width"
      tabindex="0"
      @pointerdown="startResize"
      @keydown="resizeWithKeyboard"
    />
  </aside>
</template>
