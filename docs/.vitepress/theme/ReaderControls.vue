<script setup lang="ts">
import { ChevronsLeft, ListTree } from '@lucide/vue'
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useData } from 'vitepress'

const OUTLINE_KEY = 'msforai:reader-outline'

const { page } = useData()
const outlineCompact = ref(false)
const hasOutline = ref(false)
let availabilityFrame = 0
let outlineElement: HTMLElement | null = null

function applyLayoutState() {
  document.documentElement.classList.toggle(
    'reader-outline-compact',
    outlineCompact.value
  )
  syncOutlineAccessibility()
}

function syncOutlineAccessibility() {
  outlineElement = document.querySelector<HTMLElement>(
    '.chapter-reading-layout .VPDocAsideOutline.has-outline'
  )
  if (!outlineElement) return

  const inertOutline = outlineElement as HTMLElement & { inert?: boolean }
  inertOutline.inert = outlineCompact.value
  if (outlineCompact.value) {
    outlineElement.setAttribute('aria-hidden', 'true')
    outlineElement.setAttribute('inert', '')
  } else {
    outlineElement.removeAttribute('aria-hidden')
    outlineElement.removeAttribute('inert')
  }
}

function readStoredState() {
  try {
    outlineCompact.value = localStorage.getItem(OUTLINE_KEY) === 'hidden'
  } catch {
    outlineCompact.value = false
  }
  applyLayoutState()
}

function storeState() {
  try {
    localStorage.setItem(OUTLINE_KEY, outlineCompact.value ? 'hidden' : 'visible')
  } catch {
    // The control still works for the current page when storage is unavailable.
  }
}

function updateAvailability() {
  cancelAnimationFrame(availabilityFrame)
  availabilityFrame = requestAnimationFrame(() => {
    hasOutline.value = Boolean(
      document.querySelector('.VPDocAsideOutline.has-outline')
    )
    syncOutlineAccessibility()
  })
}

function toggleOutline() {
  outlineCompact.value = !outlineCompact.value
  storeState()
  applyLayoutState()
}

watch(
  () => page.value.relativePath,
  async () => {
    await nextTick()
    updateAvailability()
  }
)

onMounted(() => {
  readStoredState()
  updateAvailability()
})

onBeforeUnmount(() => {
  cancelAnimationFrame(availabilityFrame)
  if (outlineElement) {
    const inertOutline = outlineElement as HTMLElement & { inert?: boolean }
    inertOutline.inert = false
    outlineElement.removeAttribute('aria-hidden')
    outlineElement.removeAttribute('inert')
  }
  document.documentElement.classList.remove(
    'reader-outline-compact'
  )
})
</script>

<template>
  <div v-if="hasOutline" class="reader-controls" aria-label="页面大纲显示设置">
    <button
      class="reader-control"
      type="button"
      :title="outlineCompact ? '固定展开页面大纲' : '收起为大纲摘要'"
      :aria-label="outlineCompact ? '固定展开页面大纲' : '收起为大纲摘要'"
      aria-controls="doc-outline-aria-label"
      :aria-expanded="!outlineCompact"
      :aria-pressed="outlineCompact"
      @click="toggleOutline"
    >
      <span class="reader-control__icons" :class="{ 'is-compact': outlineCompact }" aria-hidden="true">
        <ChevronsLeft class="reader-control__icon reader-control__icon--collapse" :size="18" :stroke-width="1.7" />
        <ListTree class="reader-control__icon reader-control__icon--expand" :size="18" :stroke-width="1.7" />
      </span>
    </button>
  </div>
</template>
