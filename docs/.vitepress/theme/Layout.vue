<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useData } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import ArticleEnd from './ArticleEnd.vue'
import ChapterNavigationList from './ChapterNavigationList.vue'
import ChapterSidebar from './ChapterSidebar.vue'
import ChapterSidebarToggle from './ChapterSidebarToggle.vue'
import DocumentContext from './DocumentContext.vue'
import ReaderControls from './ReaderControls.vue'

const SIDEBAR_KEY = 'msforai:chapter-sidebar'
const SIDEBAR_WIDTH_KEY = 'msforai:chapter-sidebar-width'
const DEFAULT_SIDEBAR_WIDTH = 268
const MIN_SIDEBAR_WIDTH = 220
const MAX_SIDEBAR_WIDTH = 360

const { page } = useData()
const isChapterPage = computed(() => page.value.relativePath.startsWith('chapters/'))
const chapterSidebarOpen = ref(true)
const chapterSidebarWidth = ref(DEFAULT_SIDEBAR_WIDTH)
const viewportWidth = ref(1440)
const layoutReady = ref(false)
let storedSidebarState: string | null = null
let layoutReadyFrame = 0

const chapterSidebarOverlay = computed(() => viewportWidth.value < 1280)
const layoutClasses = computed(() => ({
  'chapter-reading-layout': isChapterPage.value,
  'chapter-sidebar-open': isChapterPage.value && chapterSidebarOpen.value,
  'chapter-sidebar-overlay': isChapterPage.value && chapterSidebarOverlay.value,
  'chapter-layout-ready': isChapterPage.value && layoutReady.value
}))

const layoutStyle = computed(() => ({
  '--chapter-sidebar-width': `${chapterSidebarWidth.value}px`
}))

function clampSidebarWidth(width: number) {
  return Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, width))
}

function readSidebarPreferences() {
  try {
    storedSidebarState = localStorage.getItem(SIDEBAR_KEY)
    const storedWidth = Number(localStorage.getItem(SIDEBAR_WIDTH_KEY))
    if (Number.isFinite(storedWidth) && storedWidth > 0) {
      chapterSidebarWidth.value = clampSidebarWidth(storedWidth)
    }
  } catch {
    storedSidebarState = null
  }
}

function syncSidebarToViewport() {
  viewportWidth.value = window.innerWidth
  if (viewportWidth.value < 960) {
    chapterSidebarOpen.value = false
  } else if (storedSidebarState) {
    chapterSidebarOpen.value = storedSidebarState === 'visible'
  } else {
    chapterSidebarOpen.value = viewportWidth.value >= 1280
  }
}

function storeSidebarState() {
  storedSidebarState = chapterSidebarOpen.value ? 'visible' : 'hidden'
  try {
    localStorage.setItem(SIDEBAR_KEY, storedSidebarState)
  } catch {
    // The control still works for the current session when storage is unavailable.
  }
}

function toggleChapterSidebar() {
  chapterSidebarOpen.value = !chapterSidebarOpen.value
  storeSidebarState()
}

function closeChapterSidebar(persist = true) {
  chapterSidebarOpen.value = false
  if (persist) storeSidebarState()
}

function storeSidebarWidth(width: number) {
  chapterSidebarWidth.value = clampSidebarWidth(width)
  try {
    localStorage.setItem(SIDEBAR_WIDTH_KEY, String(chapterSidebarWidth.value))
  } catch {
    // Resizing remains available for the current session.
  }
}

function handleChapterNavigation() {
  if (chapterSidebarOverlay.value) closeChapterSidebar(false)
}

function handleKeydown(event: KeyboardEvent) {
  if (event.key === 'Escape' && chapterSidebarOverlay.value && chapterSidebarOpen.value) {
    closeChapterSidebar()
  }
}

onMounted(() => {
  readSidebarPreferences()
  syncSidebarToViewport()
  layoutReadyFrame = requestAnimationFrame(() => {
    layoutReady.value = true
  })
  window.addEventListener('resize', syncSidebarToViewport, { passive: true })
  window.addEventListener('keydown', handleKeydown)
})

onBeforeUnmount(() => {
  cancelAnimationFrame(layoutReadyFrame)
  window.removeEventListener('resize', syncSidebarToViewport)
  window.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
  <DefaultTheme.Layout :class="layoutClasses" :style="layoutStyle">
    <template #layout-top>
      <ChapterSidebar
        v-if="isChapterPage"
        :open="chapterSidebarOpen"
        :overlay="chapterSidebarOverlay"
        :width="chapterSidebarWidth"
        @close="closeChapterSidebar()"
        @navigate="handleChapterNavigation"
        @update:width="chapterSidebarWidth = $event"
        @resize-end="storeSidebarWidth"
      />
    </template>

    <template #nav-bar-content-before>
      <ChapterSidebarToggle
        v-if="isChapterPage"
        :open="chapterSidebarOpen"
        @toggle="toggleChapterSidebar"
      />
    </template>

    <template #nav-screen-content-before>
      <section v-if="isChapterPage" class="chapter-mobile-navigation">
        <div class="chapter-mobile-navigation__title">讲义章节</div>
        <ChapterNavigationList compact />
      </section>
    </template>

    <template #doc-before>
      <DocumentContext v-if="isChapterPage" />
    </template>

    <template #aside-outline-before>
      <ReaderControls v-if="isChapterPage" />
    </template>

    <template #doc-after>
      <ArticleEnd v-if="isChapterPage" />
    </template>
  </DefaultTheme.Layout>
</template>
