<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useData, useRouter, type Router } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import ArticleEnd from './ArticleEnd.vue'
import ChapterNavigationList from './ChapterNavigationList.vue'
import ChapterSidebar from './ChapterSidebar.vue'
import ChapterSidebarToggle from './ChapterSidebarToggle.vue'
import DocumentContext from './DocumentContext.vue'
import ReaderControls from './ReaderControls.vue'
import { enhanceCodeBlocks } from './codeBlocks'

const SIDEBAR_KEY = 'msforai:chapter-sidebar'
const SIDEBAR_WIDTH_KEY = 'msforai:chapter-sidebar-width'
const DEFAULT_SIDEBAR_WIDTH = 268
const MIN_SIDEBAR_WIDTH = 220
const MAX_SIDEBAR_WIDTH = 360
const ARTICLE_LOAD_DELAY = 150

const { page } = useData()
const router = useRouter()
const isChapterPage = computed(() => page.value.relativePath.startsWith('chapters/'))
const chapterSidebarOpen = ref(true)
const chapterSidebarWidth = ref(DEFAULT_SIDEBAR_WIDTH)
const chapterSidebarFooterOffset = ref(0)
const viewportWidth = ref(1440)
const layoutReady = ref(false)
const articleLoadingVisible = ref(false)
let storedSidebarState: string | null = null
let layoutReadyFrame = 0
let footerOffsetFrame = 0
let codeBlockObserver: MutationObserver | null = null
let contentSizeObserver: ResizeObserver | null = null
let articleLoadTimer: ReturnType<typeof setTimeout> | null = null
let activeArticleLoad: string | null = null
let previousBeforePageLoad: Router['onBeforePageLoad']
let previousAfterPageLoad: Router['onAfterPageLoad']
let previousAfterRouteChange: Router['onAfterRouteChange']
let installedBeforePageLoad: Router['onBeforePageLoad']
let installedAfterPageLoad: Router['onAfterPageLoad']
let installedAfterRouteChange: Router['onAfterRouteChange']

const chapterSidebarOverlay = computed(() => viewportWidth.value < 1280)
const layoutClasses = computed(() => ({
  'chapter-reading-layout': isChapterPage.value,
  'chapter-sidebar-open': isChapterPage.value && chapterSidebarOpen.value,
  'chapter-sidebar-overlay': isChapterPage.value && chapterSidebarOverlay.value,
  'chapter-layout-ready': isChapterPage.value && layoutReady.value
}))

const layoutStyle = computed(() => ({
  '--chapter-sidebar-width': `${chapterSidebarWidth.value}px`,
  '--chapter-sidebar-footer-offset': `${chapterSidebarFooterOffset.value}px`
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

function syncSidebarFooterOffset() {
  footerOffsetFrame = 0
  const footer = document.querySelector<HTMLElement>('.VPFooter')
  if (!footer) {
    chapterSidebarFooterOffset.value = 0
    return
  }

  const viewportHeight = document.documentElement.clientHeight
  chapterSidebarFooterOffset.value = Math.max(0, Math.ceil(viewportHeight - footer.getBoundingClientRect().top))
}

function scheduleSidebarFooterOffsetSync() {
  if (!footerOffsetFrame) footerOffsetFrame = requestAnimationFrame(syncSidebarFooterOffset)
}

function handleViewportChange() {
  syncSidebarToViewport()
  scheduleSidebarFooterOffsetSync()
}

function handleWheel(event: WheelEvent) {
  const scroller = event.target instanceof Element
    ? event.target.closest<HTMLElement>('.chapter-sidebar__scroller')
    : null
  if (!scroller || !chapterSidebarOpen.value) return

  if (scroller.scrollHeight <= scroller.clientHeight + 1) return

  const delta = event.deltaMode === 1 ? event.deltaY * 16
    : event.deltaMode === 2 ? event.deltaY * scroller.clientHeight
    : event.deltaY

  const atTop = scroller.scrollTop <= 0
  const atBottom = scroller.scrollTop + scroller.clientHeight >= scroller.scrollHeight - 1
  if ((delta < 0 && atTop) || (delta > 0 && atBottom)) {
    event.preventDefault()
    return
  }

  event.preventDefault()
  scroller.scrollTop += delta
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

function navigationKey(to: string) {
  const target = new URL(to, window.location.href)
  return `${target.pathname}${target.search}`
}

function isArticleTarget(to: string) {
  return /\/chapters(?:\/|$)/.test(new URL(to, window.location.href).pathname)
}

function clearArticleLoadTimer() {
  if (articleLoadTimer) clearTimeout(articleLoadTimer)
  articleLoadTimer = null
}

function beginArticleLoad(to: string) {
  clearArticleLoadTimer()
  articleLoadingVisible.value = false
  activeArticleLoad = isArticleTarget(to) ? navigationKey(to) : null
  if (!activeArticleLoad) return
  const pendingKey = activeArticleLoad
  articleLoadTimer = setTimeout(() => {
    if (activeArticleLoad === pendingKey) articleLoadingVisible.value = true
  }, ARTICLE_LOAD_DELAY)
}

function finishArticleLoad(to: string) {
  if (activeArticleLoad !== navigationKey(to)) return
  clearArticleLoadTimer()
  activeArticleLoad = null
  articleLoadingVisible.value = false
}

function installArticleLoadingHooks() {
  previousBeforePageLoad = router.onBeforePageLoad
  previousAfterPageLoad = router.onAfterPageLoad
  previousAfterRouteChange = router.onAfterRouteChange

  installedBeforePageLoad = async (to) => {
    if ((await previousBeforePageLoad?.(to)) === false) return false
    beginArticleLoad(to)
  }
  installedAfterPageLoad = async (to) => {
    await previousAfterPageLoad?.(to)
  }
  installedAfterRouteChange = async (to) => {
    try {
      await previousAfterRouteChange?.(to)
    } finally {
      await nextTick()
      finishArticleLoad(to)
    }
  }

  router.onBeforePageLoad = installedBeforePageLoad
  router.onAfterPageLoad = installedAfterPageLoad
  router.onAfterRouteChange = installedAfterRouteChange
}

function removeArticleLoadingHooks() {
  clearArticleLoadTimer()
  activeArticleLoad = null
  articleLoadingVisible.value = false
  if (router.onBeforePageLoad === installedBeforePageLoad) router.onBeforePageLoad = previousBeforePageLoad
  if (router.onAfterPageLoad === installedAfterPageLoad) router.onAfterPageLoad = previousAfterPageLoad
  if (router.onAfterRouteChange === installedAfterRouteChange) router.onAfterRouteChange = previousAfterRouteChange
}

onMounted(() => {
  installArticleLoadingHooks()
  enhanceCodeBlocks()
  codeBlockObserver = new MutationObserver(() => enhanceCodeBlocks())
  codeBlockObserver.observe(document.body, { childList: true, subtree: true })
  contentSizeObserver = new ResizeObserver(() => scheduleSidebarFooterOffsetSync())
  contentSizeObserver.observe(document.body)
  readSidebarPreferences()
  syncSidebarToViewport()
  syncSidebarFooterOffset()
  layoutReadyFrame = requestAnimationFrame(() => {
    layoutReady.value = true
  })
  window.addEventListener('resize', handleViewportChange, { passive: true })
  window.addEventListener('scroll', scheduleSidebarFooterOffsetSync, { passive: true })
  window.addEventListener('keydown', handleKeydown)
  document.addEventListener('wheel', handleWheel, { passive: false })
})

onBeforeUnmount(() => {
  removeArticleLoadingHooks()
  codeBlockObserver?.disconnect()
  contentSizeObserver?.disconnect()
  cancelAnimationFrame(layoutReadyFrame)
  cancelAnimationFrame(footerOffsetFrame)
  window.removeEventListener('resize', handleViewportChange)
  window.removeEventListener('scroll', scheduleSidebarFooterOffsetSync)
  window.removeEventListener('keydown', handleKeydown)
  document.removeEventListener('wheel', handleWheel)
})

watch(() => page.value.relativePath, () => requestAnimationFrame(() => {
  enhanceCodeBlocks()
  syncSidebarFooterOffset()
}))
</script>

<template>
  <DefaultTheme.Layout :class="layoutClasses" :style="layoutStyle">
    <template #layout-top>
      <Transition name="article-load-progress">
        <div
          v-if="articleLoadingVisible"
          class="article-load-progress"
          role="progressbar"
          aria-label="正在加载文章"
        >
          <span class="article-load-progress__bar" aria-hidden="true" />
          <span class="article-load-progress__label">正在加载文章</span>
        </div>
      </Transition>
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
