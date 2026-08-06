<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { withBase } from 'vitepress'
import {
  ArrowDown,
  ArrowRight,
  BookOpen,
  ExternalLink,
  Layers3,
} from '@lucide/vue'
import { courseNavigation } from './chapterNavigation'

const modules = computed(() => [
  ...courseNavigation.standalone,
  ...courseNavigation.chapters
])
const chapterCount = courseNavigation.chapters.length
const courseEntryLink = computed(() => courseNavigation.entry?.link ?? '/')
const pathSceneStyle = computed(() => {
  const count = Math.max(modules.value.length, 1)
  return {
    '--course-path-height': `${Math.max(240, count * 70 + 30)}svh`,
    '--course-path-height-mobile': `${Math.max(300, count * 80 + 20)}svh`
  }
})

const pathScene = ref<HTMLElement | null>(null)
const pathCardStage = ref<HTMLElement | null>(null)
const motionReady = ref(false)
const activeModule = ref(0)
const pathPercent = ref(0)

const currentCount = computed(() => String(activeModule.value + 1).padStart(2, '0'))
const totalCount = computed(() => String(modules.value.length).padStart(2, '0'))

let frameId = 0
let resizeObserver: ResizeObserver | undefined
let motionPreference: MediaQueryList | undefined
let cardElements: HTMLElement[] = []

const clamp = (value: number, min = 0, max = 1) => Math.min(max, Math.max(min, value))

function getSceneProgress(element: HTMLElement, navHeight: number, viewportHeight: number) {
  const rect = element.getBoundingClientRect()
  const stageHeight = Math.max(viewportHeight - navHeight, 1)
  const travel = Math.max(rect.height - stageHeight, 1)
  return clamp((navHeight - rect.top) / travel)
}

function updatePath(
  progress: number,
  isMobile: boolean,
  cardStageWidth: number,
  cardStageHeight: number
) {
  const element = pathScene.value
  const moduleCount = modules.value.length
  if (!element || !moduleCount) return

  const scaledProgress = progress * Math.max(moduleCount - 1, 0)
  const nextActive = Math.min(moduleCount - 1, Math.max(0, Math.floor(scaledProgress + 0.5)))
  const nextPercent = Math.round(progress * 100)
  element.style.setProperty('--path-progress-scale', `${progress}`)

  if (activeModule.value !== nextActive) activeModule.value = nextActive
  if (pathPercent.value !== nextPercent) pathPercent.value = nextPercent

  cardElements.forEach((card, index) => {
    const relative = index - scaledProgress
    const distance = Math.min(Math.abs(relative), 1)
    const opacity = relative >= 0
      ? clamp(1 - Math.max(0, relative - 0.55) * 0.95)
      : clamp(1 + relative * 1.65)
    const scale = 1 - distance * 0.055
    let x = 0
    let y = 0

    if (isMobile) {
      y = relative >= 0
        ? relative * Math.min(cardStageHeight * 0.58, 330)
        : relative * Math.min(cardStageHeight * 0.18, 96)
    } else {
      x = relative >= 0
        ? relative * Math.min(cardStageWidth * 0.78, 680)
        : relative * Math.min(cardStageWidth * 0.24, 190)
      y = distance * 12
    }

    card.style.opacity = `${opacity}`
    card.style.transform = `translate3d(${x}px, ${y}px, 0) scale(${scale})`
    card.style.zIndex = `${100 - Math.round(Math.min(Math.abs(relative), 8) * 10)}`
  })
}

function updateScenes() {
  frameId = 0
  if (!motionReady.value || !pathScene.value || !pathCardStage.value) return

  // Read every layout-dependent value before writing any scene styles.
  const viewportHeight = window.innerHeight
  const navHeight = document.querySelector<HTMLElement>('.VPNav')?.getBoundingClientRect().height ?? 0
  const isMobile = window.innerWidth <= 768
  const learningProgress = getSceneProgress(pathScene.value, navHeight, viewportHeight)
  const cardStageRect = pathCardStage.value.getBoundingClientRect()

  updatePath(learningProgress, isMobile, cardStageRect.width, cardStageRect.height)
}

function scheduleUpdate() {
  if (!frameId) frameId = window.requestAnimationFrame(updateScenes)
}

function clearMotionStyles() {
  pathScene.value?.removeAttribute('style')
  cardElements.forEach((card) => card.removeAttribute('style'))
  activeModule.value = 0
  pathPercent.value = 0
}

async function syncMotionPreference() {
  const shouldReduceMotion = motionPreference?.matches ?? false
  motionReady.value = !shouldReduceMotion
  await nextTick()

  cardElements = pathScene.value
    ? Array.from(pathScene.value.querySelectorAll<HTMLElement>('.course-module'))
    : []

  if (shouldReduceMotion) {
    clearMotionStyles()
  } else {
    updateScenes()
  }
}

onMounted(() => {
  motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)')
  motionPreference.addEventListener('change', syncMotionPreference)
  window.addEventListener('scroll', scheduleUpdate, { passive: true })
  window.addEventListener('resize', scheduleUpdate)

  resizeObserver = new ResizeObserver(scheduleUpdate)
  if (pathScene.value) resizeObserver.observe(pathScene.value)

  syncMotionPreference()
})

onBeforeUnmount(() => {
  window.removeEventListener('scroll', scheduleUpdate)
  window.removeEventListener('resize', scheduleUpdate)
  motionPreference?.removeEventListener('change', syncMotionPreference)
  resizeObserver?.disconnect()
  if (frameId) window.cancelAnimationFrame(frameId)
})
</script>

<template>
  <main class="course-home">
    <section class="course-hero" aria-labelledby="course-title">
      <div class="course-hero__stage">
        <img
          class="course-hero__diagram"
          src="/static/LMzxbUGCcoklqoxrqYxcjhdBnYf.png"
          alt=""
          aria-hidden="true"
        >
        <div class="course-shell course-hero__inner">
          <p class="course-kicker">
            <span>SEU · 131 AI CLUB</span>
            <span>OPEN COURSE / 2026</span>
          </p>

          <div class="course-hero__copy">
            <h1 id="course-title">
              <span class="course-wordmark">MS for AI</span>
              <span class="course-title-cn">人工智能缺失的一课</span>
            </h1>
            <div class="course-actions">
              <a class="course-action course-action--primary" :href="withBase(courseEntryLink)">
                开始学习
                <ArrowRight :size="18" :stroke-width="1.8" aria-hidden="true" />
              </a>
              <a class="course-action course-action--quiet" href="https://github.com/131AIClub" target="_blank" rel="noreferrer">
                <ExternalLink :size="18" :stroke-width="1.8" aria-hidden="true" />
                GitHub
              </a>
            </div>
          </div>

          <a class="course-hero__scroll" href="#learning-path">
            <ArrowDown :size="16" :stroke-width="1.8" aria-hidden="true" />
            浏览学习路径
          </a>
        </div>
      </div>
    </section>

    <section
      id="learning-path"
      ref="pathScene"
      class="course-path"
      :class="{ 'is-motion-ready': motionReady }"
      :style="pathSceneStyle"
    >
      <div class="course-path__stage">
        <div class="course-shell course-path__inner">
          <header class="course-section-heading">
            <div>
              <p class="course-section-label">LEARNING PATH</p>
              <h2>课程学习路径</h2>
            </div>
            <p>当前包含 {{ chapterCount }} 个章节，内容按学习顺序排列。</p>
          </header>

          <div class="course-path__status">
            <p>
              <span>{{ currentCount }}</span>
              <span aria-hidden="true">/</span>
              <span>{{ totalCount }}</span>
            </p>
            <div
              class="course-path__progress"
              role="progressbar"
              aria-label="学习路径浏览进度"
              aria-valuemin="0"
              aria-valuemax="100"
              :aria-valuenow="pathPercent"
            >
              <span />
            </div>
          </div>

          <div class="course-path__layout">
            <nav class="course-rail" aria-label="课程章节">
              <span class="course-rail__line" aria-hidden="true">
                <span class="course-rail__fill" />
              </span>
              <a
                v-for="(module, index) in modules"
                :key="module.link"
                :href="withBase(module.link)"
                class="course-rail__node"
                :class="{
                  'is-active': index === activeModule,
                  'is-complete': index < activeModule
                }"
                :aria-current="index === activeModule ? 'step' : undefined"
              >
                <span>{{ module.index }}</span>
                <span>{{ module.text }}</span>
              </a>
            </nav>

            <div ref="pathCardStage" class="course-module-stage">
              <article
                v-for="(module, index) in modules"
                :key="module.link"
                class="course-module"
                :class="{ 'is-active': index === activeModule }"
                :aria-hidden="motionReady && index !== activeModule ? 'true' : undefined"
                :inert="motionReady && index !== activeModule ? true : undefined"
              >
                <div class="course-module__topline">
                  <span class="course-module__index">{{ module.index }}</span>
                  <BookOpen v-if="module.kind === 'standalone'" class="course-module__icon" :size="24" :stroke-width="1.5" aria-hidden="true" />
                  <Layers3 v-else class="course-module__icon" :size="24" :stroke-width="1.5" aria-hidden="true" />
                </div>
                <p class="course-module__meta">{{ module.label }}</p>
                <h3 class="course-module__name">{{ module.text }}</h3>
                <p v-if="module.description" class="course-module__outcome">{{ module.description }}</p>
                <a class="course-module__link" :href="withBase(module.link)">
                  {{ module.kind === 'standalone' ? '阅读前言' : '进入本章' }}
                  <ArrowRight :size="18" :stroke-width="1.7" aria-hidden="true" />
                </a>
              </article>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="course-contact" aria-label="关于与联系">
      <div class="course-shell course-contact__grid">
        <div>
          <p class="course-section-label">ABOUT / CONTACT</p>
          <h2>关于与联系</h2>
        </div>
        <div class="course-contact__item">
          <p class="course-contact__label">MS FOR AI</p>
          <p>由东南大学人工智能协会 131AIClub 维护的开源课程。</p>
          <a class="course-contact__link" href="/about">
            关于我们
            <ArrowRight :size="17" :stroke-width="1.8" aria-hidden="true" />
          </a>
        </div>
        <div class="course-contact__item">
          <p class="course-contact__label">CONTACT</p>
          <p>课程源码与讲义更新发布在 GitHub。</p>
          <a class="course-contact__link" href="https://github.com/131AIClub" target="_blank" rel="noreferrer">
            访问 131AIClub
            <ExternalLink :size="17" :stroke-width="1.8" aria-hidden="true" />
          </a>
        </div>
      </div>
    </section>

  </main>
</template>
