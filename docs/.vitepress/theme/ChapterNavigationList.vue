<script setup lang="ts">
import { ChevronRight } from '@lucide/vue'
import { computed, ref, watch } from 'vue'
import { useRoute, withBase } from 'vitepress'
import {
  chapterNavigation,
  courseNavigation,
  findChapterByPath,
  isChapterRoute,
  isCurrentRoute,
  normalizeContentPath
} from './chapterNavigation'

defineProps<{
  compact?: boolean
}>()

const emit = defineEmits<{
  navigate: []
}>()

const route = useRoute()
const expandedChapters = ref(new Set<string>())
const currentPath = computed(() => normalizeContentPath(route.path))

function isCurrent(link: string) {
  return isCurrentRoute(currentPath.value, link)
}

function isExpanded(chapterLink: string) {
  return expandedChapters.value.has(chapterLink)
}

function toggleChapter(chapterLink: string) {
  const nextExpandedChapters = new Set(expandedChapters.value)
  if (nextExpandedChapters.has(chapterLink)) {
    nextExpandedChapters.delete(chapterLink)
  } else {
    nextExpandedChapters.add(chapterLink)
  }
  expandedChapters.value = nextExpandedChapters
}

watch(
  currentPath,
  (path) => {
    const activeChapter = findChapterByPath(path)
    if (activeChapter?.articles.length) {
      expandedChapters.value = new Set([...expandedChapters.value, activeChapter.link])
    }
  },
  { immediate: true }
)
</script>

<template>
  <nav
    class="chapter-navigation-list"
    :class="{ 'chapter-navigation-list--compact': compact }"
    aria-label="讲义章节"
  >
    <a
      v-for="page in courseNavigation.standalone"
      :key="page.link"
      class="chapter-navigation-item chapter-navigation-item--standalone"
      :class="{ 'is-current': isCurrent(page.link) }"
      :href="withBase(page.link)"
      :title="page.text"
      :aria-current="isCurrent(page.link) ? 'page' : undefined"
      @click="emit('navigate')"
    >
      <span class="chapter-navigation-item__index" aria-hidden="true">
        {{ page.index }}
      </span>
      <span class="chapter-navigation-item__text">{{ page.text }}</span>
    </a>

    <div
      v-for="chapter in chapterNavigation"
      :key="chapter.link"
      class="chapter-navigation-group"
      :class="{
        'is-active-chapter': isChapterRoute(currentPath, chapter),
        'is-expanded': isExpanded(chapter.link)
      }"
    >
      <div
        class="chapter-navigation-row"
        :class="{ 'is-current': isChapterRoute(currentPath, chapter) }"
      >
        <a
          class="chapter-navigation-item"
          :class="{ 'is-current': isCurrent(chapter.link) }"
          :href="withBase(chapter.link)"
          :title="chapter.text"
          :aria-current="isCurrent(chapter.link) ? 'page' : undefined"
          @click="emit('navigate')"
        >
          <span class="chapter-navigation-item__index" aria-hidden="true">
            {{ chapter.index }}
          </span>
          <span class="chapter-navigation-item__text">{{ chapter.text }}</span>
        </a>

        <button
          v-if="chapter.articles.length"
          class="chapter-navigation-toggle"
          type="button"
          :title="isExpanded(chapter.link) ? `收起${chapter.text}文章` : `展开${chapter.text}文章`"
          :aria-label="isExpanded(chapter.link) ? `收起${chapter.text}文章` : `展开${chapter.text}文章`"
          :aria-expanded="isExpanded(chapter.link)"
          @click="toggleChapter(chapter.link)"
        >
          <ChevronRight :size="15" :stroke-width="1.8" aria-hidden="true" />
        </button>
      </div>

      <div
        v-if="chapter.articles.length"
        class="chapter-article-collapse"
        :aria-hidden="!isExpanded(chapter.link)"
        :inert="isExpanded(chapter.link) ? undefined : ''"
      >
        <div class="chapter-article-collapse__inner">
          <div class="chapter-article-list" role="list">
            <a
              v-for="article in chapter.articles"
              :key="article.link"
              class="chapter-article-item"
              :class="{ 'is-current': isCurrent(article.link) }"
              :href="withBase(article.link)"
              :title="article.text"
              :aria-current="isCurrent(article.link) ? 'page' : undefined"
              role="listitem"
              @click="emit('navigate')"
            >
              {{ article.text }}
            </a>
          </div>
        </div>
      </div>
    </div>
  </nav>
</template>
