<script setup lang="ts">
import { computed } from 'vue'
import { useData, useRoute, withBase } from 'vitepress'
import { ChevronRight } from '@lucide/vue'
import {
  courseNavigation,
  findArticleByPath,
  findChapterByPath,
  findStandaloneByPath,
  normalizeContentPath
} from './chapterNavigation'

const { page } = useData()
const route = useRoute()
const currentChapter = computed(() => findChapterByPath(route.path))
const currentArticle = computed(() => findArticleByPath(route.path))
const currentStandalone = computed(() => findStandaloneByPath(route.path))
const courseEntryLink = computed(() => courseNavigation.entry?.link ?? '/')
const isChapterIndex = computed(() => {
  const chapter = currentChapter.value
  return chapter
    ? normalizeContentPath(route.path) === normalizeContentPath(chapter.link)
    : false
})

const updatedAt = computed(() => {
  const value = page.value.lastUpdated
  if (!value) return ''

  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    timeZone: 'Asia/Shanghai'
  }).format(new Date(Number(value)))
})
</script>

<template>
  <nav class="document-context" aria-label="文档位置">
    <a class="document-context__section" :href="withBase(courseEntryLink)">
      课程讲义
    </a>
    <template v-if="currentStandalone">
      <ChevronRight
        class="document-context__separator"
        :size="14"
        aria-hidden="true"
      />
      <span class="document-context__current" aria-current="page">
        {{ currentStandalone.text }}
      </span>
    </template>
    <template v-if="currentChapter">
      <ChevronRight
        class="document-context__separator"
        :size="14"
        aria-hidden="true"
      />
      <span v-if="isChapterIndex" class="document-context__current" aria-current="page">
        {{ currentChapter.text }}
      </span>
      <a v-else class="document-context__section" :href="withBase(currentChapter.link)">
        {{ currentChapter.text }}
      </a>
      <template v-if="!isChapterIndex">
        <ChevronRight
          class="document-context__separator"
          :size="14"
          aria-hidden="true"
        />
        <span class="document-context__current" aria-current="page">
          {{ currentArticle?.text ?? page.title }}
        </span>
      </template>
    </template>
    <span v-if="updatedAt" class="document-context__updated">
      最后更新于 {{ updatedAt }}
    </span>
  </nav>
</template>
