<script setup lang="ts">
import { computed } from 'vue'
import { ArrowRight } from '@lucide/vue'
import { useRoute, withBase } from 'vitepress'
import {
  findChapterByPath,
  isCurrentRoute
} from './chapterNavigation'

const route = useRoute()
const chapter = computed(() => findChapterByPath(route.path))
const visible = computed(() => {
  const current = chapter.value
  return Boolean(
    current &&
    current.articles.length &&
    isCurrentRoute(route.path, current.link)
  )
})
</script>

<template>
  <section
    v-if="visible && chapter"
    class="chapter-article-directory"
    aria-labelledby="chapter-article-directory-title"
  >
    <header class="chapter-article-directory__header">
      <h2 id="chapter-article-directory-title">本章内容</h2>
      <span>{{ chapter.articles.length }} 篇</span>
    </header>

    <ol class="chapter-article-directory__list">
      <li v-for="article in chapter.articles" :key="article.link">
        <a :href="withBase(article.link)" class="chapter-article-directory__link">
          <span class="chapter-article-directory__index" aria-hidden="true">
            {{ article.index }}
          </span>
          <span class="chapter-article-directory__copy">
            <strong>{{ article.text }}</strong>
            <span v-if="article.description">{{ article.description }}</span>
          </span>
          <ArrowRight :size="17" :stroke-width="1.7" aria-hidden="true" />
        </a>
      </li>
    </ol>
  </section>
</template>
