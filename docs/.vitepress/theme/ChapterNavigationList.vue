<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vitepress'
import { chapterNavigation } from './chapterNavigation'

defineProps<{
  compact?: boolean
}>()

const emit = defineEmits<{
  navigate: []
}>()

const route = useRoute()

function normalizePath(path: string) {
  return path.replace(/\.html$/, '').replace(/\/$/, '')
}

const currentPath = computed(() => normalizePath(route.path))

function isCurrent(link: string) {
  return currentPath.value === normalizePath(link)
}
</script>

<template>
  <nav
    class="chapter-navigation-list"
    :class="{ 'chapter-navigation-list--compact': compact }"
    aria-label="讲义章节"
  >
    <a
      v-for="chapter in chapterNavigation"
      :key="chapter.link"
      class="chapter-navigation-item"
      :class="{ 'is-current': isCurrent(chapter.link) }"
      :href="chapter.link"
      :title="chapter.text"
      :aria-current="isCurrent(chapter.link) ? 'page' : undefined"
      @click="emit('navigate')"
    >
      <span class="chapter-navigation-item__index" aria-hidden="true">
        {{ chapter.index }}
      </span>
      <span class="chapter-navigation-item__text">{{ chapter.text }}</span>
    </a>
  </nav>
</template>
