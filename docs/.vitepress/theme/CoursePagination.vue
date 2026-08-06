<script setup lang="ts">
import { computed } from 'vue'
import { ChevronLeft, ChevronRight } from '@lucide/vue'
import { useRoute, withBase } from 'vitepress'
import { findCoursePageNeighbors } from './chapterNavigation'

const route = useRoute()
const neighbors = computed(() => findCoursePageNeighbors(route.path))
</script>

<template>
  <nav
    v-if="neighbors.previous || neighbors.next"
    class="course-pagination"
    aria-label="课程文章导航"
  >
    <a
      v-if="neighbors.previous"
      class="course-pagination__link course-pagination__link--previous"
      :href="withBase(neighbors.previous.link)"
    >
      <ChevronLeft :size="20" :stroke-width="1.8" aria-hidden="true" />
      <span class="course-pagination__copy">
        <span class="course-pagination__label">上一篇</span>
        <span class="course-pagination__title">{{ neighbors.previous.text }}</span>
      </span>
    </a>

    <a
      v-if="neighbors.next"
      class="course-pagination__link course-pagination__link--next"
      :href="withBase(neighbors.next.link)"
    >
      <span class="course-pagination__copy">
        <span class="course-pagination__label">下一篇</span>
        <span class="course-pagination__title">{{ neighbors.next.text }}</span>
      </span>
      <ChevronRight :size="20" :stroke-width="1.8" aria-hidden="true" />
    </a>
  </nav>
</template>
