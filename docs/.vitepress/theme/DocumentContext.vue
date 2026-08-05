<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import { ChevronRight } from '@lucide/vue'

const { page } = useData()

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
    <a class="document-context__section" :href="withBase('/chapters/preface')">
      课程讲义
    </a>
    <ChevronRight
      class="document-context__separator"
      :size="14"
      aria-hidden="true"
    />
    <span class="document-context__current" aria-current="page">
      {{ page.title }}
    </span>
    <span v-if="updatedAt" class="document-context__updated">
      最后更新于 {{ updatedAt }}
    </span>
  </nav>
</template>
