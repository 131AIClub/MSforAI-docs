<script setup lang="ts">
import { PanelLeftClose, PanelLeftOpen } from '@lucide/vue'

defineProps<{
  open: boolean
}>()

const emit = defineEmits<{
  toggle: []
}>()
</script>

<template>
  <button
    class="chapter-sidebar-toggle"
    type="button"
    :title="open ? '收起讲义章节' : '展开讲义章节'"
    :aria-label="open ? '收起讲义章节' : '展开讲义章节'"
    aria-controls="chapter-sidebar"
    :aria-expanded="open"
    @click="emit('toggle')"
  >
    <PanelLeftClose v-if="open" :size="20" :stroke-width="1.7" aria-hidden="true" />
    <PanelLeftOpen v-else :size="20" :stroke-width="1.7" aria-hidden="true" />
  </button>
</template>
