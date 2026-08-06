<script setup lang="ts">
import { computed } from 'vue'
import {
  BadgeAlert,
  Info,
  Lightbulb,
  OctagonAlert,
  TriangleAlert
} from '@lucide/vue'

const alertIcons = {
  note: Info,
  tip: Lightbulb,
  important: BadgeAlert,
  warning: TriangleAlert,
  caution: OctagonAlert
} as const

type AlertType = keyof typeof alertIcons

const props = defineProps<{
  type: AlertType
  title: string
}>()

const icon = computed(() => alertIcons[props.type] ?? Info)
</script>

<template>
  <aside
    :class="['ms-alert', `ms-alert--${type}`]"
    role="note"
    :aria-label="title"
  >
    <div class="ms-alert__title">
      <component :is="icon" :size="18" :stroke-width="2" aria-hidden="true" />
      <span>{{ title }}</span>
    </div>
    <div class="ms-alert__body">
      <slot />
    </div>
  </aside>
</template>
