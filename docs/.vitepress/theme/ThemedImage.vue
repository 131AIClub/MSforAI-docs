<script setup lang="ts">
import { computed } from 'vue'
import { useData } from 'vitepress'

const props = withDefaults(defineProps<{ src: string; dark?: string; alt?: string }>(), { alt: '' })
const { isDark } = useData()
const resolvedSrc = computed(() => (isDark.value && props.dark ? props.dark : props.src))
</script>

<template>
  <img :src="resolvedSrc" :alt="alt" />
</template>
