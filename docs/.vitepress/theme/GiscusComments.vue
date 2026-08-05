<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useData } from 'vitepress'

const { isDark, page } = useData()
const giscusRoot = ref<HTMLElement>()

const config = {
  repo: import.meta.env.VITE_GISCUS_REPO,
  repoId: import.meta.env.VITE_GISCUS_REPO_ID,
  category: import.meta.env.VITE_GISCUS_CATEGORY,
  categoryId: import.meta.env.VITE_GISCUS_CATEGORY_ID
}

const isConfigured = computed(() => Object.values(config).every(Boolean))

function updateTheme() {
  const frame = giscusRoot.value?.querySelector<HTMLIFrameElement>('.giscus-frame')
  frame?.contentWindow?.postMessage(
    {
      giscus: {
        setConfig: { theme: isDark.value ? 'dark_dimmed' : 'light' }
      }
    },
    'https://giscus.app'
  )
}

async function renderGiscus() {
  if (!isConfigured.value) return
  await nextTick()
  if (!giscusRoot.value) return

  giscusRoot.value.replaceChildren()
  const script = document.createElement('script')
  script.src = 'https://giscus.app/client.js'
  script.async = true
  script.crossOrigin = 'anonymous'
  script.setAttribute('data-repo', config.repo)
  script.setAttribute('data-repo-id', config.repoId)
  script.setAttribute('data-category', config.category)
  script.setAttribute('data-category-id', config.categoryId)
  script.setAttribute('data-mapping', 'pathname')
  script.setAttribute('data-strict', '0')
  script.setAttribute('data-reactions-enabled', '1')
  script.setAttribute('data-emit-metadata', '0')
  script.setAttribute('data-input-position', 'top')
  script.setAttribute('data-theme', isDark.value ? 'dark_dimmed' : 'light')
  script.setAttribute('data-lang', 'zh-CN')
  script.setAttribute('data-loading', 'lazy')
  giscusRoot.value.append(script)
}

watch(isDark, updateTheme)
watch(
  () => page.value.relativePath,
  () => renderGiscus()
)

onMounted(renderGiscus)
onBeforeUnmount(() => giscusRoot.value?.replaceChildren())
</script>

<template>
  <section class="article-comments">
    <p v-if="!isConfigured" class="article-end__status">
      评论区尚未配置。
    </p>
    <div v-else ref="giscusRoot" class="giscus-container">
      <p class="article-end__status">正在加载评论...</p>
    </div>
  </section>
</template>
