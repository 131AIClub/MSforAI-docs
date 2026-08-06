<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useData } from 'vitepress'
import { ChevronDown, ChevronUp, ExternalLink, SquarePen } from '@lucide/vue'
import { data as contributorsByPath } from './contributors.data'

const REPOSITORY = '131AIClub/MSforAI-docs'
const DEFAULT_BRANCH = 'master'
const INITIAL_COUNT = 4

interface Contributor {
  key: string
  name: string
  githubUsername?: string
  avatarUrl?: string
  profileUrl?: string
}

const { page, frontmatter } = useData()
const expanded = ref(false)
const failedAvatars = ref<Record<string, boolean>>({})
const contributors = computed<Contributor[]>(
  () => contributorsByPath[page.value.relativePath] || []
)
const repositoryPath = computed(() => `docs/${page.value.relativePath}`)
const historyUrl = computed(
  () =>
    `https://github.com/${REPOSITORY}/commits/${DEFAULT_BRANCH}/${encodeURI(repositoryPath.value)}`
)
const editUrl = computed(
  () =>
    `https://github.com/${REPOSITORY}/edit/${DEFAULT_BRANCH}/${encodeURI(repositoryPath.value)}`
)
const visibleContributors = computed(() =>
  expanded.value ? contributors.value : contributors.value.slice(0, INITIAL_COUNT)
)
const hasMore = computed(() => contributors.value.length > INITIAL_COUNT)

function markAvatarFailed(key: string) {
  failedAvatars.value = { ...failedAvatars.value, [key]: true }
}

function initials(name: string) {
  return name.trim().slice(0, 1).toLocaleUpperCase()
}

watch(
  () => page.value.relativePath,
  () => {
    expanded.value = false
  }
)
</script>

<template>
  <section class="article-contributors" aria-labelledby="contributors-title">
    <div class="article-end__heading">
      <div class="article-end__title-row">
        <div id="contributors-title" role="heading" aria-level="2">本页贡献者</div>
        <a
          v-if="frontmatter.editLink !== false"
          class="article-end__edit-link"
          :href="editUrl"
          target="_blank"
          rel="noopener noreferrer"
        >
          <SquarePen :size="13" aria-hidden="true" />
          在 GitHub 上编辑此页
        </a>
      </div>
      <div class="article-end__meta">
        <span v-if="contributors.length">{{ contributors.length }} 位</span>
        <a :href="historyUrl" target="_blank" rel="noopener noreferrer">
          提交历史
          <ExternalLink :size="12" aria-hidden="true" />
        </a>
      </div>
    </div>

    <p v-if="!contributors.length" class="article-end__status">
      暂无贡献记录。
    </p>

    <template v-else>
      <ul id="contributors-list" class="contributors-list">
        <li v-for="contributor in visibleContributors" :key="contributor.key">
          <a
            v-if="contributor.profileUrl && contributor.avatarUrl && !failedAvatars[contributor.key]"
            class="contributor"
            :href="contributor.profileUrl"
            target="_blank"
            rel="noopener noreferrer"
          >
            <img
              :src="contributor.avatarUrl"
              :alt="`${contributor.name} 的头像`"
              width="34"
              height="34"
              loading="lazy"
              @error="markAvatarFailed(contributor.key)"
            />
            <span>{{ contributor.name }}</span>
          </a>

          <a
            v-else-if="contributor.profileUrl"
            class="contributor"
            :href="contributor.profileUrl"
            target="_blank"
            rel="noopener noreferrer"
          >
            <span class="contributor__fallback" aria-hidden="true">{{ initials(contributor.name) }}</span>
            <span>{{ contributor.name }}</span>
          </a>

          <span v-else class="contributor contributor--plain">
            <span class="contributor__fallback" aria-hidden="true">{{ initials(contributor.name) }}</span>
            <span>{{ contributor.name }}</span>
          </span>
        </li>
      </ul>

      <button
        v-if="hasMore"
        class="contributors-toggle"
        type="button"
        :aria-expanded="expanded"
        aria-controls="contributors-list"
        @click="expanded = !expanded"
      >
        <ChevronUp v-if="expanded" :size="15" aria-hidden="true" />
        <ChevronDown v-else :size="15" aria-hidden="true" />
        {{ expanded ? '收起' : `显示全部 ${contributors.length} 位` }}
      </button>
    </template>
  </section>
</template>
