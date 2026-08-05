<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useData } from 'vitepress'
import { ChevronDown, ChevronUp, ExternalLink } from '@lucide/vue'

const REPOSITORY = '131AIClub/MSforAI-docs'
const DEFAULT_BRANCH = 'master'
const INITIAL_COUNT = 4

interface GitHubCommit {
  author: {
    login: string
    avatar_url: string
    html_url: string
  } | null
  commit: {
    author: {
      name: string
    } | null
  }
}

interface Contributor {
  key: string
  name: string
  avatarUrl?: string
  profileUrl?: string
}

type LoadState = 'idle' | 'loading' | 'success' | 'error'

const { page } = useData()
const contributors = ref<Contributor[]>([])
const loadState = ref<LoadState>('idle')
const expanded = ref(false)
let requestController: AbortController | undefined

const relativePath = computed(() => page.value.relativePath)
const repositoryPath = computed(() => `docs/${relativePath.value}`)
const historyUrl = computed(
  () =>
    `https://github.com/${REPOSITORY}/commits/${DEFAULT_BRANCH}/${encodeURI(repositoryPath.value)}`
)
const visibleContributors = computed(() =>
  expanded.value ? contributors.value : contributors.value.slice(0, INITIAL_COUNT)
)
const hasMore = computed(() => contributors.value.length > INITIAL_COUNT)

function cacheKey(path: string) {
  return `msforai:contributors:${path}`
}

function deduplicate(commits: GitHubCommit[]) {
  const unique = new Map<string, Contributor>()

  for (const item of commits) {
    if (item.author) {
      const key = `github:${item.author.login.toLowerCase()}`
      if (!unique.has(key)) {
        unique.set(key, {
          key,
          name: item.author.login,
          avatarUrl: item.author.avatar_url,
          profileUrl: item.author.html_url
        })
      }
      continue
    }

    const name = item.commit.author?.name?.trim()
    if (!name) continue
    const key = `name:${name.toLocaleLowerCase()}`
    if (!unique.has(key)) unique.set(key, { key, name })
  }

  return [...unique.values()]
}

async function fetchAllCommits(path: string, signal: AbortSignal) {
  const commits: GitHubCommit[] = []
  let apiUrl: URL | null = new URL(
    `https://api.github.com/repos/${REPOSITORY}/commits`
  )
  apiUrl.searchParams.set('path', path)
  apiUrl.searchParams.set('per_page', '100')

  while (apiUrl) {
    const response = await fetch(apiUrl, {
      signal,
      headers: {
        Accept: 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28'
      }
    })

    if (!response.ok) throw new Error(`GitHub API returned ${response.status}`)
    commits.push(...((await response.json()) as GitHubCommit[]))

    const nextMatch = response.headers
      .get('Link')
      ?.match(/<([^>]+)>;\s*rel="next"/)
    apiUrl = nextMatch ? new URL(nextMatch[1]) : null
  }

  return commits
}

async function loadContributors(path: string) {
  requestController?.abort()
  requestController = new AbortController()
  expanded.value = false
  contributors.value = []

  if (typeof window === 'undefined' || !path.endsWith('.md')) {
    loadState.value = 'idle'
    return
  }

  const pathWithDocs = `docs/${path}`
  let stored: string | null = null
  try {
    stored = sessionStorage.getItem(cacheKey(pathWithDocs))
  } catch {
    // Continue without a cache when browser storage is unavailable.
  }
  if (stored) {
    try {
      contributors.value = JSON.parse(stored) as Contributor[]
      loadState.value = 'success'
      return
    } catch {
      sessionStorage.removeItem(cacheKey(pathWithDocs))
    }
  }

  loadState.value = 'loading'
  try {
    const commits = await fetchAllCommits(pathWithDocs, requestController.signal)
    contributors.value = deduplicate(commits)
    try {
      sessionStorage.setItem(
        cacheKey(pathWithDocs),
        JSON.stringify(contributors.value)
      )
    } catch {
      // A successful API response should still render when caching fails.
    }
    loadState.value = 'success'
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') return
    loadState.value = 'error'
  }
}

watch(relativePath, loadContributors, { immediate: true })

onBeforeUnmount(() => requestController?.abort())
</script>

<template>
  <section class="article-contributors" aria-labelledby="contributors-title">
    <div class="article-end__heading">
      <div id="contributors-title" role="heading" aria-level="2">本页贡献者</div>
      <span v-if="loadState === 'success' && contributors.length">
        {{ contributors.length }} 位
      </span>
    </div>

    <p v-if="loadState === 'loading'" class="article-end__status">
      正在加载贡献者...
    </p>

    <p v-else-if="loadState === 'error'" class="article-end__status">
      暂时无法加载贡献者。
      <a :href="historyUrl" target="_blank" rel="noopener noreferrer">
        查看提交历史
        <ExternalLink :size="13" aria-hidden="true" />
      </a>
    </p>

    <p
      v-else-if="loadState === 'success' && !contributors.length"
      class="article-end__status"
    >
      暂无贡献记录。
    </p>

    <template v-else-if="loadState === 'success'">
      <ul id="contributors-list" class="contributors-list">
        <li v-for="contributor in visibleContributors" :key="contributor.key">
          <a
            v-if="contributor.profileUrl"
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
            />
            <span>{{ contributor.name }}</span>
          </a>

          <span v-else class="contributor contributor--plain">
            <span class="contributor__fallback" aria-hidden="true">
              {{ contributor.name.slice(0, 1).toLocaleUpperCase() }}
            </span>
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
