import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createContentLoader, type ContentData } from 'vitepress'

interface CachedAuthor {
  name: string
  githubUsername?: string
  avatarUrl?: string
  profileUrl?: string
}

interface Contributor {
  key: string
  name: string
  githubUsername?: string
  avatarUrl?: string
  profileUrl?: string
}

export type ContributorsByPath = Record<string, Contributor[]>

interface Identity {
  name: string
  email: string
}

interface Participation {
  identity: Identity
  timestamp: string
}

const themeDirectory = dirname(fileURLToPath(import.meta.url))
const repositoryRoot = resolve(themeDirectory, '../../..')
const authorsPath = resolve(themeDirectory, '../contributors.authors.json')
const recordSeparator = '\u001e'
const fieldSeparator = '\u001f'

function metadataError(source: string, message: string): never {
  throw new Error(`[贡献者] ${source}：${message}`)
}

function relativePathFromUrl(url: string) {
  if (url.endsWith('/')) return `${url.replace(/^\//, '')}index.md`
  return `${url.replace(/\.html$/, '').replace(/^\//, '')}.md`
}

function parseHistorical(frontmatter: Record<string, unknown>, source: string) {
  const value = frontmatter.contributors
  if (value == null) return []
  if (!Array.isArray(value)) metadataError(source, 'contributors 必须是数组')
  return value.map((entry, index) => {
    if (typeof entry !== 'string' || !entry.trim()) {
      metadataError(`${source} contributors[${index}]`, '必须是 GitHub 用户名（非空字符串）')
    }
    return entry.trim()
  })
}

function displayHistorical(username: string): Contributor {
  return {
    key: `github:${username.toLowerCase()}`,
    name: username,
    githubUsername: username,
    avatarUrl: `https://avatars.githubusercontent.com/${encodeURIComponent(username)}?size=68`,
    profileUrl: `https://github.com/${encodeURIComponent(username)}`
  }
}

function runGit(args: string[]) {
  try {
    return execFileSync('git', args, {
      cwd: repositoryRoot,
      encoding: 'utf8',
      maxBuffer: 16 * 1024 * 1024
    })
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error)
    throw new Error(`无法读取 Git 历史。请确认在完整 Git 仓库中构建（fetch-depth: 0）。\n${detail}`)
  }
}

function assertRepositoryHistory() {
  if (!existsSync(resolve(repositoryRoot, '.git'))) {
    throw new Error('贡献者数据需要完整 Git 历史，但未找到 .git。请使用完整克隆（fetch-depth: 0）后再构建。')
  }

  if (runGit(['rev-parse', '--is-shallow-repository']).trim() === 'true') {
    throw new Error('贡献者数据需要完整 Git 历史；当前仓库是浅克隆。请使用 fetch-depth: 0 后再构建。')
  }
}

function parseIdentity(value: string): Identity | null {
  const match = value.trim().match(/^(.*?)\s*<([^>]+)>$/)
  if (!match) return null
  const name = match[1].trim()
  const email = match[2].trim().toLowerCase()
  return name && email ? { name, email } : null
}

function parseParticipations(output: string) {
  const participations: Participation[] = []
  for (const record of output.split(recordSeparator)) {
    const fields = record.trim().split(fieldSeparator)
    if (fields.length < 4) continue
    const [name, email, timestamp, body] = fields
    const author = name.trim() && email.trim()
      ? { name: name.trim(), email: email.trim().toLowerCase() }
      : null
    if (author) participations.push({ identity: author, timestamp })

    const coAuthorPattern = /^Co-Authored-By:\s*(.*?)\s*<([^>]+)>\s*$/gim
    for (const match of body.matchAll(coAuthorPattern)) {
      const coAuthor = parseIdentity(`${match[1]} <${match[2]}>`)
      if (coAuthor) participations.push({ identity: coAuthor, timestamp })
    }
  }
  return participations
}

function displayContributor(identity: Identity, authors: Record<string, CachedAuthor>) {
  const cached = authors[identity.email]
  const githubUsername = cached?.githubUsername
  const name = githubUsername || cached?.name || identity.name
  const key = githubUsername
    ? `github:${githubUsername.toLowerCase()}`
    : `author:${createHash('sha256').update(identity.email).digest('hex').slice(0, 12)}`
  return {
    key,
    name,
    ...(githubUsername
      ? {
          githubUsername,
          avatarUrl: `https://avatars.githubusercontent.com/${encodeURIComponent(githubUsername)}?size=68`,
          profileUrl: `https://github.com/${encodeURIComponent(githubUsername)}`
        }
      : {}),
    ...(cached?.avatarUrl ? { avatarUrl: cached.avatarUrl } : {}),
    ...(cached?.profileUrl ? { profileUrl: cached.profileUrl } : {})
  } satisfies Contributor
}

function loadAuthors() {
  return JSON.parse(readFileSync(authorsPath, 'utf8')) as Record<string, CachedAuthor>
}

function stagedRenameSource(file: string) {
  const repositoryPath = relative(repositoryRoot, file).split('\\').join('/')
  const fields = runGit([
    'diff',
    '--cached',
    '--name-status',
    '--find-renames',
    '-z'
  ]).split('\0')

  for (let index = 0; index < fields.length;) {
    const status = fields[index++]
    if (!status) continue
    if (status.startsWith('R') || status.startsWith('C')) {
      const source = fields[index++]
      const destination = fields[index++]
      if (destination === repositoryPath) return source
    } else {
      index += 1
    }
  }

  return null
}

function fileHistory(file: string) {
  const format = `--format=%an${fieldSeparator}%ae${fieldSeparator}%aI${fieldSeparator}%B${recordSeparator}`
  const output = runGit(['log', '--follow', format, '--', file])
  if (output.trim()) return output

  const renameSource = stagedRenameSource(file)
  return renameSource
    ? runGit(['log', '--follow', format, '--', renameSource])
    : output
}

export default createContentLoader<ContributorsByPath>('chapters/**/*.md', {
  transform(rawData: ContentData[]): ContributorsByPath {
    assertRepositoryHistory()
    const authors = loadAuthors()
    const result: ContributorsByPath = {}

    for (const entry of rawData) {
      const relativePath = relativePathFromUrl(entry.url)
      const file = resolve(repositoryRoot, 'docs', relativePath)
      const output = fileHistory(file)
      const latestByKey = new Map<string, { participation: Participation; contributor: Contributor }>()
      for (const participation of parseParticipations(output)) {
        const contributor = displayContributor(participation.identity, authors)
        const existing = latestByKey.get(contributor.key)
        if (!existing || participation.timestamp > existing.participation.timestamp) {
          latestByKey.set(contributor.key, { participation, contributor })
        }
      }

      for (const username of parseHistorical(entry.frontmatter, relativePath)) {
        const contributor = displayHistorical(username)
        if (!latestByKey.has(contributor.key)) {
          latestByKey.set(contributor.key, {
            participation: { identity: { name: contributor.name, email: '' }, timestamp: '' },
            contributor
          })
        }
      }

      result[relativePath] = [...latestByKey.values()]
        .sort((a, b) => b.participation.timestamp.localeCompare(a.participation.timestamp))
        .map(({ contributor }) => contributor)
    }

    return result
  }
})
