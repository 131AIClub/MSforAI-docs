import { mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const repository = '131AIClub/MSforAI-docs'
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const cachePath = resolve(root, 'docs/.vitepress/contributors.authors.json')
const token = process.env.GITHUB_TOKEN

if (!token) {
  console.error('GITHUB_TOKEN is required; contributor cache was not changed.')
  process.exit(1)
}

const cache = JSON.parse(readFileSync(cachePath, 'utf8'))
const headers = {
  Accept: 'application/vnd.github+json',
  Authorization: `Bearer ${token}`,
  'X-GitHub-Api-Version': '2022-11-28',
  'User-Agent': 'MSforAI-docs-contributors-updater'
}

function noreplyLogin(email) {
  const match = email.trim().match(/^(?:\d+\+)?([^@]+)@users\.noreply\.github\.com$/i)
  return match?.[1]
}

function updateIdentity(emailValue, nameValue, loginValue) {
  const email = emailValue.trim().toLowerCase()
  const name = nameValue.trim()
  if (!email || !name) return
  const githubUsername = loginValue || noreplyLogin(email) || cache[email]?.githubUsername
  cache[email] = githubUsername ? { name, githubUsername } : { name }
}

function updateCoAuthors(message) {
  const pattern = /^Co-Authored-By:\s*(.*?)\s*<([^>]+)>\s*$/gim
  for (const match of message.matchAll(pattern)) {
    updateIdentity(match[2], match[1], noreplyLogin(match[2]))
  }
}

async function fetchCommits() {
  const commits = []
  for (let page = 1; ; page += 1) {
    const url = new URL(`https://api.github.com/repos/${repository}/commits`)
    url.searchParams.set('per_page', '100')
    url.searchParams.set('page', String(page))
    const response = await fetch(url, { headers })
    if (!response.ok) throw new Error(`GitHub API returned ${response.status}`)
    const batch = await response.json()
    if (!Array.isArray(batch) || batch.length === 0) break
    commits.push(...batch)
    if (batch.length < 100) break
  }
  return commits
}

try {
  const commits = await fetchCommits()
  for (const item of commits) {
    const author = item?.commit?.author
    if (typeof author?.email === 'string' && typeof author?.name === 'string') {
      updateIdentity(author.email, author.name, item?.author?.login)
    }
    if (typeof item?.commit?.message === 'string') updateCoAuthors(item.commit.message)
  }

  const tempPath = `${cachePath}.${process.pid}.tmp`
  const sortedCache = Object.fromEntries(
    Object.entries(cache).sort(([left], [right]) => left.localeCompare(right))
  )
  mkdirSync(dirname(cachePath), { recursive: true })
  writeFileSync(tempPath, `${JSON.stringify(sortedCache, null, 2)}\n`, 'utf8')
  renameSync(tempPath, cachePath)
  console.log(`Updated ${Object.keys(cache).length} author identities.`)
} catch (error) {
  console.error(`Contributor cache update failed; existing cache was preserved. ${error instanceof Error ? error.message : error}`)
  process.exit(1)
}
