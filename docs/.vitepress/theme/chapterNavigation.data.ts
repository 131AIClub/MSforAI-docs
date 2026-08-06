import { readdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createContentLoader, type ContentData } from 'vitepress'
import type {
  ArticleNavItem,
  ChapterNavGroup,
  CourseNavigationData,
  CoursePageItem
} from './chapterNavigation'

interface PageMetadata {
  title?: unknown
  order?: unknown
  index?: unknown
  label?: unknown
  description?: unknown
  courseEntry?: unknown
}

interface ParsedPage {
  entry: ContentData
  id: string
  source: string
  title: string
  explicitOrder?: number
  inferredOrder?: number
  explicitIndex?: string
  label?: string
  description?: string
  courseEntry: boolean
}

declare const data: CourseNavigationData
export { data }

const naturalCollator = new Intl.Collator('zh-CN', {
  numeric: true,
  sensitivity: 'base'
})
const chaptersDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '../../chapters')

function metadataError(source: string, message: string): never {
  throw new Error(`[课程结构] ${source}：${message}`)
}

function optionalString(value: unknown, field: string, source: string) {
  if (value == null) return undefined
  if (typeof value !== 'string' || !value.trim()) {
    metadataError(source, `${field} 必须是非空字符串`)
  }
  return value.trim()
}

function optionalOrder(value: unknown, source: string) {
  if (value == null) return undefined
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    metadataError(source, 'order 必须是非负整数')
  }
  return value
}

function optionalEntry(value: unknown, source: string) {
  if (value == null) return false
  if (typeof value !== 'boolean') metadataError(source, 'courseEntry 必须是布尔值')
  return value
}

function stripFrontmatter(source = '') {
  if (!source.startsWith('---\n')) return source
  const end = source.indexOf('\n---\n', 4)
  return end < 0 ? source : source.slice(end + 5)
}

function humanize(value: string) {
  return value.replace(/[-_]+/g, ' ').replace(/\s+/g, ' ').trim()
}

function inferNumber(value: string) {
  const match = value.match(/(?:^|\D)(\d+)(?:\D|$)/)
  return match ? Number(match[1]) : undefined
}

function titleFromSource(source: string) {
  return stripFrontmatter(source).match(/^#\s+(.+)$/m)?.[1]?.trim()
}

function descriptionFromSource(source: string) {
  const body = stripFrontmatter(source)
  const paragraphs = body.split(/\n\s*\n/)
  for (const paragraph of paragraphs) {
    const value = paragraph.trim()
    if (
      !value ||
      /^(?:#{1,6}|```|~~~|>|[-*+]\s|\d+\.\s|<|\[.*\]:)/.test(value)
    ) continue

    const plain = value
      .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
      .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
      .replace(/[*_~`]/g, '')
      .replace(/\s+/g, ' ')
      .trim()
    if (plain) return plain.length > 120 ? `${plain.slice(0, 117)}…` : plain
  }
  return undefined
}

function parsePage(entry: ContentData, id: string, source: string): ParsedPage {
  const metadata = entry.frontmatter as PageMetadata
  const explicitTitle = optionalString(metadata.title, 'title', source)
  const explicitIndex = optionalString(metadata.index, 'index', source)
  const label = optionalString(metadata.label, 'label', source)
  const explicitDescription = optionalString(metadata.description, 'description', source)
  const sourceBody = entry.src ?? ''

  return {
    entry,
    id,
    source,
    title: explicitTitle ?? titleFromSource(sourceBody) ?? humanize(id),
    explicitOrder: optionalOrder(metadata.order, source),
    inferredOrder: inferNumber(id),
    explicitIndex,
    label,
    description: explicitDescription ?? descriptionFromSource(sourceBody),
    courseEntry: optionalEntry(metadata.courseEntry, source)
  }
}

function sortPages(a: ParsedPage, b: ParsedPage) {
  const aOrder = a.explicitOrder ?? a.inferredOrder
  const bOrder = b.explicitOrder ?? b.inferredOrder
  if (aOrder != null && bOrder != null && aOrder !== bOrder) return aOrder - bOrder
  if (aOrder != null && bOrder == null) return -1
  if (aOrder == null && bOrder != null) return 1
  return naturalCollator.compare(a.id, b.id)
}

function assertUniqueExplicitOrders(pages: ParsedPage[], scope: string) {
  const orders = new Map<number, string>()
  for (const page of pages) {
    if (page.explicitOrder == null) continue
    const existing = orders.get(page.explicitOrder)
    if (existing) {
      metadataError(
        page.source,
        `${scope} order “${page.explicitOrder}” 与 ${existing} 重复`
      )
    }
    orders.set(page.explicitOrder, page.source)
  }
}

function displayIndex(page: ParsedPage, position: number) {
  const number = page.inferredOrder ?? position + 1
  return page.explicitIndex ?? String(number).padStart(2, '0')
}

function standaloneItem(page: ParsedPage, position: number): CoursePageItem {
  return {
    id: page.id,
    text: page.title,
    link: `/chapters/${page.id}`,
    index: displayIndex(page, position),
    order: page.explicitOrder ?? page.inferredOrder ?? position,
    label: page.label ?? 'COURSE NOTE',
    description: page.description,
    kind: 'standalone'
  }
}

function articleItem(page: ParsedPage, chapterId: string, position: number): ArticleNavItem {
  return {
    id: page.id,
    text: page.title,
    link: `/chapters/${chapterId}/${page.id}`,
    index: displayIndex(page, position),
    order: page.explicitOrder ?? page.inferredOrder ?? position,
    label: page.label ?? 'ARTICLE',
    description: page.description,
    kind: 'article'
  }
}

export default createContentLoader('chapters/**/*.md', {
  includeSrc: true,
  transform(rawData: ContentData[]): CourseNavigationData {
    const standalonePages: ParsedPage[] = []
    const chapterIndexes = new Map<string, ParsedPage>()
    const articlePages = new Map<string, ParsedPage[]>()

    for (const entry of rawData) {
      const cleanUrl = entry.url.replace(/\.html$/, '')
      const segments = cleanUrl.split('/').filter(Boolean)
      if (segments[0] !== 'chapters') continue

      if (segments.length === 2 && !cleanUrl.endsWith('/')) {
        const id = segments[1]
        standalonePages.push(parsePage(entry, id, `chapters/${id}.md`))
        continue
      }

      if (segments.length === 2 && cleanUrl.endsWith('/')) {
        const id = segments[1]
        if (chapterIndexes.has(id)) metadataError(`chapters/${id}/index.md`, '章节首页重复')
        chapterIndexes.set(id, parsePage(entry, id, `chapters/${id}/index.md`))
        continue
      }

      if (segments.length === 3) {
        const [, chapterId, articleId] = segments
        const pages = articlePages.get(chapterId) ?? []
        pages.push(parsePage(entry, articleId, `chapters/${chapterId}/${articleId}.md`))
        articlePages.set(chapterId, pages)
        continue
      }

      metadataError(
        cleanUrl.replace(/^\//, ''),
        '只支持 chapters/<chapter>/*.md，不支持更深层文章目录'
      )
    }

    for (const chapterId of articlePages.keys()) {
      if (!chapterIndexes.has(chapterId)) {
        metadataError(
          `chapters/${chapterId}`,
          '目录包含文章但缺少 index.md，无法创建有效章节入口'
        )
      }
    }

    for (const directory of readdirSync(chaptersDirectory, { withFileTypes: true })) {
      if (!directory.isDirectory()) continue
      const contents = readdirSync(resolve(chaptersDirectory, directory.name))
      if (contents.length && !chapterIndexes.has(directory.name)) {
        metadataError(
          `chapters/${directory.name}`,
          '非空章节目录缺少 index.md，无法创建有效章节入口'
        )
      }
    }

    const sortedChapterPages = [...chapterIndexes.values()].sort(sortPages)
    assertUniqueExplicitOrders(sortedChapterPages, '章节')
    assertUniqueExplicitOrders(standalonePages, '独立页面')
    standalonePages.sort(sortPages)

    const standalone = standalonePages.map(standaloneItem)
    const chapters: ChapterNavGroup[] = sortedChapterPages.map((page, position) => {
      const pages = (articlePages.get(page.id) ?? []).sort(sortPages)
      assertUniqueExplicitOrders(pages, `章节 ${page.title} 的文章`)
      return {
        id: page.id,
        text: page.title,
        link: `/chapters/${page.id}/`,
        index: displayIndex(page, position),
        order: page.explicitOrder ?? page.inferredOrder ?? position,
        label: page.label ?? 'CHAPTER',
        description: page.description,
        kind: 'chapter',
        articles: pages.map((article, articlePosition) =>
          articleItem(article, page.id, articlePosition)
        )
      }
    })

    const entryPages = [...standalonePages, ...sortedChapterPages].filter(
      (page) => page.courseEntry
    )
    if (entryPages.length > 1) {
      metadataError(entryPages[1].source, `courseEntry 与 ${entryPages[0].source} 重复`)
    }

    const entryPage = entryPages[0]
    const entry = entryPage
      ? standalone.find((item) => item.id === entryPage.id) ??
        chapters.find((item) => item.id === entryPage.id)
      : standalone[0] ?? chapters[0]

    return { entry, standalone, chapters }
  }
})
