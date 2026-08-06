import { data } from './chapterNavigation.data'

export interface CoursePageItem {
  id: string
  text: string
  link: string
  index: string
  order: number
  label: string
  description?: string
  kind: 'standalone' | 'chapter' | 'article'
}

export interface ArticleNavItem extends CoursePageItem {
  kind: 'article'
}

export interface ChapterNavGroup extends CoursePageItem {
  kind: 'chapter'
  articles: ArticleNavItem[]
}

export interface CourseNavigationData {
  entry?: CoursePageItem
  standalone: CoursePageItem[]
  chapters: ChapterNavGroup[]
}

export const courseNavigation = data
export const chapterNavigation = data.chapters
export const coursePages: CoursePageItem[] = [
  ...courseNavigation.standalone,
  ...chapterNavigation.flatMap((chapter) => [chapter, ...chapter.articles])
]

export function normalizeContentPath(path: string) {
  return path.replace(/index(?:\.html)?$/, '').replace(/\.html$/, '').replace(/\/$/, '')
}

export function isCurrentRoute(path: string, link: string) {
  return normalizeContentPath(path) === normalizeContentPath(link)
}

export function isChapterRoute(path: string, chapter: ChapterNavGroup) {
  const currentPath = normalizeContentPath(path)
  const chapterPath = normalizeContentPath(chapter.link)
  return currentPath === chapterPath || currentPath.startsWith(`${chapterPath}/`)
}

export function findStandaloneByPath(path: string) {
  return courseNavigation.standalone.find((page) => isCurrentRoute(path, page.link))
}

export function findChapterByPath(path: string) {
  return chapterNavigation.find((chapter) => isChapterRoute(path, chapter))
}

export function findArticleByPath(path: string) {
  const currentPath = normalizeContentPath(path)
  const chapter = findChapterByPath(currentPath)
  return chapter?.articles.find((article) => isCurrentRoute(currentPath, article.link))
}

export function findCoursePageByPath(path: string) {
  return coursePages.find((page) => isCurrentRoute(path, page.link))
}

export function findCoursePageNeighbors(path: string) {
  const currentIndex = coursePages.findIndex((page) => isCurrentRoute(path, page.link))
  if (currentIndex < 0) return {}

  return {
    previous: coursePages[currentIndex - 1],
    next: coursePages[currentIndex + 1]
  }
}
