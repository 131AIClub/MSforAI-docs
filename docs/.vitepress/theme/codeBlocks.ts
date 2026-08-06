const CODE_BLOCK_SELECTOR = '.vp-doc div[class*="language-"]'
const INITIAL_COLLAPSE_LINES = 20

const languageNames: Record<string, string> = {
  bash: 'Bash',
  cpp: 'C++',
  css: 'CSS',
  html: 'HTML',
  javascript: 'JavaScript',
  js: 'JavaScript',
  json: 'JSON',
  markdown: 'Markdown',
  md: 'Markdown',
  powershell: 'PowerShell',
  python: 'Python',
  py: 'Python',
  shell: 'Shell',
  sh: 'Shell',
  sql: 'SQL',
  typescript: 'TypeScript',
  ts: 'TypeScript',
  vue: 'Vue'
}

function getLanguage(block: HTMLElement) {
  const match = [...block.classList].find((name) => name.startsWith('language-'))
  return match ? match.slice('language-'.length) : ''
}

function enhanceCodeBlock(block: HTMLElement) {
  if (block.dataset.codeEnhanced === 'true') return
  const code = block.querySelector('pre code')
  if (!code) return

  const language = getLanguage(block)
  const label = block.dataset.codeTitle || languageNames[language.toLowerCase()] || (language ? language : 'Code')
  const lineCount = code.querySelectorAll('.line').length || code.textContent?.split('\n').length || 1
  const codeId = `code-block-${Math.random().toString(36).slice(2, 10)}`
  const toolbar = document.createElement('div')
  toolbar.className = 'code-block-toolbar'
  const mark = document.createElement('span')
  mark.className = 'code-block-toolbar__mark'
  mark.setAttribute('aria-hidden', 'true')
  mark.textContent = '󰆍'
  const labelElement = document.createElement('span')
  labelElement.className = 'code-block-toolbar__label'
  labelElement.title = label
  labelElement.textContent = label
  toolbar.append(mark, labelElement)
  const toggle = document.createElement('button')
  toggle.type = 'button'
  toggle.className = 'code-block-toggle'
  toggle.id = `${codeId}-toggle`
  toggle.setAttribute('aria-controls', codeId)
  const toggleIcon = document.createElement('span')
  toggleIcon.className = 'code-block-toggle__icon'
  toggleIcon.setAttribute('aria-hidden', 'true')
  toggleIcon.textContent = '⌄'
  toggle.append(toggleIcon)
  toolbar.append(toggle)
  const pre = block.querySelector<HTMLElement>('pre')
  const lineNumbers = block.querySelector<HTMLElement>('.line-numbers-wrapper')
  pre?.setAttribute('id', codeId)
  block.prepend(toolbar)

  let collapsedState = lineCount > INITIAL_COLLAPSE_LINES
  let blockAnimation: Animation | null = null
  let contentAnimation: Animation | null = null
  let lineNumberAnimation: Animation | null = null

  const setCollapsed = (collapsed: boolean, animate = true) => {
    collapsedState = collapsed
    const visualHeight = block.getBoundingClientRect().height
    toggle.setAttribute('aria-expanded', String(!collapsed))
    toggle.setAttribute('aria-label', collapsed ? '展开代码' : '收起代码')
    toggle.title = collapsed ? '展开代码' : '收起代码'
    blockAnimation?.cancel()
    contentAnimation?.cancel()
    lineNumberAnimation?.cancel()

    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (!animate || reduceMotion || !pre) {
      block.classList.toggle('is-collapsed', collapsed)
      return
    }

    if (!collapsed) block.classList.remove('is-collapsed')
    const endHeight = collapsed ? toolbar.offsetHeight + 2 : block.scrollHeight
    blockAnimation = block.animate(
      [{ height: `${visualHeight}px` }, { height: `${endHeight}px` }],
      {
        duration: 180,
        easing: 'cubic-bezier(0.22, 1, 0.36, 1)',
        fill: 'both'
      }
    )
    contentAnimation = pre.animate(
      collapsed
        ? [
            { opacity: 1, transform: 'translateY(0)' },
            { opacity: 0, transform: 'translateY(-4px)' }
          ]
        : [
            { opacity: 0, transform: 'translateY(-4px)' },
            { opacity: 1, transform: 'translateY(0)' }
          ],
      { duration: 160, easing: 'cubic-bezier(0.22, 1, 0.36, 1)' }
    )
    if (lineNumbers) {
      lineNumberAnimation = lineNumbers.animate(
        collapsed ? [{ opacity: 1 }, { opacity: 0 }] : [{ opacity: 0 }, { opacity: 1 }],
        { duration: 130, easing: 'ease-out' }
      )
    }
    blockAnimation.onfinish = () => {
      if (collapsedState !== collapsed) return
      block.classList.toggle('is-collapsed', collapsed)
      blockAnimation?.cancel()
      blockAnimation = null
    }
  }
  toggle.addEventListener('click', () => setCollapsed(!collapsedState))
  toolbar.addEventListener('click', (event) => {
    const target = event.target as Element
    if (target.closest('.code-block-toolbar__label, .code-block-toggle')) return
    setCollapsed(!collapsedState)
  })
  setCollapsed(collapsedState, false)
  block.dataset.codeEnhanced = 'true'
}

export function enhanceCodeBlocks(root: ParentNode = document) {
  root.querySelectorAll<HTMLElement>(CODE_BLOCK_SELECTOR).forEach(enhanceCodeBlock)
}
