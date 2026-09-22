let mermaidApi: typeof import('mermaid').default | null = null
let renderId = 0
let currentTheme: 'default' | 'dark' | null = null
const sources = new WeakMap<HTMLElement, string>()

function isDark() {
  return document.documentElement.classList.contains('dark')
}

function config(theme: 'default' | 'dark') {
  return {
    startOnLoad: false,
    securityLevel: 'loose' as const,
    theme,
    fontFamily: 'inherit'
  }
}

async function getMermaid(theme: 'default' | 'dark') {
  if (!mermaidApi) {
    const mermaid = (await import('mermaid')).default
    mermaid.initialize(config(theme))
    mermaidApi = mermaid
    currentTheme = theme
  } else if (currentTheme !== theme) {
    mermaidApi.initialize(config(theme))
    currentTheme = theme
  }
  return mermaidApi
}

export async function renderMermaid() {
  const nodes = Array.from(document.querySelectorAll<HTMLElement>('.mermaid'))
  if (!nodes.length) return

  const theme = isDark() ? 'dark' : 'default'
  const themeChanged = currentTheme !== theme
  const mermaid = await getMermaid(theme)

  for (const node of nodes) {
    if (node.dataset.processed === 'true' && !themeChanged) continue
    const source = sources.get(node) ?? node.textContent ?? ''
    sources.set(node, source)
    node.dataset.processed = 'true'
    try {
      const { svg } = await mermaid.render(`mermaid-${renderId++}`, source)
      node.innerHTML = svg
    } catch (error) {
      node.dataset.error = 'true'
      console.error('[mermaid] 渲染失败：', error)
    }
  }
}

export function watchMermaidTheme() {
  const observer = new MutationObserver(() => {
    if ((isDark() ? 'dark' : 'default') !== currentTheme) void renderMermaid()
  })
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
  return () => observer.disconnect()
}
