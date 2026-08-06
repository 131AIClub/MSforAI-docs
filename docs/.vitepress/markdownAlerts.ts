import type MarkdownIt from 'markdown-it'
import markdownItContainer from 'markdown-it-container'
import markdownItGitHubAlerts from 'markdown-it-github-alerts'

const ALERT_TYPES = ['note', 'tip', 'important', 'warning', 'caution'] as const

type AlertType = (typeof ALERT_TYPES)[number]

const DEFAULT_TITLES: Record<AlertType, string> = {
  note: 'NOTE',
  tip: 'TIP',
  important: 'IMPORTANT',
  warning: 'WARNING',
  caution: 'CAUTION'
}

function renderAlertOpen(md: MarkdownIt, type: AlertType, title: string) {
  return `<AlertBox type="${type}" title="${md.utils.escapeHtml(title)}">\n`
}

export function configureMarkdownAlerts(md: MarkdownIt) {
  markdownItGitHubAlerts(md, {
    markers: ALERT_TYPES.map((type) => type.toUpperCase()),
    titles: DEFAULT_TITLES,
    icons: Object.fromEntries(ALERT_TYPES.map((type) => [type, '']))
  })

  md.renderer.rules.alert_open = (tokens, index) => {
    const { title, type } = tokens[index].meta as { title: string; type: AlertType }
    return renderAlertOpen(md, type, title)
  }
  md.renderer.rules.alert_close = () => '</AlertBox>\n'

  for (const type of ['note', 'important', 'caution'] as const) {
    md.use(markdownItContainer, type)
  }

  for (const type of ALERT_TYPES) {
    md.renderer.rules[`container_${type}_open`] = (tokens, index) => {
      const customTitle = tokens[index].info.trim().slice(type.length).trim()
      return renderAlertOpen(md, type, customTitle || DEFAULT_TITLES[type])
    }
    md.renderer.rules[`container_${type}_close`] = () => '</AlertBox>\n'
  }
}
