import { defineConfig } from 'vitepress'
import { katex } from '@mdit/plugin-katex'
import type { Token } from 'markdown-it'
import { configureMarkdownAlerts } from './markdownAlerts'

function escapeAttribute(value: string) {
  return value.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function findStandaloneImage(token: Token | undefined) {
  const children = token?.children
  if (!children) return null
  if (children.length === 1 && children[0].type === 'image') return children[0]
  if (
    children.length === 3 &&
    children[0].type === 'link_open' &&
    children[1].type === 'image' &&
    children[2].type === 'link_close'
  ) {
    return children[1]
  }
  return null
}

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "MS for AI",
  description: "东南大学人工智能协会 Missing Semester for AI 课程讲义",
  lang: 'zh-CN',
  lastUpdated: true,
  head: [
    ['link', { rel: 'icon', type: 'image/png', href: '/icon.png' }]
  ],
  markdown: {
    math: false,
    lineNumbers: true,
    gfmAlerts: false,
    config(md) {
      configureMarkdownAlerts(md)
      md.use(katex, {
        delimiters: 'dollars',
        throwOnError: false,
        macros: {
          '\\R': '\\mathbb{R}'
        },
        transformer(html, displayMode) {
          return displayMode
            ? html
                .replace("<p class='katex-block'", "<p v-pre class='katex-block'")
                .replace('<span class="katex"', '<span class="katex" data-math-display="block"')
            : html.replace('<span class="katex"', '<span v-pre class="katex" data-math-display="inline"')
        }
      })
      const defaultFence = md.renderer.rules.fence ?? ((tokens, idx, options, _env, self) =>
        self.renderToken(tokens, idx, options))
      const defaultParagraphOpen = md.renderer.rules.paragraph_open ?? ((tokens, idx, options, _env, self) =>
        self.renderToken(tokens, idx, options))
      const defaultParagraphClose = md.renderer.rules.paragraph_close ?? ((tokens, idx, options, _env, self) =>
        self.renderToken(tokens, idx, options))

      md.renderer.rules.fence = (tokens, idx, options, env, self) => {
        const token = tokens[idx]
        const info = token.info.trim()
        const title = info.match(/\[([^\]]+)\]/)?.[1]?.trim()
        const html = defaultFence(tokens, idx, options, env, self)
        if (!title) return html
        return html.replace(
          /<div class="language-([^" ]+)/,
          `<div data-code-title="${escapeAttribute(title)}" class="language-$1`
        )
      }

      md.renderer.rules.paragraph_open = (tokens, idx, options, env, self) => {
        const image = findStandaloneImage(tokens[idx + 1])
        const caption = image ? self.renderInlineAsText(image.children ?? [], options, env).trim() : ''
        return caption ? '<figure class="ms-figure">\n' : defaultParagraphOpen(tokens, idx, options, env, self)
      }

      md.renderer.rules.paragraph_close = (tokens, idx, options, env, self) => {
        const image = findStandaloneImage(tokens[idx - 1])
        const caption = image ? self.renderInlineAsText(image.children ?? [], options, env).trim() : ''
        return caption
          ? `<figcaption>${escapeAttribute(caption)}</figcaption>\n</figure>\n`
          : defaultParagraphClose(tokens, idx, options, env, self)
      }
    }
  },
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: {
      src: '/icon.png',
      alt: 'QQ群 594740801 群头像'
    },

    nav: [
      { text: '首页', link: '/' },
      { text: '课程讲义', link: '/chapters/preface' },
      { text: '关于', link: '/about' }
    ],

    aside: 'right',

    socialLinks: [
      { icon: 'github', link: 'https://github.com/131AIClub' }
    ],

    footer: {
      message: 'Missing Semester for Artificial Intelligence',
      copyright: 'Copyright © 2026 MS for AI'
    },

    editLink: {
      pattern: 'https://github.com/131AIClub/MSforAI-docs/edit/master/docs/:path',
      text: '在 GitHub 上编辑此页'
    },

    docFooter: {
      prev: '上一页',
      next: '下一页'
    },

    outline: {
      label: '页面导航',
      level: [2, 3]
    },

    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'medium'
      }
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档'
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换',
              closeText: '关闭'
            }
          }
        }
      }
    }
  }
})
