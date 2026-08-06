import { defineConfig } from 'vitepress'
import { configureMarkdownAlerts } from './markdownAlerts'

function escapeAttribute(value: string) {
  return value.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "MS for AI",
  description: "东南大学人工智能协会 Missing Semester for AI 课程讲义",
  lang: 'zh-CN',
  lastUpdated: true,
  head: [
    ['link', { rel: 'icon', type: 'image/png', href: '/icon.png' }],
    ['link', { rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css' }]
  ],
  markdown: {
    math: true,
    lineNumbers: true,
    gfmAlerts: false,
    config(md) {
      configureMarkdownAlerts(md)
      const defaultFence = md.renderer.rules.fence ?? ((tokens, idx, options, _env, self) =>
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
