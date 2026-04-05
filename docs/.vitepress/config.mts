import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "MS for AI",
  description: "东南大学人工智能协会 Missing Semester for AI 课程讲义",
  lang: 'zh-CN',
  lastUpdated: true,
  head: [
    ['link', { rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css' }]
  ],
  markdown: {
    math: true
  },
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: '/logo.svg',

    nav: [
      { text: '首页', link: '/' },
      { text: '讲义', link: '/chapters/preface' },
      { text: '关于我们', link: '/about' }
    ],

    sidebar: {
      '/chapters/': [
        {
          text: '课程讲义',
          items: [
            { text: '序章', link: '/chapters/preface' },
            { text: '第一章：Overview', link: '/chapters/chapter1' },
            { text: '第二章：Python 基础', link: '/chapters/chapter2' },
            { text: '第三章：NumPy', link: '/chapters/chapter3' },
            { text: '第四章：PyTorch', link: '/chapters/chapter4' },
            { text: '第五章：计算机视觉', link: '/chapters/chapter5' },
            { text: '第六章：自然语言处理', link: '/chapters/chapter6' },
            { text: '第九章：大语言模型', link: '/chapters/chapter9' }
          ]
        }
      ]
    },

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
