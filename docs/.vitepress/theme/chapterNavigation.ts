export interface ChapterNavItem {
  text: string
  link: string
  index?: string
}

export const chapterNavigation: ChapterNavItem[] = [
  { text: '序章', link: '/chapters/preface', index: '序' },
  { text: '第一章 Overview', link: '/chapters/chapter1', index: '01' },
  { text: '第二章 Python 基础', link: '/chapters/chapter2', index: '02' },
  {
    text: '第三章 NumPy，使用计算机进行线性代数计算',
    link: '/chapters/chapter3',
    index: '03'
  },
  {
    text: '第四章 深度学习框架 PyTorch',
    link: '/chapters/chapter4',
    index: '04'
  },
  { text: '第五章 计算机视觉', link: '/chapters/chapter5', index: '05' },
  {
    text: '第九章 大语言模型 LLM',
    link: '/chapters/chapter9',
    index: '09'
  }
]
