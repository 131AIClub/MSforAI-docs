const inlineDelimiters = ['$', '$'] as const
const displayDelimiters = ['$$', '$$'] as const

function closestKatex(node: Node) {
  const element = node instanceof Element ? node : node.parentElement
  return element?.closest<HTMLElement>('.katex') ?? null
}

function selectionTouchesDocument(range: Range) {
  const container =
    range.commonAncestorContainer instanceof Element
      ? range.commonAncestorContainer
      : range.commonAncestorContainer.parentElement
  return Boolean(container?.closest('.VPDoc'))
}

function replaceFormulasWithTex(fragment: DocumentFragment) {
  const formulas = fragment.querySelectorAll<HTMLElement>('.katex')

  for (const formula of formulas) {
    const annotation = formula.querySelector('annotation[encoding="application/x-tex"]')
    if (!annotation) continue

    const delimiters = formula.dataset.mathDisplay === 'block' ? displayDelimiters : inlineDelimiters
    formula.replaceWith(`${delimiters[0]}${annotation.textContent ?? ''}${delimiters[1]}`)
  }

  return formulas.length > 0
}

function copySelectedMath(event: ClipboardEvent) {
  const selection = window.getSelection()
  if (!selection || selection.isCollapsed || selection.rangeCount === 0 || !event.clipboardData) return

  const selectedRange = selection.getRangeAt(0)
  if (!selectionTouchesDocument(selectedRange)) return

  const copyRange = selectedRange.cloneRange()
  const startFormula = closestKatex(copyRange.startContainer)
  const endFormula = closestKatex(copyRange.endContainer)

  if (startFormula) copyRange.setStartBefore(startFormula)
  if (endFormula) copyRange.setEndAfter(endFormula)

  const fragment = copyRange.cloneContents()
  const htmlContainer = document.createElement('div')
  htmlContainer.append(fragment.cloneNode(true))

  if (!replaceFormulasWithTex(fragment)) return

  event.clipboardData.setData('text/html', htmlContainer.innerHTML)
  event.clipboardData.setData('text/plain', fragment.textContent ?? '')
  event.preventDefault()
}

export function installMathCopyHandler() {
  if (typeof document === 'undefined') return

  document.removeEventListener('copy', copySelectedMath)
  document.addEventListener('copy', copySelectedMath)
}
