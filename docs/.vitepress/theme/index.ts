import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import AlertBox from './AlertBox.vue'
import HomePage from './HomePage.vue'
import Layout from './Layout.vue'
import ThemedImage from './ThemedImage.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    app.component('AlertBox', AlertBox)
    app.component('HomePage', HomePage)
    app.component('ThemedImage', ThemedImage)
  }
} satisfies Theme
