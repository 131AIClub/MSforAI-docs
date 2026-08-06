# Markdown Extension Examples

This page demonstrates some of the built-in markdown extensions provided by VitePress.

## Syntax Highlighting

VitePress provides Syntax Highlighting powered by [Shiki](https://github.com/shikijs/shiki), with additional features like line-highlighting:

**Input**

````md
```js{4}
export default {
  data () {
    return {
      msg: 'Highlighted!'
    }
  }
}
```
````

**Output**

```js{4}
export default {
  data () {
    return {
      msg: 'Highlighted!'
    }
  }
}
```

## GitHub Alerts

**Input**

```md
> [!NOTE]
> Context worth keeping in mind while reading.

> [!TIP]
> A practical way to complete the task more efficiently.

> [!IMPORTANT] Required before continuing
> Complete this step before moving to the next section.

> [!WARNING]
> Check the environment before running the command.

> [!CAUTION]
> This action may overwrite existing output.
```

**Output**

> [!NOTE]
> Context worth keeping in mind while reading.

> [!TIP]
> A practical way to complete the task more efficiently.

> [!IMPORTANT] Required before continuing
> Complete this step before moving to the next section.

> [!WARNING]
> Check the environment before running the command.

> [!CAUTION]
> This action may overwrite existing output.

## Alert Containers

**Input**

```md
::: note
Context worth keeping in mind while reading.
:::

::: tip
A practical way to complete the task more efficiently.
:::

::: important Required before continuing
Complete this step before moving to the next section.
:::

::: warning
Check the environment before running the command.
:::

::: caution
This action may overwrite existing output.
:::
```

**Output**

::: note
Context worth keeping in mind while reading.
:::

::: tip
A practical way to complete the task more efficiently.
:::

::: important Required before continuing
Complete this step before moving to the next section.
:::

::: warning
Check the environment before running the command.
:::

::: caution
This action may overwrite existing output.
:::

The existing `info`, `danger`, and `details` containers remain available with their original VitePress behavior.

## More

Check out the documentation for the [full list of markdown extensions](https://vitepress.dev/guide/markdown).
