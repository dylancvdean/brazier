import { Fragment } from 'react'

/** Renders daemon hint text, where `backticked` spans are commands. */
export function InlineCode(props: { text: string }): React.JSX.Element {
  const parts = props.text.split('`')
  return (
    <>
      {parts.map((part, index) =>
        index % 2 === 1 && index < parts.length - 1 ? (
          <code key={index}>{part}</code>
        ) : (
          <Fragment key={index}>{index % 2 === 1 ? `\`${part}` : part}</Fragment>
        )
      )}
    </>
  )
}
