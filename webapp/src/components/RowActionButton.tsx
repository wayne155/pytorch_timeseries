// webapp/src/components/RowActionButton.tsx

interface RowActionButtonProps {
  model: string
  dataset: string
  view: string
}

export function RowActionButton({ model, dataset, view }: RowActionButtonProps) {
  return (
    <button
      title="Row actions"
      aria-label={`Actions for ${model} on ${dataset} (${view})`}
      disabled
      className="text-gray-300 px-1.5 py-0.5 rounded text-base leading-none cursor-not-allowed"
    >
      ⋯
    </button>
  )
}
