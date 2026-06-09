import { useState, useEffect } from 'react'
import type { TaskLeaderboardData } from '../types'

export function useLeaderboard(url = './leaderboard_data.json') {
  const [data, setData] = useState<TaskLeaderboardData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const load = () => {
    setLoading(true)
    setError(null)
    fetch(url)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json() as Promise<TaskLeaderboardData>
      })
      .then(d => { setData(d); setLoading(false) })
      .catch((e: Error) => { setError(e.message); setLoading(false) })
  }

  useEffect(() => { load() }, [url])
  return { data, error, loading, refresh: load }
}
