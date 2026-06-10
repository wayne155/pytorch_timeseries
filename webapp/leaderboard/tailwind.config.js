/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: '#FAF8F2',   // warm paper page background
          panel: '#FFFFFF',     // panels / rails
          raised: '#F2EEE4',    // hover / raised surfaces
        },
        line: {
          DEFAULT: 'rgba(28, 27, 23, 0.18)', // hairlines
          strong: 'rgba(28, 27, 23, 0.42)',
        },
        ink: {
          DEFAULT: '#1C1B17',   // near-black warm text
          dim: '#47443D',
          faint: '#7D786C',
        },
        phosphor: {
          DEFAULT: '#C2780C',   // deep amber accent
          bright: '#8A5208',    // stronger emphasis (darker on light bg)
          dim: 'rgba(194, 120, 12, 0.16)',
        },
        signal: {
          DEFAULT: '#0E7C8A',   // teal secondary accent
          dim: 'rgba(14, 124, 138, 0.10)',
        },
        worst: '#B4473A',
      },
      fontFamily: {
        display: ['"Instrument Serif"', 'Georgia', 'serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
    },
  },
  plugins: [],
}
