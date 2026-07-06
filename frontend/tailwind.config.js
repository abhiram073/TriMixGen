/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: 'var(--background)',
        surface: 'var(--surface)',
        primary: 'var(--primary)',
        secondary: 'var(--secondary)',
        border: 'var(--border)',
        text: 'var(--text)',
        'text-muted': 'var(--text-muted)',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
