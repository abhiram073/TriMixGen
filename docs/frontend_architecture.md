# TriMixGen Frontend Architecture

The frontend is a Single Page Application (SPA) built to communicate with the TriMixGen FastAPI backend.

## Tech Stack
* **Framework:** React + Vite
* **Language:** TypeScript
* **Styling:** Tailwind CSS (Custom Design Tokens in `index.css`)
* **Routing:** React Router v6
* **API Client:** Axios
* **Visualization:** Recharts
* **Icons:** Lucide React

## Core Modules
1. **`services/api.ts`**: A centralized Axios wrapper that connects to `/generate`, `/tag`, and `/health`. Uses strict TypeScript interfaces to guarantee payload structures.
2. **`hooks/useGenerate.ts`**: Encapsulates the loading state, error catching, and async generation requests.
3. **`hooks/useHistory.ts`**: Automatically syncs the last 10 successful generations into `localStorage` for session persistence.
4. **`hooks/useModelStatus.ts`**: A polling mechanism running in the background to ensure the backend is alive.

## Design Philosophy
We utilized a **Glassmorphism** design language to evoke a sense of modern AI infrastructure.
The `TokenInspector` uses specialized color mapping to dynamically colorize output sequences into visually digestible segments (Telugu vs English).
