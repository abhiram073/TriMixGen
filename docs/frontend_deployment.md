# TriMixGen Frontend Deployment Guide

This guide details how to build and deploy the React Vite application.

## Prerequisites
Ensure Node.js 18+ and `npm` are installed.

## Local Development
1. Navigate to the frontend directory:
```bash
cd frontend
```
2. Install dependencies:
```bash
npm install
```
3. Run the Vite development server:
```bash
npm run dev
```

*Note: Ensure the TriMixGen FastAPI backend is running simultaneously on `http://localhost:8000` to prevent CORS or Network Errors.*

## Production Build
To prepare the application for static hosting (e.g., Nginx, Vercel, AWS S3):
```bash
npm run build
```
This command compiles the React code and Tailwind CSS into highly optimized static assets in the `dist/` directory.

## Environment Configuration
If the production backend is hosted elsewhere, update the `API_BASE_URL` in `src/services/api.ts` (or utilize a `.env.production` file supported by Vite via `import.meta.env`).
