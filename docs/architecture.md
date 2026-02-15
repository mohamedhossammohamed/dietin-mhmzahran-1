# Dietin Project Architecture

Dietin is a modern, AI-powered health and nutrition tracking application. It is built as a Single Page Application (SPA) using React and Vite, wrapped for mobile distribution via Capacitor.

## High-Level Overview

The application follows a client-heavy architecture where most logic resides in the frontend, interacting directly with third-party services (Firebase, Gemini AI) for backend functionality.

### Core Technologies

- **Frontend Framework**: [React](https://react.dev/) (TypeScript) with [Vite](https://vitejs.dev/)
- **UI Toolkit**: [Tailwind CSS](https://tailwindcss.com/) + [shadcn/ui](https://ui.shadcn.com/)
- **State Management**: [Zustand](https://docs.pmnd.rs/zustand/getting-started/introduction) (with persistence) + [React Query](https://tanstack.com/query/latest)
- **Backend & Auth**: [Firebase](https://firebase.google.com/) (Firestore, Authentication)
- **AI Integration**: [Google Gemini 2.0 Flash](https://deepmind.google/technologies/gemini/) (via Generative AI SDK)
- **Mobile Wrapper**: [Capacitor](https://capacitorjs.com/)

## Project Structure

```
/src
  /components      # UI Components (Reusable & Feature-specific)
    /ui            # Shadcn atomic components
  /lib             # Core business logic and service integrations
    gemini.ts      # AI Logic
    firebase.ts    # Backend Configuration
    calculations.ts # Health Metrics Logic
  /pages           # Application Route Screens
  /stores          # Global State Managers (Zustand)
  /hooks           # Custom React Hooks
  App.tsx          # Main Routing & Layout
  main.tsx         # Entry Point
```

## Key Flows

1.  **Authentication**: Users sign in via Firebase Auth (Google or Email/Password).
2.  **Onboarding**: New users complete a profile setup (Age, Weight, Goal) which calculates initial health metrics using deterministic formulas (`src/lib/calculations.ts`).
3.  **Core Loop**:
    *   **Tracking**: Users log meals via text or image (processed by Gemini AI) and workouts.
    *   **Analysis**: AI provides nutrition breakdown and health scores.
    *   **Progress**: Data is stored locally and synced to Firestore for persistence.
    *   **Quotas**: Free users have daily limits on AI analysis; Pro users get unlimited access.

## Data Flow

Data primarily flows from **User Actions** -> **Zustand Store** -> **Local Persistence** -> **Firestore Sync**.
The app is designed to work offline-first, syncing data when a connection is available.
