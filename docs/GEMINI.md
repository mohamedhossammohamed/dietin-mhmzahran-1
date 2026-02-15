# GEMINI.md - Dietin Project Context

This file provides comprehensive context for AI agents (like Gemini) interacting with the **Dietin** project.

## 🚀 Project Overview
**Dietin** is an AI-powered health and nutrition assistant designed to help users track their nutrition, workouts, and health goals. It is built as a modern web application and wrapped for mobile (Android/iOS) using Capacitor.

### Core Features
*   **Nutrition Tracking**: AI-powered meal analysis (via text or image), macro tracking (calories, protein, carbs, fats), and hydration monitoring.
*   **Workout Plans**: Access and follow workout routines with AI-driven suggestions.
*   **Progress Monitoring**: Visual charts for weight, BMI, and goal completion.
*   **AI Integration**: Heavily utilizes Google Gemini for nutrition analysis, image recognition, and personalized fitness coaching.
*   **Pro Model**: Features a subscription-based "Pro" tier with unlimited AI analysis and advanced features.
*   **Offline Support**: Robust offline persistence using Firebase's IndexedDB persistence.

## 🛠️ Technical Stack
*   **Frontend**: React (TypeScript), Vite, Tailwind CSS.
*   **UI Components**: Shadcn UI, Framer Motion (animations), Lucide React (icons).
*   **State Management**: Zustand (with persistence), React Query (for server state).
*   **Backend Services**: 
    *   **Firebase**: Authentication, Firestore (database), Functions, Analytics.
    *   **Supabase/Appwrite**: Mentioned in README, though Firebase is the primary backend observed in the source.
*   **Mobile/PWA**: Capacitor (for native mobile apps), `vite-plugin-pwa` (for PWA capabilities).
*   **AI**: Google Generative AI (Gemini 2.0 Flash), OpenAI SDK.
*   **3D Elements**: React Three Fiber, Three.js (used for some UI elements).

## 🏗️ Architecture & Key Files

### Entry Points
*   `src/main.tsx`: App initialization, PWA registration, theme setup, and router configuration (HashRouter for native, BrowserRouter for web).
*   `src/App.tsx`: Main routing, protected routes, network status monitoring, and global provider wrapping.

### Logic & Services (`src/lib/`)
*   `firebase.ts`: Firebase configuration, initialization, and network/persistence management.
*   `gemini.ts`: Core AI logic for nutrition, profile, and image analysis. Includes quota checks for free vs. pro users.
*   `calculations.ts`: Deterministic health calculations (BMR, TDEE, BMI).
*   `routes.tsx`: Centralized route definitions.

### State Management (`src/stores/`)
*   `userStore.ts`: Manages user profile, daily calories, and AI quotas. Syncs with Firestore using debouncing and throttling.
*   Other stores: `mealStore.ts`, `workoutStore.ts`, `hydrationStore.ts`, `nutritionStore.ts`, `analyticsStore.ts`.

### Pages & Components
*   `src/pages/`: Contains main view components like `Index.tsx` (Dashboard), `Diet.tsx`, `Workouts.tsx`, `Profile.tsx`, etc.
*   `src/components/`: Reusable UI components.
    *   `ui/`: Shadcn UI atomic components.
    *   `MealAnalysis.tsx`: Comprehensive component for logging meals via AI or manual entry.

## 🏃‍♂️ Development Workflows

### Key Commands
*   `npm run dev`: Start local development server (typically port 3000).
*   `npm run build`: Generate production build in `dist/`.
*   `npm run build:android`: Build the app and sync/open in Android Studio.
*   `npm run lint`: Run ESLint checks.

### Environment Variables
The project requires several environment variables for Firebase, Google AI, and other services. Check `.env.example` if available.
*   `VITE_FIREBASE_API_KEY`, `VITE_FIREBASE_PROJECT_ID`, etc.
*   `VITE_GOOGLE_AI_KEY`: API key for Gemini.

## 📏 Development Conventions
*   **Style**: Use Tailwind CSS for most styling. Follow the "Apple-like" clean aesthetic (SF Pro font mentioned in code).
*   **State**: Prefer Zustand for client-side state. Use the persisted stores for data that should survive reloads.
*   **Components**: Use Shadcn UI as the base for new components.
*   **Firebase Sync**: Be mindful of Firestore read/write costs. Use the debounced/throttled update pattern found in `userStore.ts`.
*   **Internationalization**: All user-facing strings should be wrapped in `t()` using `react-i18next`.
*   **Mobile Considerations**: Ensure new features are responsive and work well within the Capacitor/PWA context (e.g., safe area insets).

## 🧩 AI Interaction Tips
*   When adding new features, check `src/lib/gemini.ts` for established AI prompt patterns.
*   Always consider the quota management logic for new AI-driven features.
*   Maintain the strict TypeScript typing found throughout the project.
