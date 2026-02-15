# State Management

Dietin uses **Zustand** for state management, chosen for its simplicity and performance. The state is persisted to `localStorage` to support offline usage and rehydrated on app launch.

## User Store (`src/stores/userStore.ts`)

This is the central store for the application, managing user data, daily logs, and app settings.

### Key State Slices

1.  **User Profile (`user`)**:
    *   Contains personal details (age, weight, height), goals, and calculated metrics (BMR, macro targets).
    *   Syncs to Firestore `users/{uid}` document.

2.  **Daily Logs (`dailyCalories`)**:
    *   A map of `date string -> DailyCalories` object.
    *   Stores meals, total calories, and macro breakdown for each day.
    *   **Note**: Meal data is primarily local-first for speed, but critical summaries are synced.

3.  **Quotas (`dailyMealAnalysis`, `dailyImageAnalysis`)**:
    *   Tracks usage of AI features.
    *   Resets automatically based on `lastQuotaReset` timestamp.

### Synchronization Strategy

The store implements a **Debounced + Throttled** synchronization mechanism to Firestore to minimize writes and costs.

- **`debouncedFirestoreUpdate`**:
    *   Queues updates to Firestore.
    *   Waits for a pause in updates (debounce).
    *   Enforces a minimum time between writes (throttle, e.g., 30s) to prevent spamming the database during rapid UI changes (like sliders).
- **Offline Support**:
    *   Changes are applied to local state immediately (Optimistic UI).
    *   Sync attempts occur in the background.
    *   Logic handles retries and connectivity restoration.

### Other Stores

- **`mealStore.ts`**: (Likely deprecated or merged into `userStore` based on usage).
- **`hydrationStore.ts`**: Manages water intake tracking.
- **`workoutStore.ts`**: Manages workout routines and logs.
