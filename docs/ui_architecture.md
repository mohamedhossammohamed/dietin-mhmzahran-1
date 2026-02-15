# UI Architecture & Components

The frontend is built with React and styled using Tailwind CSS and shadcn/ui. The structure emphasizes reusable, atomic components composed into feature-rich pages.

## Layout & Navigation (`src/App.tsx`)

- **Main Router**: Handles navigation between pages.
- **Bottom Navigation**: Persistent bar for mobile-first navigation (Home, Diet, Workouts, Profile).
- **Protected Routes**: Redirects unauthenticated users to `/auth`.
- **Onboarding Guard**: Redirects new users to `/welcome` until profile is complete.

## Components (`src/components/`)

### Key Feature Components

#### `MealAnalysis.tsx`
The core nutrition logging interface.
- **Features**:
    - Text-based food logging (AI-analyzed).
    - Image-based food logging (AI-analyzed).
    - Manual macro editing.
    - Pro features (detailed micronutrients, suggestions).
- **User Flow**: User opens modal -> Inputs description/photo -> AI processes -> Results shown -> User saves.

#### `ProSubscriptionPanel.tsx`
Handles the upgrade flow for Dietin Pro.

#### `ImproveAI.tsx`
Provides AI-driven suggestions for meal improvement (healthier alternatives, portion adjustments).

### Atomic UI Components (`src/components/ui/`)
Standard shadcn/ui components:
- `button`, `input`, `card`, `dialog`, `toast`, etc.
- Used throughout the app for consistent styling.

## Pages (`src/pages/`)

### 1. `Index.tsx` (Home/Dashboard)
- Displays daily summary (Calories consumed vs target).
- Shows progress rings for macros.
- Quick actions for logging.

### 2. `Diet.tsx`
- Detailed view of daily meals.
- List of logged items with macro breakdown.
- Calendar view for history.

### 3. `Workouts.tsx`
- Workout tracking interface.
- Lists available routines and allows logging completed sets.

### 4. `Profile.tsx`
- User settings and physical attributes.
- Settings for dietary preferences and allergies.
- Subscription management.

### 5. `Plan.tsx`
- Displays the generated diet/workout plan based on user goals.

### 6. `Auth.tsx` / `Welcome.tsx`
- Handle user authentication and initial onboarding flow.
