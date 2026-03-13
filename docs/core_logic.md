# Core Logic and Services

The core business logic of Dietin is centralized in `src/lib/`, separating heavy computations and external service calls from UI components.

## AI Integration (`src/lib/gemini.ts`)

Dietin leverages **Google's Gemini 2.0 Flash** model for intelligent features. This module handles interactions with the Generative AI API, though the implementation currently suffers from architectural debt and security risks (exposing client-side API keys).

*Note: See [docs/critical_analysis.md](./critical_analysis.md) for a comprehensive audit of the AI integration.*

### Key Functions

- `analyzeNutrition(foodDescription: string)`:
    *   **Goal**: Extracts structured nutrition data (calories, macros, health score) from natural language text.
    *   **Logic**:
        1.  Checks user quota (Free vs Pro).
        2.  Validates input using a prompt to ensure it's food-related and realistic.
        3.  Sends a structured prompt to Gemini to return JSON data.
        4.  Parses and validates the JSON response.
    *   **Error Handling**: Returns zeroed data with warnings if parsing fails or input is invalid.

- `analyzeImage(file: File)`:
    *   **Goal**: Identifies food items from an image and provides a description.
    *   **Logic**: The `genAI` instance is monkey-patched. If it detects an image, it constructs a `FormData` object and posts it to the local Python FastAPI middleware (`http://localhost:8000/api/v1/analyze/image`). The Python backend computes the volume deterministically and fetches classification strings, returning a JSON that `gemini.ts` re-wraps to look like a standard Google Generative AI response.

- `analyzeUserProfile(profile: UserProfile)`:
    *   **Note**: Currently delegates to deterministic calculations in `calculations.ts` rather than using AI, ensuring consistent and scientifically backed results for BMR/TDEE.

### Quota Management
- Free users are limited to:
    - 3 Meal Analyses per day
    - 1 Image Analysis per day
- Pro users (checked via Firestore `isPro` flag) have unlimited access.
- Quotas are tracked in `userStore` and reset daily/every 6 hours.

## Health Calculations (`src/lib/calculations.ts`)

This module contains deterministic formulas for health metrics, ensuring accuracy and reproducibility.

### Algorithms Used

- **BMR (Basal Metabolic Rate)**: Uses the **Mifflin-St Jeor** equation.
- **TDEE (Total Daily Energy Expenditure)**: BMR * Activity Multiplier.
- **Calorie Targets**:
    - **Fat Loss**: TDEE - (Weekly Goal * 7700 / 7)
    - **Muscle Gain**: TDEE + (Weekly Goal * 7700 / 7)
    - **Maintenance**: TDEE
- **Macro Distribution**:
    - **Protein**: 1.6–2.2 g/kg (higher for fat loss).
    - **Fat**: ~25% of total calories.
    - **Carbs**: Remainder.

## Backend Services (`src/lib/firebase.ts`)

Initialize Firebase services:
- **Auth**: Handles user sessions.
- **Firestore**: Database for user profiles and synced data.
- **Storage**: (If used) for profile pictures.
- **Network Handling**: Includes logic for offline detection and reconnection (`tryReconnect`, `enableNetwork`).
