import { GoogleGenerativeAI } from "@google/generative-ai";
import { UserProfile, Goal } from './types';
import { computeProfileAnalysis } from './calculations';
import { useUserStore } from "@/stores/userStore";
import { auth, db } from "@/lib/firebase";
import { doc, getDoc } from "firebase/firestore";

const originalGenAI = new GoogleGenerativeAI(import.meta.env.VITE_GOOGLE_AI_KEY);

export const genAI = {
  getGenerativeModel: (options: any) => {
    const originalModel = originalGenAI.getGenerativeModel(options);

    return {
      ...originalModel,
      generateContent: async (request: any) => {
        // Check if there is an inlineData (image) in the request
        let hasImage = false;
        let base64Data = null;
        let mimeType = null;
        let promptText = "";

        if (typeof request === "string") {
            promptText = request;
        } else if (Array.isArray(request)) {
            promptText = request.find(item => typeof item === "string") || "";
            const inlineDataPart = request.find((p: any) => p.inlineData);
            if (inlineDataPart) {
                hasImage = true;
                base64Data = inlineDataPart.inlineData.data;
                mimeType = inlineDataPart.inlineData.mimeType;
            }
        } else if (request && request.contents && request.contents[0] && request.contents[0].parts) {
          const textPart = request.contents[0].parts.find((p: any) => p.text);
          if (textPart) {
              promptText = textPart.text;
          }
          const inlineDataPart = request.contents[0].parts.find((p: any) => p.inlineData);
          if (inlineDataPart) {
            hasImage = true;
            base64Data = inlineDataPart.inlineData.data;
            mimeType = inlineDataPart.inlineData.mimeType;
          }
        }

        if (hasImage) {
          // Intercept! Route to local FastAPI instance
          const byteCharacters = atob(base64Data);
          const byteNumbers = new Array(byteCharacters.length);
          for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
          }
          const byteArray = new Uint8Array(byteNumbers);
          const blob = new Blob([byteArray], { type: mimeType || "image/jpeg" });
          const file = new File([blob], "image.jpg", { type: mimeType || "image/jpeg" });

          const formData = new FormData();
          formData.append("full_image", file);

          // MHMZ: Rerouted AI call to local FastAPI middleware for deterministic math and secure API gateway.
          const response = await fetch("http://localhost:8000/api/v1/analyze/image", {
            method: "POST",
            body: formData
          });

          if (!response.ok) throw new Error("Backend analysis failed.");
          const responseData = await response.json();

          if (responseData && responseData.data && responseData.data.warnings) {
            console.warn("Backend Warnings:", responseData.data.warnings);
          }

          // Extract data from AnalysisResponse format
          const backendData = responseData.data;

          // Return fake Gemini-like response parsing what MealAnalysis.tsx actually expects
          const fakeGeminiResponseText = JSON.stringify({
            isFood: responseData.success ?? true,
            title: backendData.foodName || "Analyzed Food",
            calories: backendData.calories || 0,
            protein: backendData.macros?.protein || 0,
            carbs: backendData.macros?.carbs || 0,
            fat: backendData.macros?.fat || 0,
            cholesterol: 0,
            magnesium: 0,
            sugar: 0,
            fiber: 0,
            sodium: 0,
            potassium: 0,
            vitaminA: 0,
            vitaminC: 0,
            calcium: 0,
            iron: 0,
            score: backendData.confidenceScore != null ? Math.round(backendData.confidenceScore * 100) : 85,
            suggestions: backendData.warnings && backendData.warnings.length > 0 ? backendData.warnings : ["Analyzed successfully using Vision Engine."],
            ingredients: ["Detected by DIETIN Backend Vision Engine"]
          });

          return {
            response: {
              text: () => fakeGeminiResponseText
            }
          };
        }

        // MHMZ: Intercept text analysis and route to FastAPI proxy
        try {
            const response = await fetch("http://localhost:8000/api/v1/proxy/generate", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    prompt: promptText,
                    system_instruction: options?.systemInstruction || null
                })
            });

            if (!response.ok) {
                throw new Error("Backend text proxy failed.");
            }

            const data = await response.json();
            return {
                response: {
                    text: () => data.text
                }
            };
        } catch (error) {
            console.error("Proxy error:", error);
            return originalModel.generateContent(request);
        }
      }
    };
  }
};

interface NutritionAnalysis {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  healthScore: number;
  warning?: string;
}

interface AnalysisResult {
  goal: string;
  calories: number;
  metabolism: number;
  protein: number;
  carbs: number;
  fat: number;
  estimatedWeeks: number;
}

// Cache for nutrition results
const nutritionCache = new Map<string, NutritionAnalysis>();

// Fallback calculation if AI fails
function calculateBasicTDEE(profile: Partial<UserProfile>) {
  const weight = profile.weight || 70;
  const height = profile.height || 170;
  const age = profile.age || 25;

  // Basic BMR calculation
  const bmr = profile.gender === 'MALE'
    ? (10 * weight) + (6.25 * height) - (5 * age) + 5
    : (10 * weight) + (6.25 * height) - (5 * age) - 161;

  // Activity multiplier
  const activityMultipliers = {
    LIGHTLY_ACTIVE: 1.375,
    MODERATELY_ACTIVE: 1.55,
    VERY_ACTIVE: 1.725,
    EXTRA_ACTIVE: 1.9,
  };

  const multiplier = activityMultipliers[profile.activityLevel || 'MODERATELY_ACTIVE'];

  return Math.round(bmr * multiplier);
}

async function checkProStatus(): Promise<boolean> {
  if (!auth.currentUser) return false;

  try {
    const userDoc = await getDoc(doc(db, "users", auth.currentUser.uid));
    if (userDoc.exists()) {
      const userData = userDoc.data();
      return userData.isPro || false;
    }
    return false;
  } catch (error) {
    console.error('Error checking pro status:', error);
    return false;
  }
}

export async function analyzeNutrition(foodDescription: string): Promise<NutritionAnalysis> {
  const { dailyMealAnalysis, incrementMealAnalysis } = useUserStore.getState();

  // Get real-time pro status from Firestore
  const isPro = await checkProStatus();

  console.log('Checking meal analysis quota:', {
    isPro,
    dailyMealAnalysis,
    limit: 3
  });

  // Check if user is not pro and has reached quota
  if (!isPro && dailyMealAnalysis >= 3) {
    const event = new CustomEvent('showErrorToast', {
      detail: { message: 'Daily meal analysis limit reached (3/3). Please subscribe to DietinPro for unlimited analysis or wait 24 hours.' }
    });
    window.dispatchEvent(event);
    throw new Error('Quota reached');
  }

  // Increment quota counter BEFORE starting analysis for non-pro users
  if (!isPro) {
    console.log('Incrementing meal analysis quota for free user');
    incrementMealAnalysis();
  }

  try {
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

    // Updated validation prompt to check for haram food and unrealistic amounts
    const validationPrompt = `Please validate if this user input is a food/drink/human consumable item.'
Analyze this input: "${foodDescription}"

Rules:
1. Accept ANY language (English, Franco-Arabic like "ma7shi/ta3miya", Arabic, French, Chinese, etc.)
2. Ignore ALL spelling mistakes completely
3. Accept common food nicknames and slang
4. Accept numeric character substitutions (like 7 for ح, 3 for ع, etc.)
5. Accept any measurement units (kg, g, lbs, pieces, etc.)
6. Accept both formal and informal food descriptions
7. REJECT if portions are unrealistic (e.g. "1000kg rice", "50kg meat", anything over 10kg)
8. REJECT haram, illegal foods like:
   - Pork
   - Alcohol
   -etc. 
9. ACCEPT all regular soft drinks and beverages (like Pepsi, Coca-Cola, etc.) as they are halal
10. REJECT if the description contains non-food items
11. REJECT if the description is nonsensical or inappropriate

Is this describing consumable food/drink with realistic portions?
Answer ONLY with "yes" or "no" followed by "|" and the reason if "no".
Example responses:
"yes"
"no|Contains pork"
"no|Unrealistic portion size"
"no|Not halal meat"`;

    const validationResult = await model.generateContent(validationPrompt);
    const validationResponse = await validationResult.response;
    const validationText = validationResponse.text().toLowerCase();

    const [isValid, reason] = validationText.split('|');
    const isFood = isValid.includes('yes');

    if (!isFood) {
      // Show error toast for invalid food
      const errorMessage = reason || "Please enter a valid halal food item";

      // Create and dispatch a custom event for showing the error toast
      const event = new CustomEvent('showErrorToast', {
        detail: { message: errorMessage.charAt(0).toUpperCase() + errorMessage.slice(1) }
      });
      window.dispatchEvent(event);

      return {
        calories: 0,
        protein: 0,
        carbs: 0,
        fat: 0,
        healthScore: 0,
        warning: errorMessage
      };
    }

    // If it is food, proceed with nutrition analysis
    const prompt = `Please analyze the nutrition facts of this food/meal:
    Calories, Protein, Carbs, Fat, Health Score (Health score based on healthiness of the food preciesly between 0 and 100, eg, 32,52, 56, 78, 90, 100)
The Meal description is: "${foodDescription}"
Return ONLY a JSON object in this exact format (no explanation, no other text):
{
  "calories": number,
  "protein": number,
  "carbs": number,
  "fat": number,
  "healthScore": number
}`;

    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();

    // Clean the response text
    const cleanedText = text.replace(/```json\n?|\n?```/g, '').trim();

    try {
      const parsed = JSON.parse(cleanedText);

      // Validate the parsed data
      if (!parsed || typeof parsed !== 'object') {
        throw new Error('Invalid response format');
      }

      // Convert to numbers with precise decimals
      const validatedData = {
        calories: Math.max(0, Number(parsed.calories) || 0),
        protein: Math.max(0, Number(Number(parsed.protein).toFixed(1)) || 0),
        carbs: Math.max(0, Number(Number(parsed.carbs).toFixed(1)) || 0),
        fat: Math.max(0, Number(Number(parsed.fat).toFixed(1)) || 0),
        healthScore: Math.min(100, Math.max(0, Number(Number(parsed.healthScore).toFixed(1)) || 0))
      };

      return validatedData;
    } catch (error) {
      console.error("Failed to parse nutrition data:", error, "Raw text:", text);
      return {
        calories: 0,
        protein: 0,
        carbs: 0,
        fat: 0,
        healthScore: 0,
        warning: "Failed to analyze food"
      };
    }
  } catch (error) {
    console.error("Error analyzing food:", error);
    throw error;
  }
}

export async function analyzeUserProfile(profile: UserProfile): Promise<AnalysisResult> {
  // Replace AI-based analysis with deterministic, formula-based calculations
  // using open-source Mifflin-St Jeor for BMR, activity multipliers for TDEE,
  // goal/weekly target to adjust calories, and macro distribution.
  const result = computeProfileAnalysis(profile);
  return {
    goal: result.goal,
    calories: result.calories,
    metabolism: result.metabolism,
    protein: result.protein,
    carbs: result.carbs,
    fat: result.fat,
    estimatedWeeks: result.estimatedWeeks,
  };
}

export async function analyzeImage(file: File): Promise<any> {
  const { dailyImageAnalysis, incrementImageAnalysis } = useUserStore.getState();

  // Get real-time pro status from Firestore
  const isPro = await checkProStatus();

  console.log('Checking image analysis quota:', {
    isPro,
    dailyImageAnalysis,
    limit: 1
  });

  // Check if user is not pro and has reached quota
  if (!isPro && dailyImageAnalysis >= 1) {
    const event = new CustomEvent('showErrorToast', {
      detail: { message: 'Daily image analysis limit reached (1/1). Please subscribe to DietinPro for unlimited analysis or wait 24 hours.' }
    });
    window.dispatchEvent(event);
    throw new Error('Quota reached');
  }

  // Increment quota counter BEFORE starting analysis for non-pro users
  if (!isPro) {
    console.log('Incrementing image analysis quota for free user');
    incrementImageAnalysis();
  }

  try {
    const formData = new FormData();
    formData.append("full_image", file);

    // MHMZ: Rerouted AI call to local FastAPI middleware for deterministic math and secure API gateway.
    const response = await fetch("http://localhost:8000/api/v1/analyze/image", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      throw new Error("Backend analysis failed.");
    }

    const data = await response.json();

    if (data && data.data && data.data.warnings) {
      console.warn("Backend Warnings:", data.data.warnings);
    }

    return data;
  } catch (error) {
    console.error('Image analysis failed:', error);
    throw error;
  }
}

export async function analyzeFood(description: string) {
  try {
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

    const prompt = `Analyze this food: "${description}"`;
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();

    try {
      const parsed = JSON.parse(text);
      return {
        calories: parsed.nutrition.calories,
        protein: parsed.nutrition.protein,
        carbs: parsed.nutrition.carbs,
        fat: parsed.nutrition.fat,
        healthScore: Math.round(parsed.healthScore)
      };
    } catch (error) {
      console.error("Failed to parse nutrition data:", error);
      return null;
    }
  } catch (error) {
    console.error("Error analyzing food:", error);
    return null;
  }
}

export async function analyzeWorkout(workoutDescription: string): Promise<any> {
  // Safely access store to avoid type issues when properties are absent
  const store: any = (useUserStore as any)?.getState?.() ?? {};
  const dailyWorkoutAnalysis: number = typeof store.dailyWorkoutAnalysis === 'number' ? store.dailyWorkoutAnalysis : 0;
  const incrementWorkoutAnalysis: undefined | (() => void) = typeof store.incrementWorkoutAnalysis === 'function' ? store.incrementWorkoutAnalysis : undefined;

  // Check if user is not pro and has reached quota
  if (dailyWorkoutAnalysis >= 5) {
    const event = new CustomEvent('showErrorToast', {
      detail: { message: 'Daily workout analysis limit reached. Please subscribe to DietinPro for unlimited analysis or wait 24 hours.' }
    });
    window.dispatchEvent(event);
    throw new Error('Quota reached');
  }

  try {
    // ... rest of the code ...

    // Increment quota counter on successful analysis
    if (incrementWorkoutAnalysis) {
      incrementWorkoutAnalysis();
    }

    // TODO: implement real workout analysis; returning null placeholder for now
    return null as any;
  } catch (error) {
    // ... rest of the code ...
  }
}
