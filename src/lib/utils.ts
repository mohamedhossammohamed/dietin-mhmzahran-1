import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
export function extractJsonFromAI(text: string): string {
  if (!text) return "";
  let cleaned = text;
  if (cleaned.includes('```')) {
    cleaned = cleaned.replace(/```(?:json)?\n([\s\S]*?)```/g, "$1").trim();
  }
  const jsonMatch = cleaned.match(/(\{|\[)[\s\S]*(\}|\])/);
  if (jsonMatch) {
    return jsonMatch[0];
  }
  return cleaned;
}
