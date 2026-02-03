import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Check, AlertTriangle, X, ArrowLeft, Lock, ChevronRight, Dumbbell, ClipboardList, BarChart3 } from 'lucide-react';
import { useNavigate, Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { useUserStore } from '@/stores/userStore';
import { GoogleGenerativeAI } from "@google/generative-ai";
import { genAI } from "@/lib/gemini";
import ProFeatures from '@/components/ProFeatures';
import { ProSubscriptionPanel } from '@/components/ProSubscriptionPanel';
import { cn } from '@/lib/utils';
import { useTranslation } from 'react-i18next';


// Example insights for non-pro users
const EXAMPLE_INSIGHTS = [
  {
    title: 'Volume Progress',
    text: 'Your training volume has increased by 15% over the past month. Keep pushing!' 
  },
  {
    title: 'Rest Optimization',
    text: 'Consider increasing rest periods between heavy sets to 2-3 minutes for better recovery.'
  },
  {
    title: 'Training Focus',
    text: 'Your upper body training is well-balanced. Consider adding more lower body exercises.'
  }
];

interface WorkoutHistory {
  date: string;
  muscleGroup: string;
  exercises: {
    name: string;
    musclesWorked: string[];
    setsCompleted: number;
    totalSets: number;
    reps: number;
    weight: number;
    restTime: number;
    rpm?: number;
    volume?: number;
    isCompleted?: boolean;
  }[];
  completionPercentage: number;
}

interface AIInsight {
  title?: string;
  text: string;
  tag?: string;
  priority?: 'high' | 'medium' | 'low';
  workoutKey?: string;
  workoutDate?: string;
  muscleGroup?: string;
}

const containerVariants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1 },
  exit: { opacity: 0 }
};

const Progress = () => {

  const navigate = useNavigate();
  const { user } = useUserStore();
  const { t, i18n } = useTranslation();
  const [workoutHistory, setWorkoutHistory] = useState<WorkoutHistory[]>(() => {
    const saved = localStorage.getItem('workoutHistory');
    return saved ? JSON.parse(saved) : [];
  });
  const [weightUnit, setWeightUnit] = useState<'kg' | 'lb'>('kg');
  const [insights, setInsights] = useState<AIInsight[]>(() => {
    try {
      const saved = localStorage.getItem('aiInsights');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [isLoading, setIsLoading] = useState(true);
  const [showProPanel, setShowProPanel] = useState(false);
  const [isInsightsExpanded, setIsInsightsExpanded] = useState(true);
  const [processedWorkoutKeys, setProcessedWorkoutKeys] = useState<string[]>(() => {
    try {
      const saved = localStorage.getItem('processedWorkoutInsightKeys');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    if (!user?.isPro) {
      // For non-Pro users, do not show or fetch AI insights
      setInsights([]);
      setIsLoading(false);
      return;
    }
    generateAIInsightsForNewWorkouts();
  }, [workoutHistory, user]);

  const generateAIInsightsForNewWorkouts = async () => {
    try {
      // Determine which workouts are new and have progress
      const hasCompleted = (w: WorkoutHistory) => w.exercises.some(e => e.setsCompleted > 0);
      const makeKey = (w: WorkoutHistory) => `${w.date}|${w.muscleGroup}|${w.exercises.length}`;
      const allKeys = workoutHistory.map(makeKey);

      // Baseline: if no processed keys exist yet, mark current workouts as processed without generating insights (start tracking from now)
      if (processedWorkoutKeys.length === 0 && allKeys.length > 0) {
        setProcessedWorkoutKeys(allKeys);
        try { localStorage.setItem('processedWorkoutInsightKeys', JSON.stringify(allKeys)); } catch {}
        setIsLoading(false);
        return;
      }

      const newWorkouts = workoutHistory.filter(w => hasCompleted(w) && !processedWorkoutKeys.includes(makeKey(w)));

      if (!newWorkouts.length) {
        setIsLoading(false);
        return;
      }

      setIsLoading(true);
      const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
      const lang = i18n.language;
      const isArabic = lang?.startsWith('ar');

      const updatedInsights: AIInsight[] = [...insights];
      const updatedKeys: string[] = [...processedWorkoutKeys];

      // Process sequentially to avoid API rate spikes
      for (const w of newWorkouts) {
        const workoutKey = makeKey(w);
        const workoutData = {
          user: {
            name: user?.name,
            age: user?.age,
            gender: user?.gender,
            weight: user?.weight,
            height: user?.height,
            experienceLevel: user?.experienceLevel,
            injuries: (user as any)?.injuries ?? [],
          },
          workout: {
            date: w.date,
            muscleGroup: w.muscleGroup,
            completionPercentage: w.completionPercentage,
            exercises: w.exercises.map(e => ({
              name: e.name,
              musclesWorked: e.musclesWorked,
              setsCompleted: e.setsCompleted,
              reps: e.reps,
              weight: e.weight,
              volume: e.volume,
              restTime: e.restTime,
            }))
          }
        };

        const prompt = `${isArabic ? 'IMPORTANT SYSTEM INSTRUCTION: The app language is Arabic. Reply ONLY in Arabic (Egyptian dialect where natural). Use Arabic for all text.\n' : ''}You are an elite strength and conditioning coach. Produce exactly ONE practical coaching suggestion for the following single workout. Avoid generic advice.

        Requirements:
        - Tailor to the workout data; if possible, reference concrete numbers (e.g., rest seconds, set/reps, volume deltas)
        - Keep under 26 words
        - Provide a short topic tag and priority
        - Output JSON only

        Locale: ${lang}
        Workout Data:
        ${JSON.stringify(workoutData, null, 2)}

        JSON schema:
        {
          "insights": [
            {
              "text": "...",
              "tag": "...",
              "priority": "high"
            }
          ]
        }`;

        try {
          const result = await model.generateContent(prompt);
          const response = result.response.text();
          const cleaned = response.replace(/```json\n?|\n?```/g, '').trim();
          const parsed = JSON.parse(cleaned);
          if (parsed?.insights && Array.isArray(parsed.insights) && parsed.insights.length) {
            const first = parsed.insights[0];
            const entry: AIInsight = {
              text: first.text ?? first.title ?? '',
              tag: first.tag,
              priority: first.priority as AIInsight['priority'],
              title: first.title,
              workoutKey,
              workoutDate: w.date,
              muscleGroup: w.muscleGroup,
            };
            if (entry.text && entry.text.trim().length > 0) {
              updatedInsights.unshift(entry); // newest on top
              updatedKeys.push(workoutKey);
            }
          }
        } catch (err) {
          console.error('Failed to generate insight for workout', w.date, err);
          // Skip on error; do not block others
          continue;
        }
      }

      setInsights(updatedInsights);
      setProcessedWorkoutKeys(updatedKeys);
      try {
        localStorage.setItem('processedWorkoutInsightKeys', JSON.stringify(updatedKeys));
      } catch {}
      try {
        localStorage.setItem('aiInsights', JSON.stringify(updatedInsights));
      } catch {}
    } finally {
      setIsLoading(false);
    }
  };

  // Get recent workouts - only include workouts with completed sets
  const getRecentWorkouts = () => {
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
    
    return workoutHistory
      .filter(workout => {
        // Only include workouts that have at least one completed exercise
        const hasCompletedExercises = workout.exercises.some(exercise => exercise.setsCompleted > 0);
        return new Date(workout.date) >= sevenDaysAgo && hasCompletedExercises;
      })
      .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
  };

  return (
    <AnimatePresence mode="sync" initial={false}>
      <motion.div 
        key="progress-page"
        className="h-full"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.4 }}
      >
        <div className="h-full overflow-y-auto -webkit-overflow-scrolling-touch">
          <div className="container mx-auto space-y-8 p-6 pb-24 max-w-[1920px]">
            <div className="flex flex-col gap-3">
              <motion.h1 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.4 }}
                className="text-[1.75rem] tracking-tight text-black font-sf-display font-sf-bold"
              >
                {t('plan.progress')}
              </motion.h1>

              {user && (
                <div className="flex justify-center w-full -mt-1">
                  <div className="bg-gray-100 rounded-full p-1 flex items-center shadow-md border border-gray-200/50">
                    <Link
                      to="/plan"
                      className="px-5 py-2 rounded-full flex items-center gap-2.5 transition-all duration-200 hover:bg-white/70"
                    >
                      <span className="text-sm font-medium text-gray-600">{t('plan.plan')}</span>
                      <ClipboardList className="w-4 h-4 text-gray-600" />
                    </Link>
                    <Link
                      to="/workouts"
                      className="px-5 py-2 rounded-full flex items-center gap-2.5 transition-all duration-200 hover:bg-white/70"
                    >
                      <span className="text-sm font-medium text-gray-600">{t('plan.library')}</span>
                      <Dumbbell className="w-4 h-4 text-gray-600" />
                    </Link>
                    <Link
                      to="/progress"
                      className="px-5 py-2 rounded-full bg-white shadow-sm flex items-center gap-2.5 transition-all duration-200 border border-gray-100"
                    >
                      <span className="text-sm font-medium text-gray-900">{t('plan.progress')}</span>
                      <BarChart3 className="w-4 h-4 text-primary" />
                    </Link>
                  </div>
                </div>
              )}
            </div>

            {/* AI Insights Section (Pro only) */}
            {user?.isPro && (
            <motion.div
              variants={containerVariants}
              initial="hidden"
              animate="visible"
              exit="exit"
              transition={{ duration: 0.4 }}
              className="space-y-6"
            >
              <motion.div 
                className="bg-white shadow-lg border border-black/5 rounded-2xl overflow-hidden relative"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.4 }}
                layout
              >

                <div 
                  className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-gray-50 transition-colors"
                  onClick={() => setIsInsightsExpanded(!isInsightsExpanded)}
                >
                  <div className="bg-gradient-to-br from-primary/10 to-primary/5 p-2.5 rounded-xl">
                    <Check className="h-5 w-5 text-primary" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-[17px] font-semibold text-gray-900 tracking-tight">{t('progress.ai_training_insights')}</h3>
                    <p className="text-[13px] text-gray-600 mt-0.5">{t('progress.ai_training_insights_desc')}</p>
                  </div>
                  <ChevronRight 
                    className={cn(
                      "w-5 h-5 text-gray-400 transition-transform duration-200",
                      isInsightsExpanded ? "rotate-90" : ""
                    )}
                  />
                </div>

                <AnimatePresence>
                  {isInsightsExpanded && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3 }}
                      className="border-t border-gray-100"
                    >
                      <div className="p-4">
                        {isLoading ? (
                          <div className="p-2">
                            <div className="bg-gradient-to-br from-primary/5 to-primary/10 rounded-2xl border border-primary/10 p-5">
                              <div className="flex items-center gap-3 mb-4">
                                <div className="h-10 w-10 rounded-xl bg-primary/20 animate-pulse" />
                                <div className="flex-1 space-y-2">
                                  <div className="h-4 w-1/3 rounded-full bg-primary/20 animate-pulse"></div>
                                  <div className="h-3 w-2/3 rounded-full bg-primary/10 animate-pulse"></div>
                                </div>
                              </div>
                              <div className="space-y-3">
                                <div className="h-3 rounded-full bg-gray-200/80 animate-pulse w-[92%]"></div>
                                <div className="h-3 rounded-full bg-gray-200/70 animate-pulse w-[87%]"></div>
                                <div className="h-3 rounded-full bg-gray-200/60 animate-pulse w-[78%]"></div>
                                <div className="h-3 rounded-full bg-gray-200/50 animate-pulse w-[64%]"></div>
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className="space-y-4">
                            <div className="bg-gray-50 border border-black/5 rounded-xl p-4">
                              <ol className="space-y-3 list-decimal list-inside">
                                {insights.map((insight, idx) => (
                                  <li key={(insight.tag ?? insight.title ?? idx).toString()} className="text-gray-800">
                                    <span className="text-gray-800">{insight.text}</span>
                                  </li>
                                ))}
                              </ol>
                            </div>

                            <div className="flex flex-wrap gap-2">
                              {insights.slice(0, 4).map((i, idx) => {
                                const label = i.tag
                                  || (i.title ? i.title : (i.text?.split(' ').slice(0, 2).join(' ') || `Tip ${idx+1}`));
                                return (
                                  <span
                                    key={(i.tag ?? i.title ?? idx).toString()}
                                    className="px-2.5 py-1 text-xs rounded-full bg-white border border-gray-200 text-gray-700"
                                  >
                                    {label}
                                  </span>
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            </motion.div>
            )}

            {/* Recent Workouts Section */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.4 }}
              className="space-y-4"
            >
<div className="space-y-4">
                {getRecentWorkouts().length === 0 ? (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="bg-white shadow-lg border border-black/5 rounded-2xl p-8 flex flex-col items-center justify-center"
                  >
                    <div className="bg-gray-100 p-4 rounded-full mb-4">
                      <Dumbbell className="w-8 h-8 text-gray-400" />
                    </div>
                    <h3 className="text-lg font-medium text-gray-900 mb-1">{t('progress.no_workouts_found')}</h3>
                    <p className="text-gray-600 text-sm text-center mb-6">{t('progress.no_workouts_desc')}</p>
                    <Button
                      onClick={() => navigate('/plan')}
                      className="bg-black text-white hover:bg-black/90"
                    >
                      {t('plan.back_to_plan')}
                    </Button>
                  </motion.div>
                ) : (
                  getRecentWorkouts().map((workout, index) => (
                    <motion.div
                      key={workout.date}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: index * 0.1 }}
                      className="bg-white shadow-lg border border-black/5 rounded-2xl p-4 hover:shadow-xl transition-shadow duration-300"
                    >
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h4 className="text-lg font-semibold text-gray-900">{workout.muscleGroup}</h4>
                          <p className="text-sm text-gray-600">
                            {new Date(workout.date).toLocaleDateString(document.dir === 'rtl' ? 'ar-SA' : 'en-US', {
                              weekday: 'long',
                              month: 'long',
                              day: 'numeric'
                            })}
                          </p>
                        </div>
                        <div className="text-right">
                          <div className="text-lg font-semibold text-gray-900">
                            {workout.completionPercentage}%
                          </div>
                          <p className="text-sm text-gray-600">{t('progress.completion')}</p>
                        </div>
                      </div>

                      <div className="space-y-3">
                        {workout.exercises.map((exercise, exerciseIndex) => (
                          <div 
                            key={exerciseIndex}
                            className="bg-gray-50 rounded-lg p-3 border border-black/5"
                          >
                            <div className="flex items-center justify-between mb-2">
                              <h5 className="font-medium text-gray-900">{exercise.name}</h5>
                              <span className="text-sm text-gray-600">
                                {exercise.setsCompleted}/{exercise.totalSets} {t('progress.sets')}
                              </span>
                            </div>
                            <div className="grid grid-cols-3 gap-2 text-sm mb-2">
                              <div>
                                <span className="text-gray-600">{t('progress.volume')}</span>
                                <p className="font-medium text-gray-900">{exercise.volume} {weightUnit}</p>
                              </div>
                              <div>
                                <span className="text-gray-600">{t('plan.exercise.weight')}</span>
                                <p className="font-medium text-gray-900">{exercise.weight} {weightUnit}</p>
                              </div>
                              <div>
                                <span className="text-gray-600">{t('plan.exercise.rest')}</span>
                                <p className="font-medium text-gray-900">{exercise.restTime}s</p>
                              </div>
                            </div>
                            <div className="grid grid-cols-3 gap-2 text-sm">
                              <div>
                                <span className="text-gray-600">{t('plan.exercise.reps')}</span>
                                <p className="font-medium text-gray-900">{exercise.reps}</p>
                              </div>
                              {exercise.rpm && (
                                <div>
                                  <span className="text-gray-600">{t('plan.workout.rpm')}</span>
                                  <p className="font-medium text-gray-900">{exercise.rpm}</p>
                                </div>
                              )}
                              <div>
                                <span className="text-gray-600">{t('progress.completion')}</span>
                                <div className="flex items-center gap-2">
                                  <span className="font-medium text-gray-900">
                                    {Math.round((exercise.setsCompleted / exercise.totalSets) * 100)}%
                                  </span>
                                  <div className="w-full bg-gray-200 rounded-full h-2">
                                    <div
                                      className="h-full bg-gradient-to-r from-primary to-primary/80 rounded-full"
                                    />
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  ))
                )}
              </div>
            </motion.div>
          </div>
        </div>
      </motion.div>

      {/* Pro Subscription Panel */}
      <ProSubscriptionPanel 
        isOpen={showProPanel} 
        onClose={() => setShowProPanel(false)} 
      />
    </AnimatePresence>
  );
};

export default Progress; 