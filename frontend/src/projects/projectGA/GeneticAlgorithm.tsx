import React, { useState, useEffect, useRef } from "react";
import ControlPanel from "./ControlPanel.tsx";
import PopulationDisplay from "./PopulationDisplay.tsx";
import FitnessChart from "./FitnessChart.tsx";
import {
  gaService,
  Individual,
  HistoryDataPoint,
} from "./gaService.ts";
import "./GeneticAlgorithm.css";

interface AppState {
  sessionId: string | null;
  isInitialized: boolean;
  isRunning: boolean;
  isComplete: boolean;
  generation: number;
  population: Individual[];
  target: string;
  bestIndividual: Individual | null;
  averageFitness: number;
  bestFitness: number;
  historyData: HistoryDataPoint[];
  error: string | null;
}

const GeneticAlgorithmPage: React.FC = () => {
  const [state, setState] = useState<AppState>({
    sessionId: null,
    isInitialized: false,
    isRunning: false,
    isComplete: false,
    generation: 0,
    population: [],
    target: "",
    bestIndividual: null,
    averageFitness: 0,
    bestFitness: 0,
    historyData: [],
    error: null,
  });

  const evolutionLoopRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isEvolving = useRef(false);
  const stateRef = useRef(state);  // Keep a ref to latest state

  // Update the ref whenever state changes
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  // Start evolution
  const handleStart = async (target: string, config: any) => {
    try {
      console.log(`🚀 Starting GA with target: "${target}", config:`, config);
      setState((prev) => ({ ...prev, error: null }));

      const response = await gaService.initialize(
        target,
        config.population_size,
        config.mutation_rate,
        config.crossover_rate,
        config.elitism_count,
        config.tournament_size
      );

      setState((prev) => ({
        ...prev,
        sessionId: response.session_id,
        isInitialized: true,
        isRunning: true,
        isComplete: false,
        generation: 0,
        target: response.target,
        bestIndividual: {
          genes: response.best_individual,
          fitness: response.best_fitness,
        },
        averageFitness: response.average_fitness,
        bestFitness: response.best_fitness,
        historyData: [],
      }));
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Failed to initialize GA";
      setState((prev) => ({
        ...prev,
        error: errorMessage,
        isRunning: false,
      }));
    }
  };

  // Evolution loop
  useEffect(() => {
    if (!state.isRunning || !state.sessionId) return;

    let isMounted = true;
    let consecutiveErrors = 0;

    const evolveStep = async () => {
      if (isEvolving.current || !isMounted) return;
      
      isEvolving.current = true;
      try {
        const currentState = stateRef.current;
        console.log(`🧬 Starting evolution step at generation ${currentState.generation}...`);
        
        // Evolve for 50 generations per step (faster convergence)
        const evolveResult = await gaService.evolve(currentState.sessionId!, 50);
        
        if (!isMounted) return;

        console.log("✅ Evolution step completed", {
          generation: evolveResult.generation,
          bestFitness: evolveResult.best_fitness,
          generationsEvolved: evolveResult.generations_evolved,
          isComplete: evolveResult.is_complete,
        });

        // Get current status
        const status = await gaService.getStatus(currentState.sessionId!);

        if (!isMounted) return;

        console.log("📊 Status:", status);

        // Get population data
        const popResponse = await gaService.getPopulation(currentState.sessionId!, 20);

        // Get history
        const historyResponse = await gaService.getHistory(currentState.sessionId!);

        if (!isMounted) return;

        console.log(`📈 Generation ${status.generation}, Fitness: ${status.best_fitness}%`);

        setState((prev) => ({
          ...prev,
          generation: status.generation,
          bestIndividual: {
            genes: status.best_individual,
            fitness: status.best_fitness,
          },
          averageFitness: status.average_fitness,
          bestFitness: status.best_fitness,
          population: popResponse.top_individuals,
          historyData: historyResponse.history,
          isComplete: status.is_complete,
          isRunning: !status.is_complete,
        }));

        consecutiveErrors = 0; // Reset error counter on success

        if (status.is_complete) {
          console.log("🎉 GA completed!");
          isEvolving.current = false;
          if (evolutionLoopRef.current) {
            clearInterval(evolutionLoopRef.current);
            evolutionLoopRef.current = null;
          }
        } else {
          isEvolving.current = false;
        }
      } catch (error) {
        if (!isMounted) return;
        
        consecutiveErrors++;
        console.error(`❌ Evolution error (${consecutiveErrors}):`, error);
        
        const errorMessage =
          error instanceof Error ? error.message : "Evolution error";
        setState((prev) => ({
          ...prev,
          error: `${errorMessage} (attempt ${consecutiveErrors})`,
          isRunning: false,
        }));
        
        isEvolving.current = false;
        
        // Stop if we have too many consecutive errors
        if (consecutiveErrors >= 3) {
          if (evolutionLoopRef.current) {
            clearInterval(evolutionLoopRef.current);
            evolutionLoopRef.current = null;
          }
        }
      }
    };

    // Run evolution every 500ms
    evolutionLoopRef.current = setInterval(evolveStep, 500);

    return () => {
      isMounted = false;
      if (evolutionLoopRef.current) {
        clearInterval(evolutionLoopRef.current);
        evolutionLoopRef.current = null;
      }
    };
  }, [state.isRunning, state.sessionId]);

  // Stop evolution
  const handleStop = () => {
    setState((prev) => ({ ...prev, isRunning: false }));
    if (evolutionLoopRef.current) {
      clearInterval(evolutionLoopRef.current);
    }
    isEvolving.current = false;
  };

  // Reset evolution
  const handleReset = async () => {
    if (state.sessionId) {
      try {
        await gaService.reset(state.sessionId);
        setState((prev) => ({
          ...prev,
          isInitialized: false,
          isRunning: false,
          isComplete: false,
          generation: 0,
          population: [],
          target: "",
          bestIndividual: null,
          averageFitness: 0,
          bestFitness: 0,
          historyData: [],
          error: null,
        }));
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : "Reset failed";
        setState((prev) => ({ ...prev, error: errorMessage }));
      }
    }
  };

  return (
    <div className="ga-page">
      <div className="ga-header">
        <h1>Genetic Algorithm</h1>
        <p>Evolve random text into Shakespeare quotes</p>
      </div>

      {state.error && (
        <div className="error-message">
          <strong>Error:</strong> {state.error}
        </div>
      )}

      <div className="ga-container">
        <div className="ga-left">
          <ControlPanel
            onStart={handleStart}
            onStop={handleStop}
            onReset={handleReset}
            isRunning={state.isRunning}
            isComplete={state.isComplete}
          />
        </div>

        <div className="ga-right">
          <PopulationDisplay
            population={state.population}
            target={state.target}
            generation={state.generation}
            bestEver={state.bestIndividual}
            averageFitness={state.averageFitness}
            isComplete={state.isComplete}
          />

          <FitnessChart data={state.historyData} />
        </div>
      </div>
    </div>
  );
};

export default GeneticAlgorithmPage;
