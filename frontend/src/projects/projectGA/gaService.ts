/**
 * GA API Service - Handles all API calls to the genetic algorithm backend
 */

import axios, { AxiosInstance } from "axios";
import { getApiUrl } from "../getApiUrl.ts";

export interface Individual {
  genes: string;
  fitness: number;
}

export interface GAConfig {
  population_size: number;
  mutation_rate: number;
  crossover_rate: number;
  elitism_count: number;
  tournament_size: number;
  charset: string;
}

export interface StatusResponse {
  session_id: string;
  initialized: boolean;
  generation: number;
  population_size: number;
  target: string;
  best_fitness: number;
  best_individual: string;
  average_fitness: number;
  is_complete: boolean;
  config: GAConfig;
}

export interface EvolutionResponse {
  session_id: string;
  generation: number;
  best_fitness: number;
  best_individual: string;
  average_fitness: number;
  is_complete: boolean;
  generations_evolved: number;
}

export interface PopulationResponse {
  session_id: string;
  generation: number;
  top_individuals: Individual[];
  average_fitness: number;
  best_fitness: number;
}

export interface HistoryDataPoint {
  generation: number;
  best_fitness: number;
  average_fitness: number;
  worst_fitness: number;
  best_individual: string;
}

export interface HistoryResponse {
  session_id: string;
  target: string;
  history: HistoryDataPoint[];
}

export interface QuotesResponse {
  quotes: string[];
}

class GAService {
  private api: AxiosInstance;
  private baseURL: string;

  constructor() {
    this.baseURL = getApiUrl();
    this.api = axios.create({
      baseURL: this.baseURL,
      headers: {
        "Content-Type": "application/json",
      },
    });
  }

  /**
   * Initialize a new GA session
   */
  async initialize(
    target: string,
    populationSize: number = 200,
    mutationRate: number = 0.01,
    crossoverRate: number = 0.8,
    elitismCount: number = 2,
    tournamentSize: number = 5,
    charset: string = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
  ): Promise<StatusResponse> {
    try {
      const response = await this.api.post<StatusResponse>(
        "/projectGA/initialize",
        {
          target,
          population_size: populationSize,
          mutation_rate: mutationRate,
          crossover_rate: crossoverRate,
          elitism_count: elitismCount,
          tournament_size: tournamentSize,
          charset,
        }
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Evolve the population for N generations
   */
  async evolve(
    sessionId: string,
    generations: number = 1
  ): Promise<EvolutionResponse> {
    try {
      const response = await this.api.post<EvolutionResponse>(
        `/projectGA/evolve/${sessionId}`,
        { generations }
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get current GA status
   */
  async getStatus(sessionId: string): Promise<StatusResponse> {
    try {
      const response = await this.api.get<StatusResponse>(
        `/projectGA/status/${sessionId}`
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get top individuals from population
   */
  async getPopulation(
    sessionId: string,
    topN: number = 20
  ): Promise<PopulationResponse> {
    try {
      const response = await this.api.get<PopulationResponse>(
        `/projectGA/population/${sessionId}`,
        { params: { top_n: topN } }
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get evolution history
   */
  async getHistory(sessionId: string): Promise<HistoryResponse> {
    try {
      const response = await this.api.get<HistoryResponse>(
        `/projectGA/history/${sessionId}`
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Reset a GA session
   */
  async reset(sessionId: string): Promise<{ success: boolean; message: string }> {
    try {
      const response = await this.api.post(`/projectGA/reset/${sessionId}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Delete a GA session
   */
  async deleteSession(
    sessionId: string
  ): Promise<{ success: boolean; message: string }> {
    try {
      const response = await this.api.delete(`/projectGA/session/${sessionId}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get Shakespeare quotes
   */
  async getQuotes(): Promise<QuotesResponse> {
    try {
      const response = await this.api.get<QuotesResponse>("/projectGA/quotes");
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Handle API errors
   */
  private handleError(error: any): Error {
    if (axios.isAxiosError(error)) {
      if (error.response) {
        return new Error(
          error.response.data?.detail || error.message || "API Error"
        );
      } else if (error.request) {
        return new Error("No response from server");
      }
    }
    return error instanceof Error ? error : new Error("Unknown error occurred");
  }
}

// Export singleton instance
export const gaService = new GAService();
