import React from "react";
import { Individual } from "./gaService.ts";
import "./styles/PopulationDisplay.css";

interface PopulationDisplayProps {
  population: Individual[];
  target: string;
  generation: number;
  bestEver: Individual | null;
  averageFitness: number;
  isComplete: boolean;
}

const PopulationDisplay: React.FC<PopulationDisplayProps> = ({
  population,
  target,
  generation,
  bestEver,
  averageFitness,
  isComplete,
}) => {
  const renderGeneWithColor = (gene: string) => {
    return (
      <div className="gene-display">
        {gene.split("").map((char, idx) => {
          const isMatch = idx < target.length && char === target[idx];
          return (
            <span
              key={idx}
              className={`gene-char ${isMatch ? "match" : "mismatch"}`}
              title={isMatch ? `✓ Matches '${target[idx]}'` : `✗ Should be '${target[idx]}'`}
            >
              {char}
            </span>
          );
        })}
      </div>
    );
  };

  return (
    <div className="population-display">
      <div className="header-info">
        <div className="info-card">
          <div className="info-label">Generation</div>
          <div className="info-value">{generation}</div>
        </div>
        <div className="info-card">
          <div className="info-label">Best Fitness</div>
          <div className="info-value">{bestEver?.fitness.toFixed(2) || 0}%</div>
        </div>
        <div className="info-card">
          <div className="info-label">Avg Fitness</div>
          <div className="info-value">{averageFitness.toFixed(2)}%</div>
        </div>
        <div className="info-card">
          <div className="info-label">Population</div>
          <div className="info-value">{population.length}</div>
        </div>
      </div>

      <div className="target-section">
        <h3>Target</h3>
        <div className="target-phrase">{target}</div>
      </div>

      <div className="best-individual-section">
        <h3>Best Individual</h3>
        {bestEver ? (
          <div className="best-individual">
            <div className="best-fitness-bar">
              <div
                className="fitness-fill"
                style={{ width: `${bestEver.fitness}%` }}
              />
              <span className="fitness-text">{bestEver.fitness.toFixed(2)}%</span>
            </div>
            {renderGeneWithColor(bestEver.genes)}
          </div>
        ) : (
          <div className="no-data">No individual yet</div>
        )}
      </div>

      <div className="population-section">
        <h3>Top 20 Individuals</h3>
        <div className="population-list">
          {population.length === 0 ? (
            <div className="no-data">No population data</div>
          ) : (
            population.map((individual, idx) => (
              <div key={idx} className="population-item">
                <div className="item-rank">#{idx + 1}</div>
                <div className="item-fitness-bar">
                  <div
                    className="fitness-fill"
                    style={{ width: `${individual.fitness}%` }}
                  />
                  <span className="fitness-text">
                    {individual.fitness.toFixed(2)}%
                  </span>
                </div>
                <div className="item-genes">{renderGeneWithColor(individual.genes)}</div>
              </div>
            ))
          )}
        </div>
      </div>

    </div>
  );
};

export default PopulationDisplay;
