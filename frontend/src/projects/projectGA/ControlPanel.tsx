import React, { useState, useEffect } from "react";
import { gaService } from "./gaService.ts";
import "./styles/ControlPanel.css";

interface ControlPanelProps {
  onStart: (target: string, config: any) => void;
  onStop: () => void;
  onReset: () => void;
  isRunning: boolean;
  isComplete: boolean;
}

const ControlPanel: React.FC<ControlPanelProps> = ({
  onStart,
  onStop,
  onReset,
  isRunning,
  isComplete,
}) => {
  const [selectedQuote, setSelectedQuote] = useState("TO BE OR NOT TO BE");
  const [customPhrase, setCustomPhrase] = useState("");
  const [useCustom, setUseCustom] = useState(false);
  const [quotes, setQuotes] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  // Parameters
  const [populationSize, setPopulationSize] = useState(200);
  const [mutationRate, setMutationRate] = useState(0.01);
  const [crossoverRate, setCrossoverRate] = useState(0.8);
  const [elitismCount, setElitismCount] = useState(2);
  const [tournamentSize, setTournamentSize] = useState(5);

  // Load quotes on component mount
  useEffect(() => {
    const loadQuotes = async () => {
      try {
        const data = await gaService.getQuotes();
        setQuotes(data.quotes);
        setLoading(false);
      } catch (error) {
        console.error("Failed to load quotes:", error);
        setLoading(false);
      }
    };
    loadQuotes();
  }, []);

  const handleStart = () => {
    const target = useCustom ? customPhrase.toUpperCase() : selectedQuote;

    if (!target.trim()) {
      alert("Please enter a target phrase");
      return;
    }

    const config = {
      population_size: populationSize,
      mutation_rate: mutationRate,
      crossover_rate: crossoverRate,
      elitism_count: elitismCount,
      tournament_size: tournamentSize,
    };

    onStart(target, config);
  };

  return (
    <div className="control-panel">
      <div className="control-section">
        <h3>Target Phrase</h3>

        {/* Quote Selection */}
        <div className="quote-selector">
          <label>
            <input
              type="radio"
              checked={!useCustom}
              onChange={() => setUseCustom(false)}
            />
            Select Shakespeare Quote
          </label>
          <select
            value={selectedQuote}
            onChange={(e) => setSelectedQuote(e.target.value)}
            disabled={useCustom || loading}
          >
            {loading ? (
              <option>Loading quotes...</option>
            ) : (
              quotes.map((quote) => (
                <option key={quote} value={quote}>
                  {quote}
                </option>
              ))
            )}
          </select>
        </div>

        {/* Custom Phrase */}
        <div className="custom-phrase">
          <label>
            <input
              type="radio"
              checked={useCustom}
              onChange={() => setUseCustom(true)}
            />
            Custom Phrase
          </label>
          <input
            type="text"
            value={customPhrase}
            onChange={(e) => setCustomPhrase(e.target.value.toUpperCase())}
            placeholder="Enter your target phrase..."
            disabled={!useCustom}
            maxLength={50}
          />
        </div>
      </div>

      {/* Parameters */}
      <div className="control-section">
        <h3>GA Parameters</h3>

        <div className="parameter">
          <label>
            Population Size: <strong>{populationSize}</strong>
          </label>
          <input
            type="range"
            min="10"
            max="500"
            step="10"
            value={populationSize}
            onChange={(e) => setPopulationSize(parseInt(e.target.value))}
            disabled={isRunning}
          />
          <span className="hint">10 - 500</span>
        </div>

        <div className="parameter">
          <label>
            Mutation Rate: <strong>{mutationRate.toFixed(3)}</strong>
          </label>
          <input
            type="range"
            min="0"
            max="0.1"
            step="0.001"
            value={mutationRate}
            onChange={(e) => setMutationRate(parseFloat(e.target.value))}
            disabled={isRunning}
          />
          <span className="hint">0.000 - 0.100</span>
        </div>

        <div className="parameter">
          <label>
            Crossover Rate: <strong>{crossoverRate.toFixed(2)}</strong>
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={crossoverRate}
            onChange={(e) => setCrossoverRate(parseFloat(e.target.value))}
            disabled={isRunning}
          />
          <span className="hint">0.00 - 1.00</span>
        </div>

        <div className="parameter">
          <label>
            Elitism Count: <strong>{elitismCount}</strong>
          </label>
          <input
            type="range"
            min="0"
            max="20"
            step="1"
            value={elitismCount}
            onChange={(e) => setElitismCount(parseInt(e.target.value))}
            disabled={isRunning}
          />
          <span className="hint">0 - 20</span>
        </div>

        <div className="parameter">
          <label>
            Tournament Size: <strong>{tournamentSize}</strong>
          </label>
          <input
            type="range"
            min="2"
            max="20"
            step="1"
            value={tournamentSize}
            onChange={(e) => setTournamentSize(parseInt(e.target.value))}
            disabled={isRunning}
          />
          <span className="hint">2 - 20</span>
        </div>
      </div>

      {/* Controls */}
      <div className="control-section button-group">
        <button
          className="btn btn-primary"
          onClick={handleStart}
          disabled={isRunning || isComplete}
        >
          {isRunning ? "Running..." : "Start Evolution"}
        </button>

        <button
          className="btn btn-danger"
          onClick={onStop}
          disabled={!isRunning}
        >
          Stop
        </button>

        <button
          className="btn btn-secondary"
          onClick={onReset}
          disabled={isRunning}
        >
          Reset
        </button>
      </div>

      {/* Status */}
      {isComplete && (
        <div className="status-complete">✅ Evolution Complete!</div>
      )}
    </div>
  );
};

export default ControlPanel;
