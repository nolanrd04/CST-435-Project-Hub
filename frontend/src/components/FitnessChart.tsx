import React, { useEffect, useRef } from "react";
import { HistoryDataPoint } from "../projects/projectGA/gaService.ts";
import "../styles/FitnessChart.css";

interface FitnessChartProps {
  data: HistoryDataPoint[];
}

const FitnessChart: React.FC<FitnessChartProps> = ({ data }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || data.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const padding = 60;
    const width = canvas.width - padding * 2;
    const height = canvas.height - padding * 2;

    // Clear canvas
    ctx.fillStyle = "#f5f5f5";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid and axes
    ctx.strokeStyle = "#ddd";
    ctx.lineWidth = 1;

    // Draw y-axis labels and gridlines
    for (let i = 0; i <= 10; i++) {
      const y = padding + (height * i) / 10;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(canvas.width - padding, y);
      ctx.stroke();

      ctx.fillStyle = "#666";
      ctx.font = "12px Arial";
      ctx.textAlign = "right";
      ctx.fillText(`${100 - i * 10}%`, padding - 10, y + 4);
    }

    // Draw axes
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, canvas.height - padding);
    ctx.lineTo(canvas.width - padding, canvas.height - padding);
    ctx.stroke();

    // Draw axis labels
    ctx.fillStyle = "#333";
    ctx.font = "14px Arial";
    ctx.textAlign = "center";
    ctx.fillText("Generation", canvas.width / 2, canvas.height - 20);

    ctx.save();
    ctx.translate(20, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("Fitness (%)", 0, 0);
    ctx.restore();

    // Plot data
    if (data.length < 2) return;

    const maxGeneration = data[data.length - 1].generation || 1;
    const genWidth = width / Math.max(maxGeneration, 1);

    // Draw best fitness line
    ctx.strokeStyle = "#4CAF50";
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((point, idx) => {
      const x = padding + ((point.generation - 1) * genWidth || 0);
      const y =
        canvas.height - padding - (point.best_fitness / 100) * height;
      if (idx === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Draw average fitness line
    ctx.strokeStyle = "#2196F3";
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((point, idx) => {
      const x = padding + ((point.generation - 1) * genWidth || 0);
      const y =
        canvas.height - padding - (point.average_fitness / 100) * height;
      if (idx === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Draw legend
    ctx.fillStyle = "#4CAF50";
    ctx.fillRect(canvas.width - 200, 20, 12, 12);
    ctx.fillStyle = "#333";
    ctx.font = "12px Arial";
    ctx.textAlign = "left";
    ctx.fillText("Best Fitness", canvas.width - 180, 28);

    ctx.fillStyle = "#2196F3";
    ctx.fillRect(canvas.width - 200, 40, 12, 12);
    ctx.fillStyle = "#333";
    ctx.fillText("Average Fitness", canvas.width - 180, 48);
  }, [data]);

  return (
    <div className="fitness-chart">
      <h3>Evolution Progress</h3>
      <canvas
        ref={canvasRef}
        width={800}
        height={400}
        className="chart-canvas"
      />
      {data.length === 0 && (
        <div className="no-data-message">
          Run evolution to see chart data
        </div>
      )}
    </div>
  );
};

export default FitnessChart;
