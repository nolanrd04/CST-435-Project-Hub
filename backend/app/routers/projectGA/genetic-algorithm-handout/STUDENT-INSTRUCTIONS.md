# Step-by-Step Instructions: Building a Genetic Algorithm React App

## Overview
In this activity, you will build a React application that uses genetic algorithms to evolve random text into Shakespeare quotes. This hands-on project will help you understand how genetic algorithms work and why they're particularly effective for certain NLP problems.

## Prerequisites
- Basic knowledge of JavaScript/TypeScript
- Familiarity with React concepts
- Node.js installed on your computer
- A GitHub account (for deployment)
- A text editor (VS Code recommended)

## Part 1: Project Setup (15 minutes)

### Step 1: Create Your React Application
Open your terminal and run:
```bash
npx create-react-app shakespeare-ga --template typescript
cd shakespeare-ga
```

### Step 2: Install Dependencies
The create-react-app command has already installed all necessary dependencies. Verify by running:
```bash
npm start
```
You should see the default React app at `http://localhost:3000`

### Step 3: Clean Up Default Files
Stop the development server (Ctrl+C) and remove unnecessary files:
```bash
rm src/logo.svg src/App.test.tsx src/setupTests.ts src/reportWebVitals.ts
```

## Part 2: Implement the Genetic Algorithm Core (30 minutes)

### Step 4: Create the Individual Class
Create a new file `src/types/Individual.ts`:

```typescript
export interface Individual {
  genes: string;
  fitness: number;
}

export class IndividualClass implements Individual {
  genes: string;
  fitness: number;

  constructor(genes: string, fitness: number = 0) {
    this.genes = genes;
    this.fitness = fitness;
  }

  static createRandom(length: number, charset: string): IndividualClass {
    // TODO: Implement random individual creation
    // Hint: Loop through length and pick random characters from charset
  }

  calculateFitness(target: string): void {
    // TODO: Calculate fitness as percentage of matching characters
    // Hint: Compare each character position and count matches
  }

  mutate(mutationRate: number, charset: string): IndividualClass {
    // TODO: Create a new individual with random mutations
    // Hint: For each character, randomly decide if it should mutate
  }
}
```

**Your Task**: Complete the three methods marked with TODO

### Step 5: Create the Genetic Algorithm Class
Create a new file `src/algorithms/GeneticAlgorithm.ts`:

```typescript
import { IndividualClass } from '../types/Individual';

export interface GAConfig {
  populationSize: number;
  mutationRate: number;
  crossoverRate: number;
  elitismCount: number;
  tournamentSize: number;
  charset: string;
}

export class GeneticAlgorithm {
  private config: GAConfig;
  private population: IndividualClass[];
  private generation: number;
  private target: string;
  private bestEver: IndividualClass | null;

  constructor(config: GAConfig) {
    this.config = config;
    this.population = [];
    this.generation = 0;
    this.target = '';
    this.bestEver = null;
  }

  initialize(target: string): void {
    // TODO: Create initial random population
  }

  evolve(): void {
    // TODO: Implement one generation of evolution
    // 1. Calculate fitness for all individuals
    // 2. Sort by fitness
    // 3. Apply elitism
    // 4. Create new population through selection, crossover, mutation
    // 5. Increment generation counter
  }

  private selectParent(): IndividualClass {
    // TODO: Implement tournament selection
    // Hint: Pick random individuals and return the fittest
  }

  private crossover(parent1: IndividualClass, parent2: IndividualClass): IndividualClass[] {
    // TODO: Implement single-point crossover
    // Hint: Pick a random point and swap genes after that point
  }

  // Add getter methods for population, generation, best individual, etc.
}
```

**Your Task**: Implement the core genetic algorithm logic

## Part 3: Build the React Components (30 minutes)

### Step 6: Create the Control Panel Component
Create `src/components/ControlPanel.tsx`:

```typescript
import React, { useState } from 'react';

interface ControlPanelProps {
  onStart: (target: string, config: any) => void;
  onStop: () => void;
  onReset: () => void;
  isRunning: boolean;
  isComplete: boolean;
}

const ControlPanel: React.FC<ControlPanelProps> = (props) => {
  // TODO: Add state for target phrase and GA parameters
  // TODO: Create UI for:
  //   - Selecting Shakespeare quotes or custom input
  //   - Adjusting population size, mutation rate, elitism
  //   - Start/Stop/Reset buttons

  return (
    <div className="control-panel">
      {/* Your implementation here */}
    </div>
  );
};

export default ControlPanel;
```

### Step 7: Create the Population Display Component
Create `src/components/PopulationDisplay.tsx`:

```typescript
import React from 'react';
import { IndividualClass } from '../types/Individual';

interface PopulationDisplayProps {
  population: IndividualClass[];
  target: string;
  generation: number;
  bestEver: IndividualClass | null;
  averageFitness: number;
  isComplete: boolean;
}

const PopulationDisplay: React.FC<PopulationDisplayProps> = (props) => {
  // TODO: Display:
  //   - Current generation number
  //   - Best fitness score
  //   - Top 10-20 individuals with color coding
  //   - Target phrase
  //   - Completion message when done

  return (
    <div className="population-display">
      {/* Your implementation here */}
    </div>
  );
};

export default PopulationDisplay;
```

### Step 8: Create the Fitness Chart Component
Create `src/components/FitnessChart.tsx`:

```typescript
import React, { useEffect, useRef } from 'react';

interface DataPoint {
  generation: number;
  bestFitness: number;
  averageFitness: number;
}

interface FitnessChartProps {
  data: DataPoint[];
}

const FitnessChart: React.FC<FitnessChartProps> = ({ data }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    // TODO: Draw chart on canvas
    // - X-axis: Generation number
    // - Y-axis: Fitness (0-100%)
    // - Plot best fitness and average fitness lines
  }, [data]);

  return (
    <div className="fitness-chart">
      <canvas ref={canvasRef} width={800} height={300} />
    </div>
  );
};

export default FitnessChart;
```

## Part 4: Integrate Everything in App.tsx (20 minutes)

### Step 9: Update the Main App Component
Replace `src/App.tsx`:

```typescript
import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import { GeneticAlgorithm } from './algorithms/GeneticAlgorithm';
import ControlPanel from './components/ControlPanel';
import PopulationDisplay from './components/PopulationDisplay';
import FitnessChart from './components/FitnessChart';

function App() {
  // TODO: Add state management for:
  //   - GA instance
  //   - Current population
  //   - Generation number
  //   - Chart data
  //   - Running/Complete status

  // TODO: Implement functions:
  //   - startEvolution(): Initialize GA and start evolution loop
  //   - evolveGeneration(): Run one generation and update display
  //   - stopEvolution(): Stop the evolution loop
  //   - resetEvolution(): Reset everything to initial state

  return (
    <div className="App">
      {/* TODO: Layout with ControlPanel and PopulationDisplay */}
    </div>
  );
}

export default App;
```

## Part 5: Add Styling (15 minutes)

### Step 10: Create Component Styles
Create CSS files for each component:
- `src/styles/ControlPanel.css`
- `src/styles/PopulationDisplay.css`
- `src/styles/FitnessChart.css`

Add styling to make your application visually appealing. Focus on:
- Color coding for matching/non-matching characters
- Progress indicators
- Smooth animations for state changes
- Responsive layout for different screen sizes

## Part 6: Testing Your Application (10 minutes)

### Step 11: Test the Application
1. Start the development server:
   ```bash
   npm start
   ```

2. Test different scenarios:
   - Short phrases (5-10 characters)
   - Medium phrases (15-20 characters)
   - Long phrases (25+ characters)
   - Different mutation rates
   - Different population sizes

3. Verify that:
   - Evolution progresses toward the target
   - Fitness increases over generations
   - The app finds the solution
   - Charts update correctly

## Part 7: Deployment (20 minutes)

### Step 12: Prepare for Deployment
1. Build the production version:
   ```bash
   npm run build
   ```

2. Test the build locally:
   ```bash
   npx serve -s build
   ```

### Step 13: Deploy to Vercel

1. Create a Vercel account at https://vercel.com

2. Install Vercel CLI:
   ```bash
   npm install -g vercel
   ```

3. Deploy:
   ```bash
   vercel --prod
   ```

4. Follow the prompts:
   - Confirm project settings
   - Choose a project name
   - Wait for deployment

5. Your app will be live at: `https://[your-project-name].vercel.app`

### Alternative: Deploy to Render.com

1. Push your code to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/[your-username]/[repo-name].git
   git push -u origin main
   ```

2. Go to https://render.com and sign up

3. Click "New +" → "Static Site"

4. Connect your GitHub repository

5. Configure:
   - Build Command: `npm install && npm run build`
   - Publish Directory: `build`

6. Click "Create Static Site"

## Part 8: Extensions and Improvements (Optional)

### Challenge Tasks
Once your basic implementation is working, try these enhancements:

1. **Advanced Selection Methods**
   - Implement roulette wheel selection
   - Add rank-based selection
   - Compare performance of different methods

2. **Crossover Variations**
   - Implement two-point crossover
   - Try uniform crossover
   - Add crossover rate parameter

3. **Adaptive Parameters**
   - Implement adaptive mutation rate
   - Add simulated annealing
   - Auto-tune parameters based on progress

4. **Performance Optimizations**
   - Use Web Workers for evolution
   - Implement parallel populations
   - Add caching for fitness calculations

5. **Additional Features**
   - Save/load evolution state
   - Export statistics to CSV
   - Add sound effects for completion
   - Implement phrase difficulty rating

## Submission Requirements

### What to Submit
1. GitHub repository link with your complete code
2. Live deployment URL (Vercel or Render)
3. Brief report (1-2 pages) including:
   - Screenshot of your working application
   - Performance analysis (generations needed for different phrases)
   - One interesting observation about genetic algorithms
   - One challenge you faced and how you solved it

### Grading Criteria
- **Functionality (40%)**: Does the GA correctly evolve text?
- **Code Quality (25%)**: Is the code well-organized and documented?
- **UI/UX (20%)**: Is the interface intuitive and visually appealing?
- **Deployment (10%)**: Is the app successfully deployed?
- **Report (5%)**: Quality of analysis and insights

## Troubleshooting Guide

### Common Issues and Solutions

**Issue**: "Module not found" errors
**Solution**: Make sure all import paths are correct and files exist

**Issue**: Evolution gets stuck
**Solution**: Check mutation rate isn't too low, ensure selection pressure

**Issue**: App crashes with large populations
**Solution**: Limit population size or optimize memory usage

**Issue**: Deployment fails
**Solution**: Ensure all dependencies are in package.json, check build output

## Resources

### Documentation
- [React Documentation](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Vercel Deployment Guide](https://vercel.com/docs)
- [Render Documentation](https://render.com/docs)

### Genetic Algorithm Resources
- [Introduction to Genetic Algorithms](https://towardsdatascience.com/introduction-to-genetic-algorithms-76145c2a5d2a)
- [GA Visualization](https://www.youtube.com/watch?v=XP8R0yzAbdo)

### Getting Help
- Office hours: [Insert times]
- Discussion forum: [Insert link]
- Email: [Insert email]

## Learning Outcomes
By completing this activity, you will:
1. Understand the core concepts of genetic algorithms
2. Implement selection, crossover, and mutation operations
3. Build a complete React application with TypeScript
4. Deploy a web application to the cloud
5. Visualize algorithm performance in real-time
6. Apply GA to solve an NLP problem

Good luck with your implementation! Remember, genetic algorithms are all about iterative improvement - just like your coding skills!