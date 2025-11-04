# Genetic Algorithms in NLP - Educational Activity Package

A comprehensive class activity package for teaching genetic algorithms through an interactive Shakespeare text evolution project. This repository contains everything needed to deliver a hands-on learning experience about evolutionary computation and its applications in Natural Language Processing.

## 📚 What's Included

### 1. Interactive HTML Handout
**File:** `genetic-algorithm-handout.html`

An interactive educational handout:
- Visual explanation of genetic algorithm concepts
- Animated SVG diagrams showing the GA lifecycle
- Interactive browser-based demo (evolve "HELLO WORLD")
- Real-time visualization of evolution in action
- Comprehensive coverage of selection, crossover, and mutation
- Activity overview and learning objectives

**Usage:** Open directly in any web browser for classroom presentation or student reference.

### 2. Complete React Application
**Directory:** `shakespeare-ga/`

A production-ready React + TypeScript application demonstrating genetic algorithms by evolving random text into Shakespeare quotes.

**Features:**
- Real-time evolution visualization with color-coded character matching
- Customizable GA parameters (population size, mutation rate, elitism)
- 10 pre-loaded Shakespeare quotes + custom phrase input
- Fitness progress chart with HTML5 Canvas
- Population display showing top individuals
- Responsive design for desktop and mobile
- Deployment-ready configuration for Vercel and Render.com

**Technical Stack:**
- React 19 with TypeScript
- Custom genetic algorithm implementation
- HTML5 Canvas for charting
- CSS3 with gradients and animations
- No external dependencies for GA logic

### 3. Step-by-Step Student Instructions
**File:** `STUDENT-INSTRUCTIONS.md`

Comprehensive guide for students including:
- Prerequisites and setup instructions
- 8 detailed implementation steps
- Code templates with TODO sections
- Testing guidelines
- Deployment instructions for Vercel and Render.com
- Extension challenges for advanced students
- Troubleshooting guide
- Grading rubric
- Learning outcomes

## 🎯 Learning Objectives

Students will:
1. Understand core genetic algorithm concepts (fitness, selection, crossover, mutation)
2. Implement evolutionary computation from scratch
3. Build a complete React application with TypeScript
4. Visualize algorithm performance in real-time
5. Deploy a web application to production
6. Apply GAs to solve NLP optimization problems

## 🧬 Why This Problem?

The activity focuses on evolving random text into Shakespeare quotes because it perfectly demonstrates GA advantages:

**The Challenge:**
- Target phrase "TO BE OR NOT TO BE" has 27^18 ≈ 7.6 × 10^25 possible combinations
- Brute force at 1 billion attempts/second: ~2.4 billion years
- Genetic algorithm: Typically solves in <1000 generations (~seconds)

**Why GAs Excel:**
- Massive search space makes exhaustive search impossible
- Simple fitness function (character matching)
- Population diversity explores multiple solutions simultaneously
- Evolutionary pressure converges on optimal solution
- No domain-specific knowledge required

## 🚀 Quick Start

### For Instructors

1. **Present the concepts:**
   ```bash
   open genetic-algorithm-handout.html
   ```

2. **Demo the working application:**
   ```bash
   cd shakespeare-ga
   npm install
   npm start
   ```
   Open http://localhost:3000

3. **Share with students:**
   - Provide link to this repository
   - Direct them to STUDENT-INSTRUCTIONS.md
   - Set implementation deadline

### For Students

1. **Clone the repository:**
   ```bash
   git clone [repository-url]
   cd AIT-204-genetic-alg
   ```

2. **Review the handout:**
   ```bash
   open genetic-algorithm-handout.html
   ```

3. **Follow the instructions:**
   Read `STUDENT-INSTRUCTIONS.md` and complete the activity

4. **Reference the working solution:**
   The `shakespeare-ga/` directory contains a complete implementation for reference

## 📦 Project Structure

```
AIT-204-genetic-alg/
├── README.md                           # This file
├── genetic-algorithm-handout.html      # Interactive educational handout
├── STUDENT-INSTRUCTIONS.md             # Step-by-step implementation guide
└── shakespeare-ga/                     # Complete React application
    ├── package.json
    ├── vercel.json                     # Vercel deployment config
    ├── render.yaml                     # Render.com deployment config
    ├── public/                         # Static assets
    └── src/
        ├── App.tsx                     # Main application component
        ├── App.css                     # Application styles
        ├── types/
        │   └── Individual.ts           # Individual/Chromosome class
        ├── algorithms/
        │   └── GeneticAlgorithm.ts     # Core GA implementation
        ├── components/
        │   ├── ControlPanel.tsx        # User controls
        │   ├── PopulationDisplay.tsx   # Evolution visualization
        │   └── FitnessChart.tsx        # Progress chart
        └── styles/
            ├── ControlPanel.css
            ├── PopulationDisplay.css
            └── FitnessChart.css
```

## 🎓 Class Activity Flow

### Session 1: Introduction (60 minutes)
1. Present genetic algorithm concepts using the HTML handout (20 min)
2. Demo the interactive browser evolution (10 min)
3. Show the completed React application (10 min)
4. Explain the activity requirements (10 min)
5. Students begin setup and Part 1 (10 min)

### Session 2-3: Implementation (120 minutes)
Students work through Parts 2-6 of STUDENT-INSTRUCTIONS.md:
- Implement Individual class
- Build GeneticAlgorithm class
- Create React components
- Add styling
- Test functionality

### Session 4: Deployment & Extensions (60 minutes)
- Deploy to Vercel or Render.com (20 min)
- Work on extension challenges (30 min)
- Prepare submission report (10 min)

## 🌐 Deployment Options

### Option 1: Vercel (Recommended)
```bash
cd shakespeare-ga
npm install -g vercel
vercel --prod
```

### Option 2: Render.com
1. Push code to GitHub
2. Connect repository at https://render.com
3. Configure as Static Site:
   - Build: `npm install && npm run build`
   - Publish: `build`

### Option 3: GitHub Pages
```bash
cd shakespeare-ga
npm install gh-pages --save-dev
npm run build
npx gh-pages -d build
```

## 🔧 Genetic Algorithm Parameters

Students can experiment with:
- **Population Size:** 50-500 (default: 200)
- **Mutation Rate:** 0-10% (default: 1%)
- **Crossover Rate:** 0-100% (default: 80%)
- **Elitism Count:** 0-10 (default: 2)
- **Tournament Size:** Fixed at 5

## 📊 Expected Results

Typical performance for "TO BE OR NOT TO BE":
- Population 100, Mutation 1%: ~300-500 generations
- Population 200, Mutation 1%: ~200-400 generations
- Population 500, Mutation 1%: ~150-300 generations

Lower mutation rates converge faster but may get stuck.
Higher mutation rates maintain diversity but converge slower.

## 🎨 Customization Ideas

### For Instructors:
1. Add more Shakespeare quotes
2. Include difficulty ratings for phrases
3. Create competition leaderboard
4. Add performance benchmarks

### For Students (Extensions):
1. Implement different selection methods (roulette wheel, rank-based)
2. Add two-point or uniform crossover
3. Implement adaptive mutation rates
4. Create parallel populations with migration
5. Add sound effects and animations
6. Export evolution statistics to CSV

## 📝 Assessment Rubric

| Criterion | Points | Description |
|-----------|--------|-------------|
| Functionality | 40 | GA correctly evolves text to target |
| Code Quality | 25 | Well-organized, documented, follows best practices |
| UI/UX | 20 | Intuitive interface, visual appeal, responsiveness |
| Deployment | 10 | Successfully deployed and accessible |
| Report | 5 | Quality of analysis and insights |
| **Total** | **100** | |

## 🐛 Troubleshooting

**Issue:** Evolution gets stuck at 90% fitness
**Solution:** Increase mutation rate slightly (1.5-2%)

**Issue:** Evolution is too slow
**Solution:** Reduce population size or increase elitism count

**Issue:** Module not found errors
**Solution:** Ensure all import paths use correct casing and file extensions

**Issue:** Deployment build fails
**Solution:** Check that all dependencies are in package.json, run `npm run build` locally first

## 📚 Additional Resources

### Genetic Algorithms
- [Introduction to Evolutionary Computing](https://link.springer.com/book/10.1007/978-3-662-44874-8) - Eiben & Smith
- [Genetic Algorithms in Search, Optimization, and Machine Learning](https://www.amazon.com/Genetic-Algorithms-Optimization-Machine-Learning/dp/0201157675) - Goldberg

### React & TypeScript
- [React Documentation](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [React TypeScript Cheatsheet](https://react-typescript-cheatsheet.netlify.app/)

### Deployment
- [Vercel Documentation](https://vercel.com/docs)
- [Render Documentation](https://render.com/docs)

## 🤝 Contributing

This is an educational project. Contributions welcome:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Suggested improvements:
- Additional algorithm implementations (PSO, simulated annealing)
- More visualization options
- Performance optimizations
- Accessibility enhancements

## 📄 License

MIT License - Free for educational use

Copyright (c) 2025 AIT-204 Course Materials

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## 👨‍🏫 About

Created for AIT-204: Genetic Algorithms and Evolutionary Computation
Grand Canyon University

**Course Topics:**
- Evolutionary computation fundamentals
- Genetic algorithms and genetic programming
- Swarm intelligence and particle optimization
- Neural evolution and neuroevolution
- Applications in optimization, machine learning, and AI

## 📧 Contact & Support

For questions or issues:
- Open an issue in this repository
- Contact course instructor
- Check STUDENT-INSTRUCTIONS.md troubleshooting section

## 🌟 Acknowledgments

- Inspired by Daniel Shiffman's "Nature of Code"
- Based on classic genetic algorithm literature
- Built with React and TypeScript
- Deployed on Vercel/Render platforms

---

**Ready to evolve?** Start with `genetic-algorithm-handout.html` and watch random text become Shakespeare! 🎭🧬