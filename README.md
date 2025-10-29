# AI Hub - CST435 Project Hub

A full-stack web application that integrates multiple AI/ML projects (RNN Text Generation, CNN Image Classification, etc.) into a single hub. Built with React/TypeScript frontend and FastAPI/Python backend.

## 🏗️ Project Architecture

```
hub-app/
├── frontend/                 # React + TypeScript frontend
│   ├── src/
│   │   ├── App.tsx          # Main hub component (component-based)
│   │   ├── App.js           # Alternative router-based hub
│   │   ├── App.css          # Shared styling
│   │   ├── components/      # Shared components
│   │   └── projects/        # Project-specific components
│   │       ├── projectRNN/  # RNN Text Generator
│   │       └── Project3/    # CNN Image Classifier
│   └── package.json
│
├── backend/                  # FastAPI + Python backend
│   ├── main.py              # FastAPI app entry point
│   └── app/
│       └── routers/
│           ├── projectRNN/  # RNN module
│           └── Project3/    # CNN module
│
├── requirements.txt         # Python dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Installation

1. **Clone/navigate to the project directory**
   ```bash
   cd hub-app
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

### Running the Application

**Terminal 1 - Start Backend (from hub-app directory):**
```bash
uvicorn backend.main:app --reload --host localhost --port 8000
```

Backend will be available at: `http://localhost:8000`
API docs available at: `http://localhost:8000/docs`

**Terminal 2 - Start Frontend (from hub-app/frontend directory):**
```bash
npm start
```

Frontend will be available at: `http://localhost:3000`

---

## 📋 Current Projects

### 1. RNN Text Generator ✨
- **Location:** `frontend/src/projects/projectRNN/` & `backend/app/routers/projectRNN/`
- **Technology:** LSTM-based RNN using TensorFlow/Keras
- **Features:**
  - Generate text from seed phrases
  - Adjust output length (10-200 words)
  - Control creativity with temperature (0.1-2.0)
  - Real-time generation with loading states
- **Training Data:** Alice in Wonderland
- **Endpoint:** `POST /generate-text`

### 2. CNN Image Classifier 🖼️
- **Location:** `frontend/src/projects/Project3/` & `backend/app/routers/Project3/`
- **Technology:** CNN using TensorFlow/Keras
- **Features:**
  - Classify vehicle images (cars, airplanes, motorbikes)
  - View confidence scores and probability distribution
  - Drag-and-drop image upload
  - Real-time classification results
- **Training Data:** Kaggle Natural Images dataset (older vehicle images)
- **Endpoint:** `POST /classify-image`

---

## 🛠️ Adding New Projects to the Hub

Follow these steps to add a new ML/AI project to the hub:

### Step 1: Backend Setup

#### 1.1 Create Backend Module
Create a new directory under `backend/app/routers/`:
```bash
mkdir backend/app/routers/ProjectX  # Replace X with project number
```

#### 1.2 Create Core ML Files
In your new `ProjectX` directory, create necessary python files for data processing and model building.


#### 1.3 Add FastAPI Endpoint
Update `backend/main.py`:

```python
# At the top with other imports
from pydantic import BaseModel

# Add response model for your project
class YourProjectResponse(BaseModel):
    prediction: str
    confidence: float
    # Add other fields as needed

# Add the endpoint (at the end of main.py)
@app.post("/your-endpoint", response_model=YourProjectResponse)
async def your_endpoint(file: UploadFile = File(...)):
    """Your project endpoint description"""
    from backend.app.routers.ProjectX.classifier import get_classifier

    try:
        image_bytes = await file.read()
        classifier = get_classifier()
        result = classifier.predict(image_bytes)

        return YourProjectResponse(
            prediction=result["prediction"],
            confidence=result["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
```

### Step 2: Frontend Setup

#### 2.1 Create Project Component
Create `frontend/src/projects/ProjectX/YourComponent.tsx`:

```typescript
import React, { useState } from 'react';

function YourProjectComponent() {
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (data: any) => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch('http://localhost:8000/your-endpoint', {
        method: 'POST',
        body: data,  // Adjust based on your API
      });

      if (!response.ok) {
        throw new Error('Request failed');
      }

      const result = await response.json();
      setResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="form">
      <h2 className="title">🎯 Your Project Name</h2>

      {/* Project Description */}
      <div style={{
      }}>
        <p style={{ margin: 0 }}>
          <strong>About this project:</strong> Your project description here.
          Explain what the model does and how to use it.
        </p>
      </div>

      {/* Your component content */}
      {/* ... */}

      {/* Results Display */}
      {result && (
        <div className="output">
          <h3>Results</h3>
          {/* Display your results here */}
        </div>
      )}
    </div>
  );
}

export default YourProjectComponent;
```

#### 2.2 Update App Navigation

**Option A: Update App.js (Router-based)**
```javascript
import YourComponent from './projects/ProjectX/YourComponent.tsx';

function App() {
  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <h1>CST435 Project Hub</h1>
          <nav>
            <ul>
              <li><Link to="/">RNN Text Generator</Link></li>
              <li><Link to="/image-classifier">Image Classifier</Link></li>
              <li><Link to="/your-project">Your Project</Link></li>
            </ul>
          </nav>
        </header>
        <main>
          <Routes>
            <Route path="/your-project" element={<YourComponent />} />
            {/* ... other routes ... */}
          </Routes>
        </main>
      </div>
    </Router>
  );
}
```

**Option B: Update App.tsx (Component-based)**
```typescript
const projects = [
  { id: 'rnn', name: 'RNN Text Generator', icon: '✨', description: 'Generate text with LSTM' },
  { id: 'cnn', name: 'Image Classifier', icon: '🖼️', description: 'Classify vehicle images' },
  { id: 'yourProject', name: 'Your Project', icon: '🎯', description: 'Your description' },
];

// In the projects loop:
{activeProject === 'yourProject' && <YourComponent />}
```

### Step 3: Styling

Use the existing CSS classes for consistency:
- `className="form"` - Main container
- `className="title"` - Heading
- `className="form-group"` - Form section
- `className="textarea"` or `className="input"` - Form controls
- `className="button"` - Primary button
- `className="button secondary"` - Secondary button
- `className="output"` - Output container
- `className="generated-text"` - Text output box

See `frontend/src/App.css` for all available styles.

### Step 4: Update requirements.txt
Add any new Python dependencies:
```bash
pip install <new-package>
pip freeze > requirements.txt
```

### Step 5: Test Your Project

1. Start the backend: `uvicorn backend.main:app --reload`
2. Start the frontend: `npm start` (from frontend directory)
3. Navigate to your project in the hub
4. Test all functionality

---

## 📁 File Structure Reference

### For a Complete Project (Like CNN Image Classifier)
```
backend/app/routers/ProjectX/
├── classifier.py          # Main ML logic
├── data_processor.py      # Input preprocessing
├── model.keras           # Trained model file
├── requirements.txt      # Project-specific dependencies
└── README.md            # Project documentation

frontend/src/projects/ProjectX/
├── YourComponent.tsx     # Main React component
└── api.ts               # Optional: API helper functions
```

---

## 🔧 Common Tasks

### Adding a New Trained Model
1. Place your model file in `backend/app/routers/ProjectX/`
2. Update the model path in your classifier's `load_model()` method
3. Restart the backend

### Changing Project Description
- **TextGenerator:** `frontend/src/projects/projectRNN/TextGenerator.tsx` (line ~96)
- **ImageClassifier:** `frontend/src/projects/Project3/ImageClassifier.tsx` (line ~109)

### Updating API Endpoints
1. Modify endpoint in `backend/main.py`
2. Update request/response models
3. Update frontend API calls to match new endpoint signature

### Styling New Components
Copy the style pattern from existing projects:
```typescript
<div style={{
  backgroundColor: '#f0f4ff',
  border: '2px solid #667eea',
  borderRadius: '12px',
  padding: '16px',
  marginBottom: '25px'
}}>
  Your content
</div>
```

---

## 🔌 API Reference

### Current Endpoints

**Text Generation:**
```
POST /generate-text
Request: { seed_text: string, num_words: int, temperature: float }
Response: { generated_text: string, seed_text: string, num_words: int, temperature: float }
```

**Image Classification:**
```
POST /classify-image
Request: multipart file upload
Response: { predicted_class: string, confidence: float, class_probabilities: object }
```

**Model Info:**
```
GET /model/info
Response: { vocab_size: int, sequence_length: int, ... }
```

**Health Check:**
```
GET /health
Response: { status: string, model_loaded: boolean }
```

---

## 🐛 Troubleshooting

### Backend won't start
- Ensure all dependencies in `requirements.txt` are installed
- Check that port 8000 is not in use: `lsof -i :8000`
- Verify model files exist in the correct locations

### Frontend won't load components
- Check console for import errors
- Verify file paths in imports match actual file locations
- Ensure TypeScript has no type errors: `npm run build`

### Model not found errors
- Verify model file exists at the path specified in your classifier
- Check file permissions
- Ensure model format matches what `tf.keras.models.load_model()` expects

### CORS errors
- Verify frontend is running on `localhost:3000`
- Check CORS middleware in `backend/main.py` allows your frontend origin

---

## 📚 Technologies Used

**Frontend:**
- React 19.2.0
- TypeScript
- Axios/Fetch API
- TailwindCSS + Custom CSS

**Backend:**
- FastAPI
- TensorFlow/Keras
- NumPy
- Pydantic

**ML/Data:**
- LSTM for NLP
- CNN for Computer Vision
- scikit-learn for preprocessing

---

## 📝 License & Credits

CST-435/AIT-204 Project Hub by Nolan and John

---

## ✅ Checklist for New Projects

- [ ] Backend directory created under `backend/app/routers/ProjectX`
- [ ] Trained model loaded successfully
- [ ] FastAPI endpoint added to `backend/main.py`
- [ ] Frontend component created
- [ ] Component imported in App.js/App.tsx
- [ ] Navigation link added
- [ ] Project description added
- [ ] Styling applied using App.css classes
- [ ] Tested locally on localhost:3000
- [ ] requirements.txt updated if new packages added
