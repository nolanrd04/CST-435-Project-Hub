# Deploying to Railway

Railway is a modern platform for deploying full-stack applications. This guide covers deploying both your backend (FastAPI) and frontend (React) to Railway.

## Prerequisites

- GitHub account with your code pushed
- Railway account (https://railway.app)
- Basic understanding of git and GitHub

## Step 1: Prepare Your Repository

1. **Ensure your code is pushed to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for Railway deployment"
   git push origin main
   ```

2. **Verify your file structure:**
   ```
   hub-app/
   ├── backend/
   │   ├── main.py
   │   └── requirements.txt
   ├── frontend/
   │   ├── package.json
   │   └── src/
   └── requirements.txt
   ```

## Step 2: Create Railway Services

### Backend Service (FastAPI)

1. **Go to Railway.app:**
   - Log in with GitHub
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Connect your GitHub account
   - Select your `hub-app` repository

2. **Configure Backend Service:**
   - After selecting your repo, Railway will detect it's a monorepo
   - Click "Add a service"
   - Select "GitHub repo" (or use existing connection)
   - Railway may auto-detect it as Python - that's correct

3. **Set Environment Variables (in Railway dashboard):**
   - Go to the project settings
   - Click on the backend service
   - Go to "Variables" tab
   - Add these variables:
     ```
     PYTHONUNBUFFERED=1
     PORT=8000
     ```

4. **Configure Build & Deploy:**
   - In the service settings, go to the "Deploy" tab
   - Set the following:
     - **Root Directory:** `/` (leave as is)
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`

5. **Enable Public Networking:**
   - In service settings, go to the "Networking" tab
   - Enable "Public Networking"
   - Note the generated URL (e.g., `https://your-backend-xxx.railway.app`)

### Frontend Service (React)

1. **Add a new service to the same Railway project:**
   - Click "Add a service" → "GitHub repo"
   - Select the same repository

2. **Configure Frontend Service:**
   - Set the following:
     - **Root Directory:** `/` (leave as is)
     - **Build Command:** `cd frontend && npm install && npm run build`
     - **Start Command:** `cd frontend && npx serve -s build -l 3000`

3. **Set Environment Variables:**
   - Go to the frontend service "Variables" tab
   - Add:
     ```
     PORT=3000
     REACT_APP_API_URL=https://your-backend-xxx.railway.app
     ```
   - Replace `your-backend-xxx.railway.app` with your actual backend URL from Step 4.5

4. **Enable Public Networking:**
   - Same as backend - go to "Networking" tab and enable

## Step 3: Update Frontend API Configuration

Your frontend needs to know the backend URL in production.

1. **Update `frontend/src/apiConfig.ts`** (or create if it doesn't exist):
   ```typescript
   const getApiUrl = (): string => {
     if (process.env.NODE_ENV === 'production') {
       return process.env.REACT_APP_API_URL || 'http://localhost:8000';
     }
     return 'http://localhost:8000';
   };

   export default getApiUrl;
   ```

2. **Update API calls in your components:**
   ```typescript
   import getApiUrl from '../apiConfig';

   const apiUrl = getApiUrl();
   const response = await fetch(`${apiUrl}/your-endpoint`, {
     method: 'POST',
     // ... rest of fetch config
   });
   ```

3. **Update `frontend/package.json` to include serve:**
   ```json
   {
     "dependencies": {
       "serve": "^14.2.0"
     }
   }
   ```

## Step 4: Deploy

1. **Push your changes:**
   ```bash
   git add .
   git commit -m "Configure for Railway deployment"
   git push origin main
   ```

2. **Railway will auto-deploy:**
   - Once you push, Railway automatically deploys both services
   - Check the deployment status in the Railway dashboard

3. **View Logs:**
   - Go to each service → "Deployments" tab
   - Click the latest deployment to see logs

## Step 5: Verify Deployment

1. **Check Backend:**
   - Visit `https://your-backend-xxx.railway.app/docs`
   - You should see the FastAPI interactive documentation

2. **Check Frontend:**
   - Visit `https://your-frontend-xxx.railway.app`
   - Your React app should load
   - Try using one of the project features

3. **Check CORS:**
   - If you get CORS errors, verify your backend CORS settings allow your frontend domain
   - In `backend/main.py`, update allowed origins if needed

## Troubleshooting

### Backend won't start
- Check logs: Railway dashboard → service → Deployments
- Common issues:
  - Missing dependencies: ensure `requirements.txt` is up to date
  - Port conflicts: use `$PORT` environment variable
  - Python version: Railway defaults to Python 3.11

### Frontend won't load components
- Check browser console for errors
- Verify `REACT_APP_API_URL` environment variable is set correctly
- Check that backend URL is accessible

### API calls failing
- Verify backend service is running (check Railway logs)
- Ensure frontend has correct `REACT_APP_API_URL`
- Check CORS settings in `backend/main.py`

### Files too large
- Railway has file upload limits
- Remove unnecessary files from git:
  - Add to `.gitignore`: `*.keras`, `*.h5`, `__pycache__/`, `node_modules/`
  - Use Git LFS for large model files if needed

## Advanced: Using railway.json

Alternatively, create a `railway.json` file in your project root:

```json
{
  "services": [
    {
      "name": "backend",
      "runtime": "python@3.11",
      "buildCommand": "pip install -r requirements.txt",
      "startCommand": "cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT",
      "envVars": {
        "PYTHONUNBUFFERED": "1"
      }
    },
    {
      "name": "frontend",
      "runtime": "node@18",
      "buildCommand": "cd frontend && npm install && npm run build",
      "startCommand": "cd frontend && npx serve -s build -l 3000",
      "envVars": {
        "PORT": "3000"
      }
    }
  ]
}
```

## Cost & Limits

Railway Free Plan (as of 2024):
- **Monthly credit:** $5
- **Services:** Pay-as-you-go
- **Typical cost for this app:** $0-5/month

## Next Steps

- Monitor your deployment in Railway dashboard
- Set up automatic deployments on git push (already enabled)
- Configure custom domain if desired
- Set up monitoring and alerts

---

For more help: https://docs.railway.app/
