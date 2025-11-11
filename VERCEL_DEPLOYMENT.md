# Deploying to Vercel

Vercel is the platform created by the Next.js team and is optimized for React applications. This guide covers deploying your React frontend to Vercel while your backend runs on Railway or another platform.

## Prerequisites

- GitHub account with your code pushed
- Vercel account (https://vercel.com)
- Backend already deployed (Railway, Render, or another platform)
- Basic Git knowledge

## Why Vercel for Frontend?

- ⚡ Optimized for React/Next.js
- 🚀 Instant deployments on push
- 🔄 Automatic preview deployments
- 💰 Generous free tier
- 📊 Built-in analytics
- 🌍 Global CDN

## Step 1: Prepare Your Repository

1. **Ensure code is pushed to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push origin main
   ```

2. **Create a `.env.production` file** (optional, but recommended):
   ```
   REACT_APP_API_URL=https://your-backend-xxx.railway.app
   ```
   (Replace with your actual backend URL)

## Step 2: Deploy to Vercel

### Method 1: Using Vercel Dashboard (Easiest)

1. **Go to Vercel.com:**
   - Sign in with GitHub
   - Click "Add New..." → "Project"
   - Click "Import Git Repository"

2. **Select Your Repository:**
   - Find and select your `hub-app` (or whatever your repo name is)
   - Click "Import"

3. **Configure Project Settings:**
   
   **Build & Development Settings:**
   - **Framework Preset:** React
   - **Build Command:** `cd frontend && npm run build`
   - **Output Directory:** `frontend/build`
   - **Install Command:** `npm install`
   
   **Root Directory:**
   - Keep as `.` (root)

4. **Set Environment Variables:**
   - In the "Environment Variables" section, add:
     ```
     REACT_APP_API_URL = https://your-backend-xxx.railway.app
     ```
   - Replace with your actual backend URL
   - Select environments: Production, Preview, Development

5. **Deploy:**
   - Click "Deploy"
   - Vercel will build and deploy your frontend
   - You'll get a deployment URL (e.g., `https://hub-app.vercel.app`)

### Method 2: Using vercel.json (Advanced)

Create `vercel.json` in your project root:

```json
{
  "buildCommand": "cd frontend && npm run build",
  "outputDirectory": "frontend/build",
  "env": {
    "REACT_APP_API_URL": "@react-app-api-url"
  },
  "envPrefix": "REACT_APP_"
}
```

Then push to GitHub and deploy as above.

## Step 3: Update Frontend API Configuration

Make sure your frontend can use the production backend URL:

1. **Check `frontend/src/apiConfig.ts`:**
   ```typescript
   const getApiUrl = (): string => {
     if (process.env.NODE_ENV === 'production') {
       // Use environment variable set in Vercel
       return process.env.REACT_APP_API_URL || 'http://localhost:8000';
     }
     return 'http://localhost:8000';
   };

   export default getApiUrl;
   ```

2. **Update all API calls:**
   - If you have any hardcoded URLs like `http://localhost:8000`, replace them with:
   ```typescript
   import getApiUrl from '../apiConfig';
   const apiUrl = getApiUrl();
   fetch(`${apiUrl}/your-endpoint`);
   ```

3. **Check for localhost references:**
   ```bash
   # Search for hardcoded API URLs
   grep -r "localhost:8000" frontend/src/
   grep -r "http://localhost" frontend/src/
   ```

## Step 4: Connect Domains (Optional)

1. **Custom Domain:**
   - Go to your Vercel project settings
   - Click "Domains"
   - Add your custom domain (e.g., `hub.yourdomain.com`)
   - Follow DNS instructions

2. **Subdomain Example:**
   - Point `hub.yourdomain.com` to your Vercel deployment
   - DNS CNAME: `cname.vercel.com`

## Step 5: Verify Deployment

1. **Check Frontend:**
   - Visit your Vercel URL (e.g., `https://hub-app.vercel.app`)
   - Your React app should load and display correctly

2. **Test Features:**
   - Try using one of the projects (text generation, image classification, etc.)
   - Check browser console (F12) for any errors
   - Verify API calls are reaching your backend

3. **Check Logs:**
   - In Vercel dashboard, go to "Deployments"
   - Click your latest deployment
   - View logs to troubleshoot any issues

## Step 6: Set Up Continuous Deployment

Vercel automatically deploys when you push to your connected GitHub branch:

1. **Push to Main Branch:**
   ```bash
   git add .
   git commit -m "Update frontend"
   git push origin main
   ```

2. **Vercel will automatically:**
   - Build your frontend
   - Deploy to production
   - Show deployment in dashboard

3. **Preview Deployments:**
   - Every pull request gets an auto-generated preview URL
   - Share with team for review
   - Merge PR to deploy to production

## Troubleshooting

### Build fails with "not found"
- Check that your project root directory contains both `frontend/` and other files
- Verify `Build Command` is: `cd frontend && npm run build`
- Verify `Output Directory` is: `frontend/build`

### Frontend won't connect to backend
- Check CORS headers in your backend
- Verify `REACT_APP_API_URL` is set correctly in Vercel
- Check that backend service is running and accessible
- Look at browser console Network tab to see failed requests

### White screen on load
- Check browser console (F12) for JavaScript errors
- Check Vercel deployment logs
- Verify `npm build` works locally: `cd frontend && npm run build`

### API returns 404
- Backend service may be down
- Wrong API URL in environment variable
- Check backend logs in Railway/your hosting platform

## Environment Variables Setup

To manage different environments:

1. **Development (Local):**
   - `.env.local`: `REACT_APP_API_URL=http://localhost:8000`

2. **Production (Vercel):**
   - Set in Vercel dashboard
   - `REACT_APP_API_URL=https://your-backend-url.railway.app`

3. **Push to git:**
   ```bash
   echo "REACT_APP_API_URL=http://localhost:8000" > frontend/.env.local
   git add frontend/.env.local  # Or add to .gitignore if you prefer
   ```

## Monitoring & Analytics

1. **Real-time Metrics:**
   - Vercel dashboard shows real-time traffic
   - View in "Analytics" tab

2. **Core Web Vitals:**
   - Monitor performance
   - See which pages need optimization

3. **Error Tracking:**
   - Check deployment logs
   - Monitor browser console errors

## Cost & Limits

**Vercel Free Plan:**
- Unlimited deployments
- 100 GB bandwidth/month
- Serverless function execution time: 10s (free tier)
- Typical cost for this app: **$0/month**

**When You Might Need Pro:**
- Increased bandwidth needs
- Custom analytics
- Advanced security features

## Advanced: Monorepo with Workspaces

If you want better monorepo support, create `package.json` in root:

```json
{
  "workspaces": [
    "frontend",
    "backend"
  ]
}
```

Then in `vercel.json`:
```json
{
  "buildCommand": "npm run build --prefix frontend",
  "outputDirectory": "frontend/build"
}
```

## Next Steps

1. ✅ Deploy frontend to Vercel
2. ✅ Verify backend is running on Railway
3. ✅ Test API connectivity
4. ✅ Set up custom domain (optional)
5. ✅ Monitor deployments and errors
6. Consider: Caching, optimization, analytics

## Useful Links

- Vercel Docs: https://vercel.com/docs
- Vercel GitHub Integration: https://vercel.com/docs/git/vercel-for-github
- React Deployment: https://create-react-app.dev/docs/deployment/
- Environment Variables: https://vercel.com/docs/concepts/projects/environment-variables

---

## Quick Reference

| Step | Command/Action |
|------|---|
| 1. | Push code to GitHub: `git push origin main` |
| 2. | Go to vercel.com and import repo |
| 3. | Set build command: `cd frontend && npm run build` |
| 4. | Set output directory: `frontend/build` |
| 5. | Add env var: `REACT_APP_API_URL` |
| 6. | Click Deploy |
| 7. | Update frontend with API URL |
| 8. | Test at your Vercel URL |
