# TrustLock - AI-Powered Deepfake & Fraud Detection Platform

A cutting-edge security platform with blockchain-based authenticity certificates.

## Features

- 🔍 **AI Deepfake Detection** - Detect manipulated images, videos, and audio
- ⛓️ **Blockchain Certificates** - Immutable authenticity verification
- 🛡️ **Fraud Prevention** - Real-time threat detection
- 🔐 **E2E Encryption** - Military-grade data protection
- 📊 **Privacy Analytics** - Differential privacy enabled
- 📋 **Audit Trails** - Complete transparency

## Tech Stack

- **Frontend**: React + Vite + TypeScript + Tailwind CSS
- **Backend**: Node.js + Express
- **Database**: MongoDB
- **Auth**: JWT with refresh tokens
- **Blockchain**: Ethereum (for certificates)

## Quick Start

### Prerequisites
- Node.js 18+
- MongoDB (local or Atlas)

### Installation

```bash
# Install dependencies
cd frontend && npm install
cd ../backend && npm install

# Configure backend
cp backend/.env.example backend/.env
# Edit backend/.env with your settings
```

### Running

**Terminal 1 - Backend:**
```bash
cd backend && npm run dev
```

**Terminal 2 - Frontend:**
```bash
cd frontend && npm run dev
```

Visit `http://localhost:5173`

## Deployment

### Backend (Render.com)

1. Push code to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Configure:
   - **Root Directory:** `backend`
   - **Build Command:** `npm install`
   - **Start Command:** `npm start`
5. Add environment variables:
   - `NODE_ENV=production`
   - `JWT_SECRET=<generate-secure-key>`
   - `JWT_REFRESH_SECRET=<generate-secure-key>`
   - `MONGO_URI=<your-mongodb-atlas-uri>`

### Frontend (Vercel)

1. Go to [vercel.com](https://vercel.com) → New Project
2. Import your GitHub repo
3. Configure:
   - **Root Directory:** `frontend`
   - **Framework Preset:** Vite
4. Add environment variable:
   - `VITE_API_URL=https://your-render-backend-url.onrender.com`

## Project Structure

```
TrustLock/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── store/
│   │   ├── lib/
│   │   └── App.tsx
│   └── package.json
├── backend/
│   ├── src/
│   │   ├── models/
│   │   ├── routes/
│   │   ├── middleware/
│   │   └── app.js
│   └── package.json
└── README.md
```

## Day 1 Features ✅
- [x] Project setup (Vite + React + Express)
- [x] User authentication (signup/login)
- [x] JWT with refresh tokens
- [x] Protected routes
- [x] Basic dashboard UI
- [x] Audit logging foundation

## Upcoming Features
- Day 2: Consent management UI + backend
- Day 3: End-to-end encrypted notes
- Day 4: Data Subject Requests (export/delete)
- Day 5: Audit logs UI + admin view
- Day 6: Privacy analytics with differential privacy
- Day 7-10: Polish, testing, deployment

## Security Features
- Password hashing with bcrypt (12 rounds)
- JWT access tokens (15min expiry)
- Secure HTTP headers with Helmet
- Rate limiting on API endpoints
- CORS configuration
- Audit logging for sensitive actions

## License
MIT
