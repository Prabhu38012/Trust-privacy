import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuthStore } from './store/authStore'
import Layout from './components/Layout'
import Landing from './pages/Landing'
import Login from './pages/Auth/Login'
import Signup from './pages/Auth/Signup'
import Dashboard from './pages/Dashboard'
import DeepfakeScan from './pages/DeepfakeScan'
<<<<<<< HEAD
import DocumentAnalysis from './pages/DocumentAnalysis'
=======
>>>>>>> 4336965e78d04836c64348343ce98ab69529cd81
import Certificate from './pages/Certificate'
import Admin from './pages/Admin'
import Settings from './pages/Settings'
import Certificates from './pages/Certificates'

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuthStore()
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" />
}

function PublicRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuthStore()
  return !isAuthenticated ? <>{children}</> : <Navigate to="/dashboard" />
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<PublicRoute><Landing /></PublicRoute>} />
      <Route path="/login" element={<PublicRoute><Login /></PublicRoute>} />
      <Route path="/signup" element={<PublicRoute><Signup /></PublicRoute>} />
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route index element={<Dashboard />} />
        <Route path="scan" element={<DeepfakeScan />} />
<<<<<<< HEAD
        <Route path="document" element={<DocumentAnalysis />} />
=======
>>>>>>> 4336965e78d04836c64348343ce98ab69529cd81
        <Route path="certificate" element={<Certificate />} />
        <Route path="admin" element={<Admin />} />
        <Route path="settings" element={<Settings />} />
        <Route path="certificates" element={<Certificates />} />
        <Route path="/dashboard/certificates" element={<Certificates />} />
      </Route>
      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  )
}

export default App
