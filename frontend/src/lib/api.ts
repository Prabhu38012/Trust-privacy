import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 300000, // 5 minutes for large uploads
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    const stored = localStorage.getItem('privacyhub-auth')
    if (stored) {
      try {
        const parsed = JSON.parse(stored)
        const token = parsed?.state?.accessToken
        if (token && token !== 'null' && token !== 'undefined') {
          config.headers.Authorization = `Bearer ${token}`
        }
      } catch (e) {
        console.error('Error parsing auth token:', e)
      }
    }
    return config
  },
  (error) => Promise.reject(error)
)

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      console.log('Auth error - clearing storage and redirecting to login')
      localStorage.removeItem('privacyhub-auth')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export default api
