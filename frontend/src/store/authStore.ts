import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import api from '../lib/api'

interface User {
  id: string
  email: string
  name?: string
  role?: string
  createdAt?: string // Add this
}

interface AuthState {
  user: User | null
  accessToken: string | null
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null
  login: (email: string, password: string) => Promise<void>
  signup: (email: string, password: string) => Promise<void>
  logout: () => void
  clearError: () => void
  fetchUser: () => Promise<void>
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      accessToken: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      login: async (email: string, password: string) => {
        set({ isLoading: true, error: null })
        try {
          const response = await api.post('/auth/login', { email, password })
          const { accessToken, user } = response.data
          set({
            accessToken,
            user,
            isAuthenticated: true,
            isLoading: false,
          })
        } catch (err: any) {
          set({
            error: err.response?.data?.message || 'Login failed',
            isLoading: false,
          })
          throw err
        }
      },

      signup: async (email: string, password: string) => {
        set({ isLoading: true, error: null })
        try {
          const response = await api.post('/auth/signup', { email, password })
          const { accessToken, user } = response.data
          set({
            accessToken,
            user,
            isAuthenticated: true,
            isLoading: false,
          })
        } catch (err: any) {
          set({
            error: err.response?.data?.message || 'Signup failed',
            isLoading: false,
          })
          throw err
        }
      },

      logout: () => {
        set({
          user: null,
          accessToken: null,
          isAuthenticated: false,
        })
      },

      clearError: () => set({ error: null }),

      fetchUser: async () => {
        const token = get().accessToken
        if (!token) return
        try {
          const response = await api.get('/user/me')
          set({ user: response.data.user })
        } catch {
          get().logout()
        }
      },
    }),
    {
      name: 'privacyhub-auth',
      partialize: (state) => ({
        accessToken: state.accessToken,
        user: state.user,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
)
