import { useEffect, useState, useCallback } from 'react'
import { useAuthStore } from '../store/authStore'
import { Link } from 'react-router-dom'
import axios from 'axios'

interface DashboardStats {
  totalScans: number
  recentScans: number
  totalCertificates: number
  onChainCertificates: number
  fraudAlerts: number
  securityScore: number
}

interface Activity {
  id: string
  filename: string
  verdict: string
  score: number
  onChain: boolean
  timestamp: string
}

interface ScanTrend {
  date: string
  count: number
}

export default function Dashboard() {
  const { user, token } = useAuthStore()
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [recentActivity, setRecentActivity] = useState<Activity[]>([])
  const [scanTrend, setScanTrend] = useState<ScanTrend[]>([])
  const [loading, setLoading] = useState(true)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  const fetchStats = useCallback(async () => {
    try {
      const response = await axios.get(
        `${import.meta.env.VITE_API_URL || 'http://localhost:3001'}/api/user/stats`,
        { headers: { Authorization: `Bearer ${token}` } }
      )
      setStats(response.data.stats)
      setRecentActivity(response.data.recentActivity || [])
      setScanTrend(response.data.scanTrend || [])
      setLastUpdated(new Date())
    } catch (error) {
      console.error('Failed to fetch stats:', error)
    } finally {
      setLoading(false)
    }
  }, [token])

  useEffect(() => {
    fetchStats()
    // Poll every 30 seconds for real-time updates
    const interval = setInterval(fetchStats, 30000)
    return () => clearInterval(interval)
  }, [fetchStats])

  const getVerdictColor = (verdict: string) => {
    switch (verdict) {
      case 'AUTHENTIC':
      case 'LIKELY_AUTHENTIC':
        return 'text-emerald-400'
      case 'SUSPICIOUS':
      case 'LIKELY_DEEPFAKE':
        return 'text-rose-400'
      default:
        return 'text-amber-400'
    }
  }

  const getVerdictBg = (verdict: string) => {
    switch (verdict) {
      case 'AUTHENTIC':
      case 'LIKELY_AUTHENTIC':
        return 'bg-emerald-500/10 border-emerald-500/20'
      case 'SUSPICIOUS':
      case 'LIKELY_DEEPFAKE':
        return 'bg-rose-500/10 border-rose-500/20'
      default:
        return 'bg-amber-500/10 border-amber-500/20'
    }
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Security Dashboard</h1>
          <p className="text-gray-400 mt-1">Welcome back, {user?.email}</p>
        </div>
        <div className="flex items-center gap-3">
          {lastUpdated && (
            <span className="text-xs text-gray-500">
              Updated {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <div className="flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <span className="text-xs text-emerald-400">Live</span>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Deepfake Scans"
          value={loading ? '—' : stats?.totalScans || 0}
          subtitle={loading ? '' : `+${stats?.recentScans || 0} this week`}
          icon={
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          }
          color="cyan"
          loading={loading}
        />
        <StatCard
          title="Certificates"
          value={loading ? '—' : stats?.totalCertificates || 0}
          subtitle={loading ? '' : `${stats?.onChainCertificates || 0} on blockchain`}
          icon={
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          }
          color="purple"
          loading={loading}
        />
        <StatCard
          title="Fraud Alerts"
          value={loading ? '—' : stats?.fraudAlerts || 0}
          subtitle={stats?.fraudAlerts === 0 ? 'No threats detected' : 'Requires attention'}
          icon={
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          }
          color="rose"
          loading={loading}
          alert={stats?.fraudAlerts ? stats.fraudAlerts > 0 : false}
        />
        <StatCard
          title="Security Score"
          value={loading ? '—' : `${stats?.securityScore || 100}%`}
          subtitle={getSecurityLabel(stats?.securityScore || 100)}
          icon={
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
          color="emerald"
          loading={loading}
        />
      </div>

      {/* Quick Actions + Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Quick Actions */}
        <div className="lg:col-span-2 space-y-6">
          <h2 className="text-xl font-semibold text-white">Quick Actions</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Link to="/dashboard/scan" className="group">
              <div className="glass-card rounded-2xl p-6 bg-gradient-to-br from-cyan-500/10 to-blue-500/5 hover:border-cyan-500/30 transition-all hover:-translate-y-1">
                <div className="flex items-center gap-4">
                  <div className="w-14 h-14 rounded-xl bg-cyan-500/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                    <svg className="w-7 h-7 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-white">Deepfake Scan</h3>
                    <p className="text-sm text-gray-400">Analyze media for AI manipulation</p>
                  </div>
                </div>
                <div className="mt-4 flex items-center text-cyan-400 text-sm font-medium">
                  Start Scan
                  <svg className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                  </svg>
                </div>
              </div>
            </Link>

            <Link to="/dashboard/document-analysis" className="group">
              <div className="glass-card rounded-2xl p-6 bg-gradient-to-br from-purple-500/10 to-pink-500/5 hover:border-purple-500/30 transition-all hover:-translate-y-1">
                <div className="flex items-center gap-4">
                  <div className="w-14 h-14 rounded-xl bg-purple-500/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                    <svg className="w-7 h-7 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-white">Document Analysis</h3>
                    <p className="text-sm text-gray-400">Verify document authenticity</p>
                  </div>
                </div>
                <div className="mt-4 flex items-center text-purple-400 text-sm font-medium">
                  Analyze Document
                  <svg className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                  </svg>
                </div>
              </div>
            </Link>

            <Link to="/dashboard/certificates" className="group">
              <div className="glass-card rounded-2xl p-6 bg-gradient-to-br from-emerald-500/10 to-teal-500/5 hover:border-emerald-500/30 transition-all hover:-translate-y-1">
                <div className="flex items-center gap-4">
                  <div className="w-14 h-14 rounded-xl bg-emerald-500/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                    <svg className="w-7 h-7 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-white">View Certificates</h3>
                    <p className="text-sm text-gray-400">Manage blockchain certificates</p>
                  </div>
                </div>
                <div className="mt-4 flex items-center text-emerald-400 text-sm font-medium">
                  View All
                  <svg className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                  </svg>
                </div>
              </div>
            </Link>

            <Link to="/dashboard/settings" className="group">
              <div className="glass-card rounded-2xl p-6 bg-gradient-to-br from-amber-500/10 to-orange-500/5 hover:border-amber-500/30 transition-all hover:-translate-y-1">
                <div className="flex items-center gap-4">
                  <div className="w-14 h-14 rounded-xl bg-amber-500/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                    <svg className="w-7 h-7 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-white">Settings</h3>
                    <p className="text-sm text-gray-400">Configure your preferences</p>
                  </div>
                </div>
                <div className="mt-4 flex items-center text-amber-400 text-sm font-medium">
                  Open Settings
                  <svg className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                  </svg>
                </div>
              </div>
            </Link>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="glass-card rounded-2xl p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-white">Recent Activity</h2>
            <Link to="/dashboard/certificates" className="text-sm text-primary-400 hover:text-primary-300">
              View all
            </Link>
          </div>

          {loading ? (
            <div className="space-y-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="animate-pulse flex items-center gap-3">
                  <div className="w-10 h-10 bg-white/5 rounded-lg" />
                  <div className="flex-1 space-y-2">
                    <div className="h-4 bg-white/5 rounded w-3/4" />
                    <div className="h-3 bg-white/5 rounded w-1/2" />
                  </div>
                </div>
              ))}
            </div>
          ) : recentActivity.length > 0 ? (
            <div className="space-y-4">
              {recentActivity.map((activity) => (
                <div
                  key={activity.id}
                  className={`flex items-center gap-3 p-3 rounded-xl border ${getVerdictBg(activity.verdict)} transition-all hover:scale-[1.02]`}
                >
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${activity.onChain ? 'bg-emerald-500/20' : 'bg-white/5'
                    }`}>
                    {activity.onChain ? (
                      <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                      </svg>
                    ) : (
                      <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-white truncate">{activity.filename}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className={`text-xs font-medium ${getVerdictColor(activity.verdict)}`}>
                        {activity.verdict.replace('_', ' ')}
                      </span>
                      <span className="text-xs text-gray-500">•</span>
                      <span className="text-xs text-gray-500">
                        {new Date(activity.timestamp).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                  <span className="text-sm font-mono text-gray-400">{activity.score}%</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-white/5 flex items-center justify-center">
                <svg className="w-8 h-8 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <p className="text-gray-400 text-sm">No recent activity</p>
              <p className="text-gray-500 text-xs mt-1">Start by scanning a file</p>
            </div>
          )}
        </div>
      </div>

      {/* Activity Chart */}
      {scanTrend.length > 0 && (
        <div className="glass-card rounded-2xl p-6">
          <h2 className="text-xl font-semibold text-white mb-6">Scan Activity (7 Days)</h2>
          <div className="flex items-end gap-2 h-32">
            {scanTrend.map((day, i) => {
              const maxCount = Math.max(...scanTrend.map(d => d.count), 1)
              const height = (day.count / maxCount) * 100
              return (
                <div key={i} className="flex-1 flex flex-col items-center gap-2">
                  <div
                    className="w-full bg-gradient-to-t from-cyan-500/30 to-cyan-500/10 rounded-t transition-all hover:from-cyan-500/50 hover:to-cyan-500/20"
                    style={{ height: `${Math.max(height, 4)}%` }}
                  />
                  <span className="text-xs text-gray-500">
                    {new Date(day.date).toLocaleDateString('en', { weekday: 'short' })}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Account Info */}
      <div className="glass-card rounded-2xl p-6">
        <h2 className="text-xl font-semibold text-white mb-6">Account Information</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <InfoCard label="Email" value={user?.email || 'N/A'} />
          <InfoCard label="User ID" value={user?.id || 'N/A'} mono />
          <InfoCard label="Member Since" value={user?.createdAt ? new Date(user.createdAt).toLocaleDateString() : 'N/A'} />
          <InfoCard label="Security Level" value="Standard" badge="Upgrade" />
        </div>
      </div>
    </div>
  )
}

function StatCard({
  title,
  value,
  subtitle,
  icon,
  color,
  loading,
  alert = false
}: {
  readonly title: string
  readonly value: string | number
  readonly subtitle: string
  readonly icon: React.ReactNode
  readonly color: 'cyan' | 'purple' | 'rose' | 'emerald'
  readonly loading: boolean
  readonly alert?: boolean
}) {
  const colors = {
    cyan: {
      bg: 'from-cyan-500/20 to-cyan-500/5',
      border: 'border-cyan-500/20 hover:border-cyan-500/40',
      icon: 'text-cyan-400 bg-cyan-500/20',
      text: 'text-cyan-400'
    },
    purple: {
      bg: 'from-purple-500/20 to-purple-500/5',
      border: 'border-purple-500/20 hover:border-purple-500/40',
      icon: 'text-purple-400 bg-purple-500/20',
      text: 'text-purple-400'
    },
    rose: {
      bg: 'from-rose-500/20 to-rose-500/5',
      border: 'border-rose-500/20 hover:border-rose-500/40',
      icon: 'text-rose-400 bg-rose-500/20',
      text: 'text-rose-400'
    },
    emerald: {
      bg: 'from-emerald-500/20 to-emerald-500/5',
      border: 'border-emerald-500/20 hover:border-emerald-500/40',
      icon: 'text-emerald-400 bg-emerald-500/20',
      text: 'text-emerald-400'
    }
  }

  const c = colors[color]

  return (
    <div className={`glass-card bg-gradient-to-br ${c.bg} rounded-2xl p-6 border ${c.border} transition-all hover:-translate-y-1`}>
      <div className="flex items-start justify-between">
        <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${c.icon}`}>
          {icon}
        </div>
        {alert && (
          <span className="relative flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-rose-500"></span>
          </span>
        )}
      </div>
      <div className="mt-4">
        <p className="text-gray-400 text-sm">{title}</p>
        {loading ? (
          <div className="h-9 w-20 bg-white/5 rounded animate-pulse mt-2" />
        ) : (
          <p className="text-3xl font-bold text-white mt-1">{value}</p>
        )}
        <p className={`text-sm mt-1 ${c.text}`}>{subtitle}</p>
      </div>
    </div>
  )
}

function InfoCard({ label, value, mono = false, badge }: {
  readonly label: string
  readonly value: string
  readonly mono?: boolean
  readonly badge?: string
}) {
  return (
    <div className="bg-white/5 rounded-xl p-4">
      <p className="text-gray-400 text-sm">{label}</p>
      <div className="flex items-center justify-between mt-1">
        <p className={`text-white ${mono ? 'font-mono text-sm truncate' : ''}`}>{value}</p>
        {badge && (
          <button className="text-xs px-2 py-1 rounded-full bg-primary-500/20 text-primary-400 hover:bg-primary-500/30 transition-colors">
            {badge}
          </button>
        )}
      </div>
    </div>
  )
}

function getSecurityLabel(score: number): string {
  if (score >= 90) return 'Excellent'
  if (score >= 70) return 'Good'
  if (score >= 50) return 'Fair'
  return 'Needs Attention'
}
