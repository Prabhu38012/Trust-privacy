import { useAuthStore } from '../store/authStore'
import { Link } from 'react-router-dom'

export default function Dashboard() {
  const { user } = useAuthStore()

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white">Security Dashboard</h1>
        <p className="text-gray-400 mt-1">Welcome back, {user?.email}</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <StatCard title="Deepfake Scans" value="0" change="+0%" color="cyan" />
        <StatCard title="Certificates Issued" value="0" change="+0%" color="purple" />
        <StatCard title="Fraud Alerts" value="0" change="0%" color="rose" />
        <StatCard title="Security Score" value="100%" change="Excellent" color="emerald" />
      </div>

      {/* Feature Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Deepfake Detection Card */}
        <Link to="/dashboard/scan">
          <FeatureCard
            icon={
              <svg className="w-6 h-6 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            }
            title="Deepfake Detection"
            description="Upload images or videos to scan for AI-generated manipulation"
            buttonText="Start Scan"
            gradient="from-cyan-500/10 to-blue-500/5"
          />
        </Link>

        {/* Blockchain Certificates */}
        <FeatureCard
          icon={
            <svg className="w-6 h-6 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          }
          title="Authenticity Certificates"
          description="Generate blockchain-verified certificates for your media"
          buttonText="Create Certificate"
          gradient="from-purple-500/10 to-pink-500/5"
        />

        {/* Fraud Prevention */}
        <FeatureCard
          icon={
            <svg className="w-6 h-6 text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          }
          title="Fraud Detection"
          description="Real-time monitoring and alerts for suspicious activities"
          buttonText="View Alerts"
          gradient="from-rose-500/10 to-orange-500/5"
        />

        {/* Encrypted Storage */}
        <FeatureCard
          icon={
            <svg className="w-6 h-6 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
          }
          title="Encrypted Notes"
          description="Store sensitive information with end-to-end encryption"
          buttonText="Coming Soon"
          gradient="from-emerald-500/10 to-teal-500/5"
          disabled
        />

        {/* Data Export */}
        <FeatureCard
          icon={
            <svg className="w-6 h-6 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
          }
          title="Data Requests"
          description="Export or delete your data with full transparency"
          buttonText="Coming Soon"
          gradient="from-blue-500/10 to-indigo-500/5"
          disabled
        />

        {/* Audit Logs */}
        <FeatureCard
          icon={
            <svg className="w-6 h-6 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
          }
          title="Audit Logs"
          description="Complete history of all security events and actions"
          buttonText="Coming Soon"
          gradient="from-amber-500/10 to-yellow-500/5"
          disabled
        />
      </div>

      {/* Account Info */}
      <div className="glass-card rounded-2xl p-6">
        <h2 className="text-xl font-semibold text-white mb-6">Account Information</h2>
        <div className="space-y-4">
          <InfoRow label="Email" value={user?.email || 'N/A'} />
          <InfoRow label="User ID" value={user?.id || 'N/A'} mono />
          <InfoRow label="Member Since" value={user?.createdAt ? new Date(user.createdAt).toLocaleDateString() : 'N/A'} />
          <InfoRow label="Security Level" value="Standard" badge="Upgrade" />
        </div>
      </div>
    </div>
  )
}

function StatCard({ title, value, change, color }: { title: string; value: string; change: string; color: string }) {
  const colors: Record<string, string> = {
    cyan: 'border-cyan-500/20',
    purple: 'border-purple-500/20',
    rose: 'border-rose-500/20',
    emerald: 'border-emerald-500/20',
  }

  return (
    <div className={`glass-card rounded-2xl p-6 border ${colors[color]}`}>
      <p className="text-gray-400 text-sm">{title}</p>
      <p className="text-3xl font-bold text-white mt-2">{value}</p>
      <p className="text-gray-500 text-sm mt-1">{change}</p>
    </div>
  )
}

function FeatureCard({ icon, title, description, buttonText, gradient, disabled = false }: {
  icon: React.ReactNode
  title: string
  description: string
  buttonText: string
  gradient: string
  disabled?: boolean
}) {
  return (
    <div className={`glass-card bg-gradient-to-br ${gradient} rounded-2xl p-6 hover:border-white/20 transition-all group ${disabled ? '' : 'cursor-pointer hover:-translate-y-1'}`}>
      <div className="w-12 h-12 glass rounded-xl flex items-center justify-center mb-4">
        {icon}
      </div>
      <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
      <p className="text-gray-400 text-sm mb-4">{description}</p>
      <span
        className={`inline-block px-4 py-2 rounded-lg text-sm font-medium transition-all ${
          disabled
            ? 'bg-white/5 text-gray-500 cursor-not-allowed'
            : 'bg-white/10 text-white group-hover:bg-white/20'
        }`}
      >
        {buttonText}
      </span>
    </div>
  )
}

function InfoRow({ label, value, mono = false, badge }: { label: string; value: string; mono?: boolean; badge?: string }) {
  return (
    <div className="flex justify-between items-center py-3 border-b border-white/5 last:border-0">
      <span className="text-gray-400">{label}</span>
      <div className="flex items-center gap-3">
        <span className={`text-white ${mono ? 'font-mono text-sm' : ''}`}>{value}</span>
        {badge && (
          <button className="text-xs text-primary-400 hover:text-primary-300 transition-colors">
            {badge}
          </button>
        )}
      </div>
    </div>
  )
}
