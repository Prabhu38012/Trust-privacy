import { useState, useEffect } from 'react'
import api from '../lib/api'

export default function Admin() {
  const [stats, setStats] = useState<any>(null)
  const [blockchain, setBlockchain] = useState<any>(null)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      const [blockchainRes] = await Promise.all([
        api.get('/certificate/blockchain/status')
      ])
      setBlockchain(blockchainRes.data)
    } catch (error) {
      console.error('Failed to load admin data:', error)
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white">Admin Dashboard</h1>
        <p className="text-gray-400 mt-1">System overview and management</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Scans"
          value="1,234"
          change="+12.5%"
          icon="📊"
        />
        <StatCard
          title="Deepfakes Detected"
          value="234"
          change="+5.2%"
          icon="🔴"
        />
        <StatCard
          title="Certificates Issued"
          value="892"
          change="+18.3%"
          icon="🔒"
        />
        <StatCard
          title="Active Users"
          value="456"
          change="+8.1%"
          icon="👥"
        />
      </div>

      {/* Blockchain Status */}
      {blockchain && (
        <div className="glass-card rounded-2xl p-8">
          <h2 className="text-xl font-semibold text-white mb-6">Blockchain Status</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div>
              <p className="text-gray-400 text-sm mb-1">Chain Length</p>
              <p className="text-2xl font-bold text-white">{blockchain.chainLength}</p>
            </div>
            <div>
              <p className="text-gray-400 text-sm mb-1">Pending Certificates</p>
              <p className="text-2xl font-bold text-yellow-400">{blockchain.pendingCertificates}</p>
            </div>
            <div>
              <p className="text-gray-400 text-sm mb-1">Chain Valid</p>
              <p className={`text-2xl font-bold ${blockchain.isValid ? 'text-emerald-400' : 'text-rose-400'}`}>
                {blockchain.isValid ? '✓ Yes' : '✗ No'}
              </p>
            </div>
            <div>
              <p className="text-gray-400 text-sm mb-1">Last Block</p>
              <p className="text-2xl font-bold text-white">#{blockchain.lastBlock?.index}</p>
            </div>
          </div>
        </div>
      )}

      {/* Recent Activity */}
      <div className="glass-card rounded-2xl p-8">
        <h2 className="text-xl font-semibold text-white mb-6">Recent Scans</h2>
        <div className="space-y-4">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="flex items-center justify-between p-4 glass rounded-xl hover:-translate-y-0.5 transition-transform">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 rounded-full bg-primary-500/20 flex items-center justify-center">
                  <span className="text-primary-400">👤</span>
                </div>
                <div>
                  <p className="text-white font-medium">User{i}@example.com</p>
                  <p className="text-gray-400 text-sm">Scanned video_{i}.mp4</p>
                </div>
              </div>
              <div className="text-right">
                <p className={`font-semibold ${i % 2 === 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {i % 2 === 0 ? 'AUTHENTIC' : 'DEEPFAKE'}
                </p>
                <p className="text-gray-400 text-sm">{i} hours ago</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function StatCard({ title, value, change, icon }: any) {
  const isPositive = change.startsWith('+')
  return (
    <div className="glass-card rounded-2xl p-6">
      <div className="flex items-center justify-between mb-4">
        <span className="text-3xl">{icon}</span>
        <span className={`text-sm font-medium ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
          {change}
        </span>
      </div>
      <p className="text-gray-400 text-sm mb-1">{title}</p>
      <p className="text-3xl font-bold text-white">{value}</p>
    </div>
  )
}
