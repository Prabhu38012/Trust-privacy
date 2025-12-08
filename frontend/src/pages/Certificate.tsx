import { useState } from 'react'
import api from '../lib/api'

export default function Certificate() {
  const [certificateHash, setCertificateHash] = useState('')
  const [verification, setVerification] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const handleVerify = async () => {
    if (!certificateHash.trim()) return
    setLoading(true)
    try {
      const response = await api.post('/certificate/verify', { certificateHash })
      setVerification(response.data)
    } catch (error) {
      console.error('Verification failed:', error)
      setVerification({ valid: false })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white">Certificate Verification</h1>
        <p className="text-gray-400 mt-1">Verify blockchain-secured certificates</p>
      </div>

      {/* Verify Certificate */}
      <div className="glass-card rounded-2xl p-8">
        <h2 className="text-xl font-semibold text-white mb-6">Verify Certificate</h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Certificate Hash
            </label>
            <input
              type="text"
              value={certificateHash}
              onChange={(e) => setCertificateHash(e.target.value)}
              placeholder="Enter certificate hash..."
              className="w-full px-4 py-3 bg-dark-700 border border-white/10 rounded-xl text-white focus:outline-none focus:border-primary-500 transition-colors"
            />
          </div>

          <button
            onClick={handleVerify}
            disabled={loading || !certificateHash.trim()}
            className="w-full py-3 px-6 bg-gradient-to-r from-primary-500 to-primary-600 hover:from-primary-600 hover:to-primary-700 text-white rounded-xl font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Verifying...' : 'Verify Certificate'}
          </button>
        </div>
      </div>

      {/* Verification Result */}
      {verification && (
        <div className={`glass-card rounded-2xl p-8 bg-gradient-to-br ${verification.valid ? 'from-emerald-500/20 to-emerald-500/5 border-emerald-500/30' : 'from-rose-500/20 to-rose-500/5 border-rose-500/30'} border`}>
          <div className="flex items-center gap-4 mb-6">
            {verification.valid ? (
              <div className="w-16 h-16 rounded-full bg-emerald-500/20 flex items-center justify-center">
                <svg className="w-8 h-8 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
            ) : (
              <div className="w-16 h-16 rounded-full bg-rose-500/20 flex items-center justify-center">
                <svg className="w-8 h-8 text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </div>
            )}
            <div>
              <h3 className={`text-2xl font-bold ${verification.valid ? 'text-emerald-400' : 'text-rose-400'}`}>
                {verification.valid ? 'Certificate Valid' : 'Certificate Invalid'}
              </h3>
              <p className="text-gray-400">
                {verification.valid ? 'Verified on blockchain' : 'Not found in blockchain'}
              </p>
            </div>
          </div>

          {verification.valid && (
            <div className="grid grid-cols-2 gap-4 mt-6">
              <div className="glass rounded-xl p-4">
                <p className="text-gray-400 text-sm">Block Number</p>
                <p className="text-white font-semibold text-lg">#{verification.block}</p>
              </div>
              <div className="glass rounded-xl p-4">
                <p className="text-gray-400 text-sm">Timestamp</p>
                <p className="text-white font-semibold text-lg">
                  {new Date(verification.timestamp).toLocaleDateString()}
                </p>
              </div>
              {verification.certificate?.verdict && (
                <div className="glass rounded-xl p-4 col-span-2">
                  <p className="text-gray-400 text-sm">Verdict</p>
                  <p className="text-white font-semibold text-lg">{verification.certificate.verdict}</p>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* How it Works */}
      <div className="glass-card rounded-2xl p-8">
        <h2 className="text-xl font-semibold text-white mb-4">How Certificate Verification Works</h2>
        <div className="space-y-4">
          <div className="flex gap-4">
            <div className="w-8 h-8 rounded-full bg-primary-500/20 flex items-center justify-center flex-shrink-0">
              <span className="text-primary-400 font-semibold">1</span>
            </div>
            <div>
              <h3 className="text-white font-medium">Blockchain Storage</h3>
              <p className="text-gray-400 text-sm">Every scan generates a certificate stored on our blockchain</p>
            </div>
          </div>
          <div className="flex gap-4">
            <div className="w-8 h-8 rounded-full bg-primary-500/20 flex items-center justify-center flex-shrink-0">
              <span className="text-primary-400 font-semibold">2</span>
            </div>
            <div>
              <h3 className="text-white font-medium">Cryptographic Hash</h3>
              <p className="text-gray-400 text-sm">Certificate data is hashed using SHA-256 encryption</p>
            </div>
          </div>
          <div className="flex gap-4">
            <div className="w-8 h-8 rounded-full bg-primary-500/20 flex items-center justify-center flex-shrink-0">
              <span className="text-primary-400 font-semibold">3</span>
            </div>
            <div>
              <h3 className="text-white font-medium">Immutable Verification</h3>
              <p className="text-gray-400 text-sm">Certificates cannot be altered or forged once recorded</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
