import { useState, useEffect, useCallback } from 'react'
import api from '../lib/api'

interface Certificate {
  id: string
  scanId: string
  filename: string
  fileHash: string
  verdict: string
  score: number
  timestamp: string
  blockchain: {
    onChain: boolean
    transactionHash?: string
    explorerUrl?: string
    network?: string
  }
}

interface VerificationResult {
  verified: boolean
  certificate: {
    id: string
    filename: string
    verdict: string
    score: number
    timestamp: string
    storedHash?: string
    fileHash?: string
  }
  verification?: {
    hashMatch: boolean
    storedHash: string
    providedHash: string
    onChain: boolean
    onChainVerified: boolean
    transactionHash?: string
    explorerUrl?: string
  }
  blockchain?: {
    onChain: boolean
    verified: boolean
    transactionHash?: string
    explorerUrl?: string
    network?: string
  }
}

export default function Certificates() {
  const [certificates, setCertificates] = useState<Certificate[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedCert, setSelectedCert] = useState<Certificate | null>(null)
  const [verifyFile, setVerifyFile] = useState<File | null>(null)
  const [verifying, setVerifying] = useState(false)
  const [verificationResult, setVerificationResult] = useState<VerificationResult | null>(null)
  const [quickVerifyId, setQuickVerifyId] = useState('')

  useEffect(() => {
    fetchCertificates()
  }, [])

  const fetchCertificates = async () => {
    try {
      const response = await api.get('/certificate/list')
      setCertificates(response.data.certificates || [])
    } catch (error) {
      console.error('Failed to fetch certificates:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleVerifyWithFile = async () => {
    if (!selectedCert || !verifyFile) return
    setVerifying(true)
    setVerificationResult(null)

    try {
      const formData = new FormData()
      formData.append('file', verifyFile)
      formData.append('certificateId', selectedCert.id)

      const response = await api.post('/certificate/verify', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      
      setVerificationResult(response.data)
    } catch (error) {
      console.error('Verification failed:', error)
      setVerificationResult({
        verified: false,
        certificate: { id: selectedCert.id, filename: selectedCert.filename, verdict: selectedCert.verdict, score: selectedCert.score, timestamp: selectedCert.timestamp }
      })
    } finally {
      setVerifying(false)
    }
  }

  const handleClientSideVerify = useCallback(async () => {
    if (!selectedCert || !verifyFile) return
    setVerifying(true)
    setVerificationResult(null)

    try {
      // Compute hash client-side using SubtleCrypto
      const arrayBuffer = await verifyFile.arrayBuffer()
      const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer)
      const hashArray = Array.from(new Uint8Array(hashBuffer))
      const clientHash = '0x' + hashArray.map(b => b.toString(16).padStart(2, '0')).join('')

      // Send hash to server for comparison
      const response = await api.post('/certificate/verify', {
        certificateId: selectedCert.id,
        clientHash
      })
      
      setVerificationResult(response.data)
    } catch (error) {
      console.error('Client-side verification failed:', error)
    } finally {
      setVerifying(false)
    }
  }, [selectedCert, verifyFile])

  const handleQuickVerify = async (certId: string) => {
    try {
      const response = await api.get(`/certificate/verify/${certId}`)
      setVerificationResult(response.data)
      setSelectedCert(certificates.find(c => c.id === certId) || null)
    } catch (error) {
      console.error('Quick verify failed:', error)
    }
  }

  const handlePublicVerify = async () => {
    if (!quickVerifyId.trim()) return
    setVerifying(true)
    setVerificationResult(null)

    try {
      const response = await api.get(`/certificate/verify/${quickVerifyId.trim()}`)
      setVerificationResult(response.data)
    } catch (error: any) {
      setVerificationResult({
        verified: false,
        certificate: { id: quickVerifyId, filename: 'Unknown', verdict: 'UNKNOWN', score: 0, timestamp: '' }
      })
    } finally {
      setVerifying(false)
    }
  }

  const getVerdictColor = (verdict: string) => {
    const colors: Record<string, string> = {
      'AUTHENTIC': 'text-emerald-400',
      'LIKELY_AUTHENTIC': 'text-green-400',
      'UNCERTAIN': 'text-yellow-400',
      'SUSPICIOUS': 'text-orange-400',
      'LIKELY_DEEPFAKE': 'text-rose-400'
    }
    return colors[verdict] || 'text-gray-400'
  }

  const getVerdictBg = (verdict: string) => {
    const colors: Record<string, string> = {
      'AUTHENTIC': 'bg-emerald-500/20',
      'LIKELY_AUTHENTIC': 'bg-green-500/20',
      'UNCERTAIN': 'bg-yellow-500/20',
      'SUSPICIOUS': 'bg-orange-500/20',
      'LIKELY_DEEPFAKE': 'bg-rose-500/20'
    }
    return colors[verdict] || 'bg-gray-500/20'
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white">Certificates</h1>
        <p className="text-gray-400 mt-1">View and verify your blockchain certificates</p>
      </div>

      {/* Public Verify Section */}
      <div className="glass-card rounded-2xl p-6">
        <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
          <span>🔍</span> Public Certificate Verification
        </h2>
        <p className="text-gray-400 text-sm mb-4">
          Enter a certificate ID to verify its authenticity
        </p>
        <div className="flex gap-4">
          <input
            type="text"
            placeholder="Enter Certificate ID..."
            value={quickVerifyId}
            onChange={(e) => setQuickVerifyId(e.target.value)}
            className="flex-1 bg-dark-600 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-primary-500"
          />
          <button
            onClick={handlePublicVerify}
            disabled={!quickVerifyId.trim() || verifying}
            className="px-6 py-3 bg-primary-500 hover:bg-primary-600 text-white rounded-xl font-medium transition-all disabled:opacity-50"
          >
            {verifying ? 'Verifying...' : 'Verify'}
          </button>
        </div>
      </div>

      {/* Verification Result */}
      {verificationResult && (
        <div className={`glass-card rounded-2xl p-6 border ${verificationResult.verified ? 'border-emerald-500/30' : 'border-rose-500/30'}`}>
          <div className="flex items-center gap-4 mb-6">
            <div className={`w-16 h-16 rounded-2xl flex items-center justify-center ${verificationResult.verified ? 'bg-emerald-500/20' : 'bg-rose-500/20'}`}>
              {verificationResult.verified ? (
                <svg className="w-8 h-8 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              ) : (
                <svg className="w-8 h-8 text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              )}
            </div>
            <div>
              <h3 className={`text-2xl font-bold ${verificationResult.verified ? 'text-emerald-400' : 'text-rose-400'}`}>
                {verificationResult.verified ? 'Certificate Verified ✓' : 'Verification Failed ✗'}
              </h3>
              <p className="text-gray-400">{verificationResult.certificate.filename}</p>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Certificate Details */}
            <div className="space-y-3">
              <h4 className="text-sm font-medium text-gray-400 uppercase tracking-wider">Certificate Details</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-500">ID</span>
                  <span className="text-white font-mono text-sm">{verificationResult.certificate.id.slice(0, 20)}...</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Verdict</span>
                  <span className={getVerdictColor(verificationResult.certificate.verdict)}>
                    {verificationResult.certificate.verdict.replace('_', ' ')}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Score</span>
                  <span className="text-white">{verificationResult.certificate.score}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Issued</span>
                  <span className="text-white">{new Date(verificationResult.certificate.timestamp).toLocaleString()}</span>
                </div>
              </div>
            </div>

            {/* Verification Details */}
            <div className="space-y-3">
              <h4 className="text-sm font-medium text-gray-400 uppercase tracking-wider">Verification</h4>
              <div className="space-y-2">
                {verificationResult.verification && (
                  <>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Hash Match</span>
                      <span className={verificationResult.verification.hashMatch ? 'text-emerald-400' : 'text-rose-400'}>
                        {verificationResult.verification.hashMatch ? '✓ Match' : '✗ Mismatch'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">On-Chain</span>
                      <span className={verificationResult.verification.onChain ? 'text-emerald-400' : 'text-gray-500'}>
                        {verificationResult.verification.onChain ? '✓ Yes' : 'No'}
                      </span>
                    </div>
                    {verificationResult.verification.onChain && (
                      <div className="flex justify-between">
                        <span className="text-gray-500">Chain Verified</span>
                        <span className={verificationResult.verification.onChainVerified ? 'text-emerald-400' : 'text-rose-400'}>
                          {verificationResult.verification.onChainVerified ? '✓ Verified' : '✗ Failed'}
                        </span>
                      </div>
                    )}
                  </>
                )}
                {verificationResult.blockchain && (
                  <>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Network</span>
                      <span className="text-white">{verificationResult.blockchain.network || 'N/A'}</span>
                    </div>
                    {verificationResult.blockchain.explorerUrl && (
                      <a
                        href={verificationResult.blockchain.explorerUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary-400 hover:text-primary-300 text-sm flex items-center gap-1"
                      >
                        View on Explorer
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                        </svg>
                      </a>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>

          <button
            onClick={() => setVerificationResult(null)}
            className="mt-6 text-gray-400 hover:text-white text-sm"
          >
            ← Back to certificates
          </button>
        </div>
      )}

      {/* File Verification Section */}
      {selectedCert && !verificationResult && (
        <div className="glass-card rounded-2xl p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-xl font-semibold text-white">Verify Certificate</h2>
              <p className="text-gray-400 text-sm">Upload the original file to verify against the stored hash</p>
            </div>
            <button onClick={() => { setSelectedCert(null); setVerifyFile(null) }} className="text-gray-400 hover:text-white">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="bg-dark-600/50 rounded-xl p-4 mb-6">
            <div className="flex items-center gap-4">
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${getVerdictBg(selectedCert.verdict)}`}>
                {selectedCert.blockchain?.onChain ? '⛓️' : '📄'}
              </div>
              <div>
                <p className="text-white font-medium">{selectedCert.filename}</p>
                <p className="text-gray-500 text-sm">
                  {selectedCert.verdict.replace('_', ' ')} • {selectedCert.score}%
                </p>
              </div>
            </div>
          </div>

          <div className="border-2 border-dashed border-white/10 rounded-xl p-8 text-center mb-6">
            <input
              type="file"
              onChange={(e) => setVerifyFile(e.target.files?.[0] || null)}
              className="hidden"
              id="verifyFileInput"
            />
            <label htmlFor="verifyFileInput" className="cursor-pointer">
              {verifyFile ? (
                <div>
                  <p className="text-white font-medium">{verifyFile.name}</p>
                  <p className="text-gray-500 text-sm">{(verifyFile.size / (1024 * 1024)).toFixed(2)} MB</p>
                </div>
              ) : (
                <div>
                  <p className="text-gray-400">Drop file here or click to browse</p>
                  <p className="text-gray-600 text-sm mt-1">Select the original file to verify</p>
                </div>
              )}
            </label>
          </div>

          <div className="flex gap-4">
            <button
              onClick={handleClientSideVerify}
              disabled={!verifyFile || verifying}
              className="flex-1 py-3 bg-primary-500 hover:bg-primary-600 text-white rounded-xl font-medium transition-all disabled:opacity-50"
            >
              {verifying ? 'Verifying...' : 'Verify (Client-side Hash)'}
            </button>
            <button
              onClick={handleVerifyWithFile}
              disabled={!verifyFile || verifying}
              className="flex-1 py-3 bg-white/5 hover:bg-white/10 text-white rounded-xl font-medium border border-white/10 transition-all disabled:opacity-50"
            >
              {verifying ? 'Verifying...' : 'Verify (Server-side Hash)'}
            </button>
          </div>
        </div>
      )}

      {/* Certificates List */}
      {!verificationResult && !selectedCert && (
        <div className="glass-card rounded-2xl p-6">
          <h2 className="text-xl font-semibold text-white mb-6">Your Certificates</h2>
          
          {loading ? (
            <div className="text-center py-12">
              <div className="animate-spin w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full mx-auto mb-4" />
              <p className="text-gray-400">Loading certificates...</p>
            </div>
          ) : certificates.length === 0 ? (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-dark-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <p className="text-gray-400">No certificates yet</p>
              <p className="text-gray-600 text-sm mt-1">Scan a file and generate a certificate to get started</p>
            </div>
          ) : (
            <div className="space-y-4">
              {certificates.map((cert) => (
                <div key={cert.id} className="bg-dark-600/50 rounded-xl p-4 hover:bg-dark-600 transition-all">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${getVerdictBg(cert.verdict)}`}>
                        {cert.blockchain?.onChain ? '⛓️' : '📄'}
                      </div>
                      <div>
                        <p className="text-white font-medium">{cert.filename}</p>
                        <div className="flex items-center gap-3 mt-1">
                          <span className={`text-sm ${getVerdictColor(cert.verdict)}`}>
                            {cert.verdict.replace('_', ' ')}
                          </span>
                          <span className="text-gray-600">•</span>
                          <span className="text-gray-500 text-sm">{cert.score}%</span>
                          <span className="text-gray-600">•</span>
                          <span className="text-gray-500 text-sm">
                            {new Date(cert.timestamp).toLocaleDateString()}
                          </span>
                          {cert.blockchain?.onChain && (
                            <>
                              <span className="text-gray-600">•</span>
                              <span className="text-emerald-400 text-sm">On-Chain ✓</span>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handleQuickVerify(cert.id)}
                        className="px-4 py-2 bg-white/5 hover:bg-white/10 text-white rounded-lg text-sm transition-all"
                      >
                        Quick Verify
                      </button>
                      <button
                        onClick={() => setSelectedCert(cert)}
                        className="px-4 py-2 bg-primary-500/20 hover:bg-primary-500/30 text-primary-400 rounded-lg text-sm transition-all"
                      >
                        Verify with File
                      </button>
                      {cert.blockchain?.explorerUrl && (
                        <a
                          href={cert.blockchain.explorerUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="p-2 bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white rounded-lg transition-all"
                        >
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                          </svg>
                        </a>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
