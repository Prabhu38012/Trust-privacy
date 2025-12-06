import { useState, useCallback } from 'react'
import api from '../lib/api'

interface FrameResult {
  frame_number: number
  image: string
  deepfake_probability: number
  confidence: number
  details?: {
    neural_network?: number
    compression_artifacts?: number
    color_consistency?: number
    frequency_analysis?: number
  }
}

interface ScanResult {
  jobId: string
  status: string
  result: {
    verdict: string
    verdict_confidence: string
    deepfake_score: number
    frames_analyzed: number
    frames: FrameResult[]
    analysis_summary?: {
      average_score: number
      max_score: number
      consistency: number
      neural_network_avg: number
      artifact_detection_avg: number
    }
    metadata?: {
      mode: string
      device: string
    }
  }
  processingTime: number
}

// Helper function for score color classes
function getScoreColorClass(score: number): string {
  if (score < 0.3) return 'text-emerald-400'
  if (score < 0.6) return 'text-amber-400'
  return 'text-rose-400'
}

function getScoreBgClass(score: number): string {
  if (score < 0.3) return 'bg-emerald-500'
  if (score < 0.6) return 'bg-amber-500'
  return 'bg-rose-500'
}

export default function DeepfakeScan() {
  const [file, setFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [scanResult, setScanResult] = useState<ScanResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [certificateData, setCertificateData] = useState<any>(null)

  const handleDrag = useCallback((e: React.DragEvent) => { e.preventDefault(); e.stopPropagation() }, [])
  const handleDragIn = useCallback((e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true) }, [])
  const handleDragOut = useCallback((e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false) }, [])
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation(); setIsDragging(false)
    if (e.dataTransfer.files?.[0]) { setFile(e.dataTransfer.files[0]); setError(null); setScanResult(null) }
  }, [])
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) { setFile(e.target.files[0]); setError(null); setScanResult(null) }
  }

  const handleUpload = async () => {
    if (!file) return
    setIsUploading(true); setUploadProgress(0); setError(null)
    const formData = new FormData()
    formData.append('file', file)
    try {
      const progressInterval = setInterval(() => setUploadProgress(prev => Math.min(prev + 5, 90)), 300)
      const response = await api.post('/scan/upload', formData, { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 300000 })
      clearInterval(progressInterval)
      setUploadProgress(100)
      
      // Handle response - it might come directly or nested in result
      const data = response.data
      if (data.result) {
        setScanResult(data)
      } else {
        // If response is flat, wrap it
        setScanResult({
          jobId: data.jobId || data.job_id || '',
          status: data.status || 'completed',
          result: {
            verdict: data.verdict || 'UNKNOWN',
            verdict_confidence: data.verdict_confidence || 'LOW',
            deepfake_score: data.deepfake_score || 0,
            frames_analyzed: data.frames_analyzed || 0,
            frames: data.frames || [],
            analysis_summary: data.analysis_summary,
            metadata: data.metadata
          },
          processingTime: data.processingTime || 0
        })
      }
    } catch (err: any) {
      console.error('Upload error:', err)
      setError(err.response?.data?.message || err.response?.data?.error || 'Failed to scan file. Make sure ML service is running.')
    } finally { setIsUploading(false) }
  }

  const resetScan = () => { setFile(null); setScanResult(null); setError(null); setUploadProgress(0) }

  const getVerdictColor = (verdict?: string) => {
    if (!verdict) return 'text-gray-400'
    const colors: Record<string, string> = {
      'AUTHENTIC': 'text-emerald-400',
      'LIKELY_AUTHENTIC': 'text-green-400',
      'UNCERTAIN': 'text-yellow-400',
      'SUSPICIOUS': 'text-orange-400',
      'LIKELY_DEEPFAKE': 'text-rose-400'
    }
    return colors[verdict] || 'text-gray-400'
  }

  const getVerdictBg = (verdict?: string) => {
    if (!verdict) return 'from-gray-500/20 to-gray-500/5 border-gray-500/30'
    const bgs: Record<string, string> = {
      'AUTHENTIC': 'from-emerald-500/20 to-emerald-500/5 border-emerald-500/30',
      'LIKELY_AUTHENTIC': 'from-green-500/20 to-green-500/5 border-green-500/30',
      'UNCERTAIN': 'from-yellow-500/20 to-yellow-500/5 border-yellow-500/30',
      'SUSPICIOUS': 'from-orange-500/20 to-orange-500/5 border-orange-500/30',
      'LIKELY_DEEPFAKE': 'from-rose-500/20 to-rose-500/5 border-rose-500/30'
    }
    return bgs[verdict] || 'from-gray-500/20 to-gray-500/5 border-gray-500/30'
  }

  const formatVerdict = (verdict?: string) => {
    if (!verdict) return 'UNKNOWN'
    return verdict.split('_').join(' ')
  }

  const generateCertificate = async () => {
    if (!scanResult) return
    try {
      const response = await api.post('/certificate/generate', {
        scanId: scanResult.jobId,
        verdict: scanResult.result.verdict,
        score: scanResult.result.deepfake_score,
        filename: file?.name || 'unknown'
      })
      
      const certData = response.data.certificate
      
      setCertificateData(certData)
      alert('Certificate generated!')
    } catch (error) {
      console.error('Certificate generation failed:', error)
      alert('Failed to generate certificate')
    }
  }

  const downloadReport = async () => {
    if (!scanResult || !certificateData) return
    try {
      const response = await api.post('/report/generate', {
        scanResult,
        certificate: certificateData
      }, {
        responseType: 'blob'
      })
      
      const blob = new Blob([response.data], { type: 'application/pdf' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `trustlock-report-${certificateData.id}.pdf`
      a.click()
      URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Report download failed:', error)
      alert('Failed to download report')
    }
  }

  const handleUploadZoneClick = () => {
    document.getElementById('fileInput')?.click()
  }

  const handleUploadZoneKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      document.getElementById('fileInput')?.click()
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white">Deepfake Detection</h1>
        <p className="text-gray-400 mt-1">AI-powered analysis to detect manipulated media</p>
      </div>

      {/* Upload Section */}
      {!scanResult && (
        <div className="glass-card rounded-2xl p-8">
          <div
            role="button"
            tabIndex={0}
            onDragEnter={handleDragIn}
            onDragLeave={handleDragOut}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={handleUploadZoneClick}
            onKeyDown={handleUploadZoneKeyDown}
            className={`upload-zone border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${isDragging ? 'dragging border-primary-500' : 'border-white/10'}`}
          >
            <input id="fileInput" type="file" accept="video/*,image/*" onChange={handleFileSelect} className="hidden" />
            <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-primary-500/20 to-cyan-500/20 flex items-center justify-center">
              <svg className="w-8 h-8 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            {file ? (
              <div><p className="text-white font-medium">{file.name}</p><p className="text-gray-500 text-sm mt-1">{(file.size / (1024 * 1024)).toFixed(2)} MB</p></div>
            ) : (
              <div><p className="text-white font-medium">Drop your file here</p><p className="text-gray-500 text-sm mt-1">or click to browse</p><p className="text-gray-600 text-xs mt-4">Supports MP4, AVI, MOV, JPG, PNG (max 100MB)</p></div>
            )}
          </div>
          {error && <div className="mt-4 p-4 bg-rose-500/10 border border-rose-500/20 rounded-xl text-rose-400 text-sm">{error}</div>}
          {isUploading && (
            <div className="mt-6">
              <div className="flex justify-between text-sm mb-2"><span className="text-gray-400">Analyzing with AI...</span><span className="text-primary-400">{uploadProgress}%</span></div>
              <div className="h-2 bg-dark-600 rounded-full overflow-hidden"><div className="h-full bg-gradient-to-r from-primary-500 to-cyan-500 transition-all duration-300" style={{ width: `${uploadProgress}%` }} /></div>
            </div>
          )}
          <div className="mt-6 flex gap-4">
            <button onClick={handleUpload} disabled={!file || isUploading} className="flex-1 py-3 px-6 bg-gradient-to-r from-primary-500 to-primary-600 hover:from-primary-600 hover:to-primary-700 text-white rounded-xl font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed">{isUploading ? 'Analyzing...' : 'Start Analysis'}</button>
            {file && !isUploading && <button onClick={resetScan} className="py-3 px-6 bg-white/5 hover:bg-white/10 text-white rounded-xl font-medium transition-all border border-white/10">Clear</button>}
          </div>
        </div>
      )}

      {/* Results Section */}
      {scanResult?.result && (
        <div className="space-y-6">
          {/* Mode Badge */}
          {scanResult.result.metadata?.mode && (
            <div className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${scanResult.result.metadata.mode === 'PRODUCTION' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'}`}>
              {scanResult.result.metadata.mode === 'PRODUCTION' ? '🔒 Production Mode' : '⚠️ Demo Mode'}
            </div>
          )}

          {/* Verdict Card */}
          <div className={`glass-card rounded-2xl p-8 bg-gradient-to-br ${getVerdictBg(scanResult.result.verdict)} border`}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm uppercase tracking-wider">Verdict</p>
                <h2 className={`text-4xl font-bold mt-2 ${getVerdictColor(scanResult.result.verdict)}`}>
                  {formatVerdict(scanResult.result.verdict)}
                </h2>
                <p className="text-gray-400 mt-2">Confidence: {scanResult.result.verdict_confidence || 'N/A'}</p>
              </div>
              <div className="text-right">
                <div className="text-6xl font-bold text-white">{(scanResult.result.deepfake_score || 0).toFixed(1)}%</div>
                <p className="text-gray-500 text-sm">Manipulation Score</p>
              </div>
            </div>
          </div>

          {/* Analysis Details */}
          {scanResult.result.analysis_summary && (
            <div className="glass-card rounded-2xl p-6">
              <h3 className="text-xl font-semibold text-white mb-4">Analysis Breakdown</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard label="Neural Network" value={scanResult.result.analysis_summary.neural_network_avg || 0} />
                <MetricCard label="Artifact Detection" value={scanResult.result.analysis_summary.artifact_detection_avg || 0} />
                <MetricCard label="Max Frame Score" value={scanResult.result.analysis_summary.max_score || 0} />
                <MetricCard label="Consistency" value={scanResult.result.analysis_summary.consistency || 0} isGood />
              </div>
            </div>
          )}

          {/* Frames Grid */}
          {scanResult.result.frames && scanResult.result.frames.length > 0 && (
            <div className="glass-card rounded-2xl p-6">
              <h3 className="text-xl font-semibold text-white mb-6">Analyzed Frames ({scanResult.result.frames_analyzed || scanResult.result.frames.length})</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {scanResult.result.frames.map((frame) => {
                  const probability = frame.deepfake_probability || 0
                  const scoreColor = getScoreColorClass(probability)
                  const scoreBg = getScoreBgClass(probability)
                  
                  return (
                    <div key={frame.frame_number} className="glass rounded-xl overflow-hidden hover:-translate-y-1 transition-all">
                      <div className="aspect-video relative">
                        {frame.image && <img src={`data:image/png;base64,${frame.image}`} alt={`Frame ${frame.frame_number}`} className="w-full h-full object-cover" />}
                        <div className="absolute top-2 left-2 px-2 py-1 bg-black/60 rounded text-xs text-white">#{frame.frame_number}</div>
                      </div>
                      <div className="p-3">
                        <div className="flex justify-between items-center">
                          <span className="text-gray-400 text-xs">Score</span>
                          <span className={`text-sm font-medium ${scoreColor}`}>
                            {(probability * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="mt-2 h-1.5 bg-dark-600 rounded-full overflow-hidden">
                          <div 
                            className={`h-full rounded-full ${scoreBg}`} 
                            style={{ width: `${probability * 100}%` }} 
                          />
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-4">
            <button onClick={resetScan} className="flex-1 py-3 px-6 bg-gradient-to-r from-primary-500 to-primary-600 hover:from-primary-600 hover:to-primary-700 text-white rounded-xl font-medium transition-all">Scan Another File</button>
            <button onClick={generateCertificate} disabled={certificateData} className="py-3 px-6 bg-white/5 hover:bg-white/10 text-white rounded-xl font-medium transition-all border border-white/10 disabled:opacity-50">
              {certificateData ? '✓ Generated' : 'Generate Certificate'}
            </button>
            {certificateData && (
              <button onClick={downloadReport} className="py-3 px-6 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 rounded-xl font-medium transition-all border border-emerald-500/30">
                Download PDF
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

interface MetricCardProps {
  readonly label: string
  readonly value: number
  readonly isGood?: boolean
}

function MetricCard({ label, value, isGood = false }: MetricCardProps) {
  const getColor = (): string => {
    if (isGood) {
      if (value > 70) return 'text-emerald-400'
      if (value > 40) return 'text-amber-400'
      return 'text-rose-400'
    }
    if (value < 30) return 'text-emerald-400'
    if (value < 60) return 'text-amber-400'
    return 'text-rose-400'
  }
  
  return (
    <div className="glass rounded-xl p-4">
      <p className="text-gray-400 text-xs mb-1">{label}</p>
      <p className={`text-2xl font-bold ${getColor()}`}>{(value || 0).toFixed(1)}%</p>
    </div>
  )
}
