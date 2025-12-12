import { useState, useCallback } from 'react'
import api from '../lib/api'

interface TamperingIndicator {
    type: string
    severity: 'high' | 'medium' | 'low'
    message: string
}

interface DocumentResult {
    jobId: string
    status: string
    filename: string
    file_type: 'image' | 'pdf'
    page_count: number
    result: {
        verdict: string
        tampering_score: number
        explanation: string
        original_image?: string
        ela: {
            image?: string
            score: number
        }
        exif: {
            has_data: boolean
            metadata: Record<string, string>
            warnings: string[]
            suspicious_indicators: string[]
        }
        tampering_indicators: TamperingIndicator[]
    }
    metadata: {
        pdf_support: boolean
        exif_support: boolean
    }
}

// Helper function for verdict styling
function getVerdictColor(verdict?: string): string {
    if (!verdict) return 'text-gray-400'
    const colors: Record<string, string> = {
        'LIKELY_AUTHENTIC': 'text-emerald-400',
        'POSSIBLY_MODIFIED': 'text-yellow-400',
        'SUSPICIOUS': 'text-orange-400',
        'LIKELY_TAMPERED': 'text-rose-400'
    }
    return colors[verdict] || 'text-gray-400'
}

function getVerdictBg(verdict?: string): string {
    if (!verdict) return 'from-gray-500/20 to-gray-500/5 border-gray-500/30'
    const bgs: Record<string, string> = {
        'LIKELY_AUTHENTIC': 'from-emerald-500/20 to-emerald-500/5 border-emerald-500/30',
        'POSSIBLY_MODIFIED': 'from-yellow-500/20 to-yellow-500/5 border-yellow-500/30',
        'SUSPICIOUS': 'from-orange-500/20 to-orange-500/5 border-orange-500/30',
        'LIKELY_TAMPERED': 'from-rose-500/20 to-rose-500/5 border-rose-500/30'
    }
    return bgs[verdict] || 'from-gray-500/20 to-gray-500/5 border-gray-500/30'
}

function formatVerdict(verdict?: string): string {
    if (!verdict) return 'UNKNOWN'
    return verdict.split('_').join(' ')
}

function getSeverityColor(severity: string): string {
    switch (severity) {
        case 'high': return 'bg-rose-500/20 text-rose-400 border-rose-500/30'
        case 'medium': return 'bg-orange-500/20 text-orange-400 border-orange-500/30'
        case 'low': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
        default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
    }
}

export default function DocumentAnalysis() {
    const [file, setFile] = useState<File | null>(null)
    const [isDragging, setIsDragging] = useState(false)
    const [isAnalyzing, setIsAnalyzing] = useState(false)
    const [progress, setProgress] = useState(0)
    const [result, setResult] = useState<DocumentResult | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [showMetadata, setShowMetadata] = useState(false)
    const [activeView, setActiveView] = useState<'original' | 'ela'>('original')

    const handleDrag = useCallback((e: React.DragEvent) => { e.preventDefault(); e.stopPropagation() }, [])
    const handleDragIn = useCallback((e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true) }, [])
    const handleDragOut = useCallback((e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false) }, [])

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragging(false)
        if (e.dataTransfer.files?.[0]) {
            setFile(e.dataTransfer.files[0])
            setError(null)
            setResult(null)
        }
    }, [])

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files?.[0]) {
            setFile(e.target.files[0])
            setError(null)
            setResult(null)
        }
    }

    const handleAnalyze = async () => {
        if (!file) return

        setIsAnalyzing(true)
        setProgress(0)
        setError(null)

        const formData = new FormData()
        formData.append('file', file)

        try {
            const progressInterval = setInterval(() => setProgress(prev => Math.min(prev + 8, 90)), 400)

            const response = await api.post('/document/analyze', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                timeout: 120000 // 2 minute timeout
            })

            clearInterval(progressInterval)
            setProgress(100)
            setResult(response.data)
        } catch (err: unknown) {
            console.error('Analysis error:', err)
            const errorObj = err as { response?: { data?: { message?: string; error?: string }; status?: number } }

            if (errorObj.response?.status === 413) {
                setError('File too large. Maximum size is 50MB.')
            } else if (errorObj.response?.status === 415) {
                setError(errorObj.response?.data?.message || 'Unsupported file type.')
            } else if (errorObj.response?.status === 503) {
                setError('ML service not available. Please ensure the ML service is running.')
            } else {
                setError(errorObj.response?.data?.message || 'Failed to analyze document.')
            }
        } finally {
            setIsAnalyzing(false)
        }
    }

    const resetAnalysis = () => {
        setFile(null)
        setResult(null)
        setError(null)
        setProgress(0)
        setActiveView('original')
        setShowMetadata(false)
    }

    const handleUploadZoneClick = () => {
        document.getElementById('docFileInput')?.click()
    }

    const handleUploadZoneKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            document.getElementById('docFileInput')?.click()
        }
    }

    return (
        <div className="space-y-8">
            <div>
                <h1 className="text-3xl font-bold text-white">Document Analysis</h1>
                <p className="text-gray-400 mt-1">Detect tampering in documents and images using Error Level Analysis</p>
            </div>

            {/* Upload Section */}
            {!result && (
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
                        className={`upload-zone border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${isDragging ? 'dragging border-cyan-500' : 'border-white/10'}`}
                    >
                        <input
                            id="docFileInput"
                            type="file"
                            accept="image/*,.pdf"
                            onChange={handleFileSelect}
                            className="hidden"
                        />
                        <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-blue-500/20 flex items-center justify-center">
                            <svg className="w-8 h-8 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                        </div>
                        {file ? (
                            <div>
                                <p className="text-white font-medium">{file.name}</p>
                                <p className="text-gray-500 text-sm mt-1">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                            </div>
                        ) : (
                            <div>
                                <p className="text-white font-medium">Drop your document here</p>
                                <p className="text-gray-500 text-sm mt-1">or click to browse</p>
                                <p className="text-gray-600 text-xs mt-4">Supports JPG, PNG, PDF (max 50MB)</p>
                            </div>
                        )}
                    </div>

                    {error && (
                        <div className="mt-4 p-4 bg-rose-500/10 border border-rose-500/20 rounded-xl text-rose-400 text-sm">
                            {error}
                        </div>
                    )}

                    {isAnalyzing && (
                        <div className="mt-6">
                            <div className="flex justify-between text-sm mb-2">
                                <span className="text-gray-400">Analyzing document...</span>
                                <span className="text-cyan-400">{progress}%</span>
                            </div>
                            <div className="h-2 bg-dark-600 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-300"
                                    style={{ width: `${progress}%` }}
                                />
                            </div>
                        </div>
                    )}

                    <div className="mt-6 flex gap-4">
                        <button
                            onClick={handleAnalyze}
                            disabled={!file || isAnalyzing}
                            className="flex-1 py-3 px-6 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white rounded-xl font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {isAnalyzing ? 'Analyzing...' : 'Analyze Document'}
                        </button>
                        {file && !isAnalyzing && (
                            <button
                                onClick={resetAnalysis}
                                className="py-3 px-6 bg-white/5 hover:bg-white/10 text-white rounded-xl font-medium transition-all border border-white/10"
                            >
                                Clear
                            </button>
                        )}
                    </div>
                </div>
            )}

            {/* Results Section */}
            {result?.result && (
                <div className="space-y-6">
                    {/* File Info Badge */}
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-cyan-500/20 text-cyan-400">
                        {result.file_type === 'pdf' ? '📄 PDF Document' : '🖼️ Image'}
                        {result.page_count > 1 && ` (${result.page_count} pages)`}
                    </div>

                    {/* Verdict Card */}
                    <div className={`glass-card rounded-2xl p-8 bg-gradient-to-br ${getVerdictBg(result.result.verdict)} border`}>
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-gray-400 text-sm uppercase tracking-wider">Document Verdict</p>
                                <h2 className={`text-4xl font-bold mt-2 ${getVerdictColor(result.result.verdict)}`}>
                                    {formatVerdict(result.result.verdict)}
                                </h2>
                            </div>
                            <div className="text-right">
                                <div className="text-6xl font-bold text-white">{result.result.tampering_score.toFixed(1)}%</div>
                                <p className="text-gray-500 text-sm">Tampering Score</p>
                            </div>
                        </div>

                        {/* Explanation */}
                        <div className="mt-6 p-4 bg-black/20 rounded-xl">
                            <p className="text-gray-300 text-sm leading-relaxed">{result.result.explanation}</p>
                        </div>
                    </div>

                    {/* ELA Comparison */}
                    <div className="glass-card rounded-2xl p-6">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-xl font-semibold text-white">Error Level Analysis</h3>
                            <div className="flex bg-dark-600 rounded-lg p-1">
                                <button
                                    onClick={() => setActiveView('original')}
                                    className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${activeView === 'original' ? 'bg-cyan-500 text-white' : 'text-gray-400 hover:text-white'}`}
                                >
                                    Original
                                </button>
                                <button
                                    onClick={() => setActiveView('ela')}
                                    className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${activeView === 'ela' ? 'bg-cyan-500 text-white' : 'text-gray-400 hover:text-white'}`}
                                >
                                    ELA View
                                </button>
                            </div>
                        </div>

                        <div className="grid md:grid-cols-2 gap-4">
                            {/* Original Image */}
                            <div className={`relative rounded-xl overflow-hidden bg-dark-600 ${activeView === 'original' ? 'ring-2 ring-cyan-500' : ''}`}>
                                <div className="absolute top-2 left-2 px-2 py-1 bg-black/60 rounded text-xs text-white z-10">
                                    Original
                                </div>
                                {result.result.original_image && (
                                    <img
                                        src={`data:image/png;base64,${result.result.original_image}`}
                                        alt="Original document"
                                        className="w-full h-auto"
                                    />
                                )}
                            </div>

                            {/* ELA Image */}
                            <div className={`relative rounded-xl overflow-hidden bg-dark-600 ${activeView === 'ela' ? 'ring-2 ring-cyan-500' : ''}`}>
                                <div className="absolute top-2 left-2 px-2 py-1 bg-black/60 rounded text-xs text-white z-10">
                                    ELA ({result.result.ela.score.toFixed(1)}%)
                                </div>
                                {result.result.ela.image ? (
                                    <img
                                        src={`data:image/png;base64,${result.result.ela.image}`}
                                        alt="ELA analysis"
                                        className="w-full h-auto"
                                    />
                                ) : (
                                    <div className="aspect-video flex items-center justify-center text-gray-500">
                                        ELA not available
                                    </div>
                                )}
                            </div>
                        </div>

                        <p className="mt-4 text-gray-500 text-sm">
                            💡 <strong>How to read ELA:</strong> Bright areas indicate regions with different compression levels.
                            Edited or pasted elements often appear brighter than the surrounding areas.
                        </p>
                    </div>

                    {/* Tampering Indicators */}
                    {result.result.tampering_indicators.length > 0 && (
                        <div className="glass-card rounded-2xl p-6">
                            <h3 className="text-xl font-semibold text-white mb-4">Tampering Indicators</h3>
                            <div className="space-y-3">
                                {result.result.tampering_indicators.map((indicator, index) => (
                                    <div
                                        key={index}
                                        className={`flex items-start gap-3 p-3 rounded-lg border ${getSeverityColor(indicator.severity)}`}
                                    >
                                        <span className="text-lg">
                                            {indicator.severity === 'high' ? '🔴' : indicator.severity === 'medium' ? '🟡' : '🟢'}
                                        </span>
                                        <div>
                                            <span className="text-xs font-medium uppercase opacity-70">{indicator.type}</span>
                                            <p className="text-sm mt-0.5">{indicator.message}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* EXIF Metadata */}
                    <div className="glass-card rounded-2xl p-6">
                        <button
                            onClick={() => setShowMetadata(!showMetadata)}
                            className="w-full flex items-center justify-between text-left"
                        >
                            <div>
                                <h3 className="text-xl font-semibold text-white">EXIF Metadata</h3>
                                <p className="text-gray-500 text-sm mt-1">
                                    {result.result.exif.has_data ? 'Metadata found' : 'No metadata available'}
                                </p>
                            </div>
                            <svg
                                className={`w-6 h-6 text-gray-400 transition-transform ${showMetadata ? 'rotate-180' : ''}`}
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                            </svg>
                        </button>

                        {showMetadata && (
                            <div className="mt-4 space-y-4">
                                {/* Suspicious Indicators */}
                                {result.result.exif.suspicious_indicators.length > 0 && (
                                    <div className="p-4 bg-rose-500/10 border border-rose-500/20 rounded-xl">
                                        <h4 className="text-rose-400 font-medium mb-2">⚠️ Suspicious Indicators</h4>
                                        <ul className="space-y-1">
                                            {result.result.exif.suspicious_indicators.map((indicator, i) => (
                                                <li key={i} className="text-rose-300 text-sm">{indicator}</li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                                {/* Warnings */}
                                {result.result.exif.warnings.length > 0 && (
                                    <div className="p-4 bg-yellow-500/10 border border-yellow-500/20 rounded-xl">
                                        <h4 className="text-yellow-400 font-medium mb-2">⚡ Warnings</h4>
                                        <ul className="space-y-1">
                                            {result.result.exif.warnings.map((warning, i) => (
                                                <li key={i} className="text-yellow-300 text-sm">{warning}</li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                                {/* Metadata Table */}
                                {Object.keys(result.result.exif.metadata).length > 0 && (
                                    <div className="overflow-x-auto">
                                        <table className="w-full text-sm">
                                            <thead>
                                                <tr className="border-b border-white/10">
                                                    <th className="text-left py-2 px-3 text-gray-400 font-medium">Property</th>
                                                    <th className="text-left py-2 px-3 text-gray-400 font-medium">Value</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {Object.entries(result.result.exif.metadata)
                                                    .filter(([key]) => !key.startsWith('_'))
                                                    .slice(0, 20)
                                                    .map(([key, value]) => (
                                                        <tr key={key} className="border-b border-white/5">
                                                            <td className="py-2 px-3 text-gray-300">{key}</td>
                                                            <td className="py-2 px-3 text-gray-500 font-mono text-xs truncate max-w-xs">
                                                                {value}
                                                            </td>
                                                        </tr>
                                                    ))}
                                            </tbody>
                                        </table>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Actions */}
                    <div className="flex gap-4">
                        <button
                            onClick={resetAnalysis}
                            className="flex-1 py-3 px-6 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white rounded-xl font-medium transition-all"
                        >
                            Analyze Another Document
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}
