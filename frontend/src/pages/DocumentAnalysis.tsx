import { useState, useCallback } from 'react'
import api from '../lib/api'

// Day 6: Document Analysis interfaces
interface ELAResult {
    ela_image: string
    ela_raw: string
    brightness_std: number
    brightness_mean: number
    suspect_percentage: number
    quality_used: number
    amplification: number
}

interface EXIFResult {
    camera: string | null
    software: string | null
    date_time_original: string | null
    date_time_modified: string | null
    editing_detected: boolean
    editing_software: string[]
    tamper_indicators: string[]
}

interface PageResult {
    page_number: number
    image: string
    ela: ELAResult | null
    ela_interpretation: string
    exif: EXIFResult | null
}

interface DocumentResult {
    job_id: string
    filename: string
    file_type: string
    pages: PageResult[]
    summary: {
        verdict: string
        confidence: string
        tamper_score: number
        tamper_indicators: string[]
        pages_analyzed: number
    }
}

export default function DocumentAnalysis() {
    const [file, setFile] = useState<File | null>(null)
    const [isDragging, setIsDragging] = useState(false)
    const [isAnalyzing, setIsAnalyzing] = useState(false)
    const [progress, setProgress] = useState(0)
    const [result, setResult] = useState<DocumentResult | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [selectedPage, setSelectedPage] = useState(0)
    const [showELA, setShowELA] = useState(false)

    const handleDrag = useCallback((e: React.DragEvent) => { e.preventDefault(); e.stopPropagation() }, [])
    const handleDragIn = useCallback((e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true) }, [])
    const handleDragOut = useCallback((e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false) }, [])

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault(); e.stopPropagation(); setIsDragging(false)
        if (e.dataTransfer.files?.[0]) {
            setFile(e.dataTransfer.files[0]); setError(null); setResult(null)
        }
    }, [])

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files?.[0]) {
            setFile(e.target.files[0]); setError(null); setResult(null)
        }
    }

    const analyzeDocument = async () => {
        if (!file) return
        setIsAnalyzing(true); setProgress(0); setError(null)

        const formData = new FormData()
        formData.append('file', file)

        try {
            const progressInterval = setInterval(() => setProgress(prev => Math.min(prev + 8, 85)), 400)
            const response = await api.post('/scan/document', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                timeout: 120000
            })
            clearInterval(progressInterval)
            setProgress(100)
            setResult(response.data)
            setSelectedPage(0)
        } catch (err: unknown) {
            const error = err as { response?: { data?: { message?: string } } }
            setError(error.response?.data?.message || 'Document analysis failed')
        } finally {
            setIsAnalyzing(false)
        }
    }

    const resetAnalysis = () => {
        setFile(null); setResult(null); setError(null); setProgress(0); setShowELA(false)
    }

    const getVerdictColor = (verdict: string) => {
        const colors: Record<string, string> = {
            'LIKELY_AUTHENTIC': 'text-emerald-400',
            'POSSIBLY_TAMPERED': 'text-amber-400',
            'LIKELY_TAMPERED': 'text-rose-400',
            'SAMPLE_DOCUMENT': 'text-purple-400',
            'LIKELY_FRAUDULENT': 'text-rose-500',
            'SUSPICIOUS': 'text-orange-400'
        }
        return colors[verdict] || 'text-gray-400'
    }

    const getVerdictBg = (verdict: string) => {
        const bgs: Record<string, string> = {
            'LIKELY_AUTHENTIC': 'from-emerald-500/20 to-emerald-500/5 border-emerald-500/30',
            'POSSIBLY_TAMPERED': 'from-amber-500/20 to-amber-500/5 border-amber-500/30',
            'LIKELY_TAMPERED': 'from-rose-500/20 to-rose-500/5 border-rose-500/30',
            'SAMPLE_DOCUMENT': 'from-purple-500/20 to-purple-500/5 border-purple-500/30',
            'LIKELY_FRAUDULENT': 'from-rose-600/20 to-rose-600/5 border-rose-600/30',
            'SUSPICIOUS': 'from-orange-500/20 to-orange-500/5 border-orange-500/30'
        }
        return bgs[verdict] || 'from-gray-500/20 to-gray-500/5 border-gray-500/30'
    }

    const currentPage = result?.pages[selectedPage]

    return (
        <div className="space-y-8">
            <div>
                <h1 className="text-3xl font-bold text-white">Document Analysis</h1>
                <p className="text-gray-400 mt-1">ELA tampering detection & metadata analysis</p>
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
                        onClick={() => document.getElementById('docFileInput')?.click()}
                        onKeyDown={(e) => { if (e.key === 'Enter') document.getElementById('docFileInput')?.click() }}
                        className={`upload-zone border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${isDragging ? 'dragging border-cyan-500' : 'border-white/10'}`}
                    >
                        <input id="docFileInput" type="file" accept="image/*,application/pdf" onChange={handleFileSelect} className="hidden" />
                        <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-blue-500/20 flex items-center justify-center">
                            <svg className="w-8 h-8 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                        </div>
                        {file ? (
                            <div><p className="text-white font-medium">{file.name}</p><p className="text-gray-500 text-sm mt-1">{(file.size / (1024 * 1024)).toFixed(2)} MB</p></div>
                        ) : (
                            <div><p className="text-white font-medium">Drop your document here</p><p className="text-gray-500 text-sm mt-1">or click to browse</p><p className="text-gray-600 text-xs mt-4">Supports JPG, PNG, PDF (max 50MB)</p></div>
                        )}
                    </div>

                    {error && <div className="mt-4 p-4 bg-rose-500/10 border border-rose-500/20 rounded-xl text-rose-400 text-sm">{error}</div>}

                    {isAnalyzing && (
                        <div className="mt-6">
                            <div className="flex justify-between text-sm mb-2"><span className="text-gray-400">Analyzing document...</span><span className="text-cyan-400">{progress}%</span></div>
                            <div className="h-2 bg-dark-600 rounded-full overflow-hidden"><div className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-300" style={{ width: `${progress}%` }} /></div>
                        </div>
                    )}

                    <div className="mt-6 flex gap-4">
                        <button onClick={analyzeDocument} disabled={!file || isAnalyzing} className="flex-1 py-3 px-6 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white rounded-xl font-medium transition-all disabled:opacity-50">{isAnalyzing ? 'Analyzing...' : 'Analyze Document'}</button>
                        {file && !isAnalyzing && <button onClick={resetAnalysis} className="py-3 px-6 bg-white/5 hover:bg-white/10 text-white rounded-xl font-medium transition-all border border-white/10">Clear</button>}
                    </div>
                </div>
            )}

            {/* Results Section */}
            {result && (
                <div className="space-y-6">
                    {/* Verdict Card */}
                    <div className={`glass-card rounded-2xl p-8 bg-gradient-to-br ${getVerdictBg(result.summary.verdict)} border`}>
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-gray-400 text-sm uppercase tracking-wider">Document Verdict</p>
                                <h2 className={`text-4xl font-bold mt-2 ${getVerdictColor(result.summary.verdict)}`}>{result.summary.verdict.replace(/_/g, ' ')}</h2>
                                <p className="text-gray-400 mt-2">Confidence: {result.summary.confidence}</p>
                            </div>
                            <div className="text-right">
                                <div className="text-6xl font-bold text-white">{result.summary.tamper_score.toFixed(0)}%</div>
                                <p className="text-gray-500 text-sm">Tamper Score</p>
                            </div>
                        </div>
                    </div>

                    {/* Tamper Indicators */}
                    {result.summary.tamper_indicators.length > 0 && (
                        <div className="glass-card rounded-2xl p-6 border border-amber-500/30">
                            <h3 className="text-lg font-semibold text-amber-400 mb-3">⚠️ Tamper Indicators</h3>
                            <ul className="space-y-2">
                                {result.summary.tamper_indicators.map((indicator, i) => (
                                    <li key={i} className="text-gray-300 text-sm flex items-start gap-2">
                                        <span className="text-amber-400 mt-0.5">•</span>{indicator}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* Page Selector (for multi-page PDFs) */}
                    {result.pages.length > 1 && (
                        <div className="flex gap-2 overflow-x-auto pb-2">
                            {result.pages.map((page, i) => (
                                <button key={i} onClick={() => setSelectedPage(i)} className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${selectedPage === i ? 'bg-cyan-500 text-white' : 'bg-white/5 text-gray-400 hover:bg-white/10'}`}>
                                    Page {page.page_number}
                                </button>
                            ))}
                        </div>
                    )}

                    {/* Image Comparison */}
                    {currentPage && (
                        <div className="glass-card rounded-2xl p-6">
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-xl font-semibold text-white">{showELA ? 'Error Level Analysis' : 'Original Image'}</h3>
                                <button onClick={() => setShowELA(!showELA)} className="px-4 py-2 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-400 rounded-lg text-sm font-medium transition-all">
                                    {showELA ? 'Show Original' : 'Show ELA'}
                                </button>
                            </div>
                            <div className="aspect-video bg-dark-800 rounded-xl overflow-hidden flex items-center justify-center">
                                <img
                                    src={`data:image/png;base64,${showELA && currentPage.ela ? currentPage.ela.ela_image : currentPage.image}`}
                                    alt={showELA ? 'ELA Analysis' : 'Original'}
                                    className="max-w-full max-h-full object-contain"
                                />
                            </div>
                            {currentPage.ela && (
                                <div className="mt-4 p-4 bg-dark-800/50 rounded-xl">
                                    <p className="text-gray-300 text-sm">{currentPage.ela_interpretation}</p>
                                </div>
                            )}
                        </div>
                    )}

                    {/* EXIF Metadata */}
                    {currentPage?.exif && (
                        <div className="glass-card rounded-2xl p-6">
                            <h3 className="text-xl font-semibold text-white mb-4">EXIF Metadata</h3>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                <div className="p-3 bg-dark-800/50 rounded-xl">
                                    <p className="text-gray-500 text-xs">Camera</p>
                                    <p className="text-white text-sm mt-1">{currentPage.exif.camera || 'Unknown'}</p>
                                </div>
                                <div className="p-3 bg-dark-800/50 rounded-xl">
                                    <p className="text-gray-500 text-xs">Software</p>
                                    <p className="text-white text-sm mt-1">{currentPage.exif.software || 'Unknown'}</p>
                                </div>
                                <div className="p-3 bg-dark-800/50 rounded-xl">
                                    <p className="text-gray-500 text-xs">Date Taken</p>
                                    <p className="text-white text-sm mt-1">{currentPage.exif.date_time_original || 'Unknown'}</p>
                                </div>
                                <div className={`p-3 rounded-xl ${currentPage.exif.editing_detected ? 'bg-rose-500/20 border border-rose-500/30' : 'bg-dark-800/50'}`}>
                                    <p className="text-gray-500 text-xs">Editing Detected</p>
                                    <p className={`text-sm mt-1 font-medium ${currentPage.exif.editing_detected ? 'text-rose-400' : 'text-emerald-400'}`}>
                                        {currentPage.exif.editing_detected ? 'Yes' : 'No'}
                                    </p>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Actions */}
                    <div className="flex gap-4">
                        <button onClick={resetAnalysis} className="flex-1 py-3 px-6 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white rounded-xl font-medium transition-all">
                            Analyze Another Document
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}
