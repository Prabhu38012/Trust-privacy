import { Link } from 'react-router-dom'

export default function Landing() {
  return (
    <div className="min-h-screen bg-dark-900 grid-bg">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 bg-dark-900/80 backdrop-blur-md border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-accent-cyan rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <span className="text-xl font-semibold text-white">TrustLock</span>
            </div>
            
            <div className="hidden md:flex items-center space-x-8">
              <a href="#deepfake" className="text-gray-400 hover:text-white transition-colors text-sm">DEEPFAKE DETECTION</a>
              <a href="#features" className="text-gray-400 hover:text-white transition-colors text-sm">FEATURES</a>
              <a href="#pricing" className="text-gray-400 hover:text-white transition-colors text-sm">PRICING</a>
              <a href="#resources" className="text-gray-400 hover:text-white transition-colors text-sm">RESOURCES</a>
            </div>

            <Link 
              to="/signup" 
              className="bg-primary-500 hover:bg-primary-600 text-white px-5 py-2 rounded-full text-sm font-medium transition-all hover:shadow-lg hover:shadow-primary-500/25"
            >
              Get Started
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6 relative overflow-hidden">
        {/* Background glow effects */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-primary-500/10 rounded-full blur-[120px]" />
        <div className="absolute top-1/3 left-1/4 w-[300px] h-[300px] bg-accent-cyan/10 rounded-full blur-[100px]" />
        
        <div className="max-w-7xl mx-auto relative">
          <div className="flex flex-col lg:flex-row items-center gap-12">
            {/* Left Feature Card */}
            <div className="hidden lg:block flex-1">
              <FeatureCard 
                icon={<FingerprintIcon />}
                title="Password Leak Defense"
                code="L-0318-F728-L1"
              />
            </div>

            {/* Center - Main Display */}
            <div className="flex-1 flex flex-col items-center">
              <div className="relative">
                {/* Main security card */}
                <div className="w-80 h-48 bg-gradient-to-br from-dark-600 to-dark-800 rounded-2xl border border-white/10 p-6 glow-box animate-float">
                  <div className="flex items-center justify-center h-full">
                    <div className="w-16 h-16 bg-gradient-to-br from-primary-500/20 to-accent-cyan/20 rounded-2xl flex items-center justify-center border border-white/10">
                      <div className="w-10 h-10 bg-gradient-to-br from-rose-400 to-rose-600 rounded-lg" />
                    </div>
                  </div>
                  {/* Binary/code overlay effect */}
                  <div className="absolute inset-0 rounded-2xl overflow-hidden opacity-20">
                    <div className="text-[8px] text-primary-400 font-mono leading-tight p-2 break-all">
                      01001010110101001010110101010010101101010100101011010101001010110
                    </div>
                  </div>
                </div>
                
                {/* Glow effect underneath */}
                <div className="absolute -bottom-4 left-1/2 -translate-x-1/2 w-3/4 h-8 bg-primary-500/30 blur-xl rounded-full" />
              </div>
            </div>

            {/* Right Feature Card */}
            <div className="hidden lg:block flex-1">
              <FeatureCard 
                icon={<SecurityDotsIcon />}
                title="Advanced Login Security"
                align="right"
              />
            </div>
          </div>

          {/* Hero Text */}
          <div className="text-center mt-20">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6">
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-400 via-accent-cyan to-primary-400">
                Trusted Data Protection
              </span>
            </h1>
            <p className="text-xl md:text-2xl text-gray-300 mb-4">
              Maximum Data Safety, Minimum Effort
            </p>
            <p className="text-gray-500 max-w-2xl mx-auto mb-10">
              Take control of your data stack—encrypt sensitive info, minimize risks, meet 
              compliance, prevent vendor lock-in, and scale securely.
            </p>
            
            <Link 
              to="/signup"
              className="inline-flex items-center px-8 py-3 bg-dark-600 hover:bg-dark-500 text-white rounded-full border border-white/10 transition-all hover:border-white/20 hover:shadow-lg"
            >
              Try It Free
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="products" className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              AI-Powered Security Features
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Protect your digital identity with cutting-edge deepfake detection and blockchain-verified authenticity
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <SecurityFeatureCard 
              icon="🔍"
              title="Deepfake Detection"
              description="AI-powered analysis to detect manipulated images, videos, and audio in real-time"
              gradient="from-rose-500/20 to-orange-500/20"
            />
            <SecurityFeatureCard 
              icon="⛓️"
              title="Blockchain Certificates"
              description="Immutable authenticity certificates stored on blockchain for tamper-proof verification"
              gradient="from-primary-500/20 to-accent-cyan/20"
            />
            <SecurityFeatureCard 
              icon="🛡️"
              title="Fraud Prevention"
              description="Real-time fraud detection and prevention using advanced machine learning models"
              gradient="from-emerald-500/20 to-teal-500/20"
            />
            <SecurityFeatureCard 
              icon="🔐"
              title="E2E Encryption"
              description="Military-grade encryption for all your sensitive data and communications"
              gradient="from-violet-500/20 to-purple-500/20"
            />
            <SecurityFeatureCard 
              icon="📊"
              title="Privacy Analytics"
              description="Differential privacy enabled analytics that protect individual user data"
              gradient="from-blue-500/20 to-cyan-500/20"
            />
            <SecurityFeatureCard 
              icon="📋"
              title="Audit Trails"
              description="Complete transparency with immutable audit logs for all security events"
              gradient="from-amber-500/20 to-yellow-500/20"
            />
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <div className="bg-gradient-to-r from-dark-700 to-dark-600 rounded-3xl p-12 border border-white/10 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-64 h-64 bg-primary-500/10 rounded-full blur-[100px]" />
            <div className="relative">
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
                Ready to Secure Your Data?
              </h2>
              <p className="text-gray-400 mb-8">
                Join thousands of organizations protecting their digital assets with TrustLock
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Link 
                  to="/signup"
                  className="px-8 py-3 bg-primary-500 hover:bg-primary-600 text-white rounded-full font-medium transition-all hover:shadow-lg hover:shadow-primary-500/25"
                >
                  Start Free Trial
                </Link>
                <Link 
                  to="/login"
                  className="px-8 py-3 bg-dark-500 hover:bg-dark-400 text-white rounded-full font-medium border border-white/10 transition-all"
                >
                  Sign In
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/5 py-12 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-accent-cyan rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <span className="text-white font-semibold">TrustLock</span>
            </div>
            <p className="text-gray-500 text-sm">
              © 2024 TrustLock. AI-Powered Security for the Digital Age.
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}

// Feature Card Component
function FeatureCard({ icon, title, code, align = 'left' }: { icon: React.ReactNode; title: string; code?: string; align?: 'left' | 'right' }) {
  return (
    <div className={`flex flex-col ${align === 'right' ? 'items-end' : 'items-start'}`}>
      <div className="bg-dark-700/50 backdrop-blur border border-white/10 rounded-2xl p-6 max-w-xs">
        <div className="flex items-center gap-4 mb-3">
          {code && <span className="text-gray-500 text-sm font-mono">{code}</span>}
          <div className="w-12 h-12 bg-dark-600 rounded-xl flex items-center justify-center border border-white/10">
            {icon}
          </div>
        </div>
        <p className="text-gray-400 text-sm">{title}</p>
      </div>
    </div>
  )
}

// Security Feature Card Component
function SecurityFeatureCard({ icon, title, description, gradient }: { icon: string; title: string; description: string; gradient: string }) {
  return (
    <div className={`bg-gradient-to-br ${gradient} rounded-2xl p-6 border border-white/10 hover:border-white/20 transition-all group hover:-translate-y-1`}>
      <div className="text-4xl mb-4">{icon}</div>
      <h3 className="text-xl font-semibold text-white mb-2">{title}</h3>
      <p className="text-gray-400 text-sm">{description}</p>
    </div>
  )
}

// Icon Components
function FingerprintIcon() {
  return (
    <svg className="w-6 h-6 text-accent-cyan" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 11c0 3.517-1.009 6.799-2.753 9.571m-3.44-2.04l.054-.09A13.916 13.916 0 008 11a4 4 0 118 0c0 1.017-.07 2.019-.203 3m-2.118 6.844A21.88 21.88 0 0015.171 17m3.839 1.132c.645-2.266.99-4.659.99-7.132A8 8 0 008 4.07M3 15.364c.64-1.319 1-2.8 1-4.364 0-1.457.39-2.823 1.07-4" />
    </svg>
  )
}

function SecurityDotsIcon() {
  return (
    <div className="grid grid-cols-4 gap-1">
      {[...Array(8)].map((_, i) => (
        <div key={i} className={`w-2 h-2 rounded-full ${i < 3 ? 'bg-gray-600' : 'bg-primary-500'}`} />
      ))}
    </div>
  )
}
