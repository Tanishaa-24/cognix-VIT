export default function Header() {
  return (
    <header style={{ background: '#0a1020', borderBottom: '1px solid #1e2d4d' }} className="sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg,#3b82f6,#8b5cf6)' }}>
            <span className="text-white text-sm font-black">C</span>
          </div>
          <div>
            <div className="font-bold text-white text-base leading-none">CascadeIQ</div>
            <div className="text-xs mt-0.5" style={{ color: '#64748b' }}>Predictive Disease Cascade Intelligence</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-400" style={{ animation: 'pulse 2s infinite' }} />
          <span className="text-xs" style={{ color: '#64748b' }}>AI Engine Active</span>
        </div>
      </div>
    </header>
  )
}
