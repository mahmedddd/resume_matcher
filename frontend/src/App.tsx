import React, { useState, useEffect } from 'react'
import { internshipApi } from './api'
import { 
  Briefcase, 
  MapPin, 
  Search, 
  Upload, 
  RefreshCw, 
  Globe, 
  CheckCircle2, 
  AlertCircle,
  Trophy,
  Zap
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

// --- Types ---
interface Internship {
  id: number
  title: string
  company: string
  city: string
  is_remote: boolean
  description: string
  skills_required: string[]
  url: string
  source: string
  scraped_at: string
  match_score?: number
  matching_skills?: string[]
  missing_skills?: string[]
}

interface Stats {
  total_internships: number
  total_companies: number
  remote_count: number
  onsite_count: number
  by_city: { city: string; count: number }[]
}

function App() {
  const [internships, setInternships] = useState<Internship[]>([])
  const [stats, setStats] = useState<Stats | null>(null)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'browse' | 'match'>('browse')
  const [matchingResults, setMatchingResults] = useState<{ matches: Internship[], cv_skills: any } | null>(null)
  const [filter, setFilter] = useState({ city: '', search: '', remote_only: false })
  const [scrapeStatus, setScrapeStatus] = useState({ is_running: false, last_result: {} })

  useEffect(() => {
    fetchStats()
    fetchInternships()
    const interval = setInterval(checkScrapeStatus, 3000)
    return () => clearInterval(interval)
  }, [filter])

  const fetchStats = async () => {
    try {
      const res = await internshipApi.getStats()
      setStats(res.data)
    } catch (e) { console.error(e) }
  }

  const fetchInternships = async () => {
    setLoading(true)
    try {
      const res = await internshipApi.getInternships(filter)
      setInternships(res.data.items)
    } catch (e) { console.error(e) }
    setLoading(false)
  }

  const checkScrapeStatus = async () => {
    try {
      const res = await internshipApi.getScrapeStatus()
      setScrapeStatus(res.data)
    } catch (e) { console.error(e) }
  }

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return
    const file = e.target.files[0]
    const formData = new FormData()
    formData.append('file', file)
    
    setLoading(true)
    setActiveTab('match')
    try {
      const res = await internshipApi.uploadCV(formData)
      setMatchingResults(res.data)
    } catch (e) { 
      alert("Failed to match CV. Ensure backend is running.")
    }
    setLoading(false)
  }

  const handleScrape = async () => {
    try {
      await internshipApi.triggerScrape()
      setScrapeStatus({ ...scrapeStatus, is_running: true })
    } catch (e) { console.error(e) }
  }

  return (
    <div className="container" style={{ padding: '40px 20px' }}>
      {/* Header */}
      <nav style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '60px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ 
            background: 'linear-gradient(135deg, #6366f1, #a855f7)', 
            padding: '10px', 
            borderRadius: '12px',
            boxShadow: '0 0 20px var(--primary-glow)'
          }}>
            <Shield size={32} />
          </div>
          <div>
            <h1 style={{ fontSize: '1.5rem', fontWeight: 800 }}>RESUME<span style={{ color: '#a855f7' }}>MATCHER</span></h1>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', fontWeight: 500 }}>AI AGENT v1.0 • PAKISTAN</p>
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: '16px' }}>
          <button className={`btn-primary ${activeTab === 'browse' ? 'active' : ''}`} 
                  onClick={() => setActiveTab('browse')}
                  style={{ background: activeTab === 'match' ? 'transparent' : '', border: activeTab === 'match' ? '1px solid var(--glass-border)' : '' }}>
            <Globe size={18} /> Browse
          </button>
          <label className="btn-primary" style={{ cursor: 'pointer' }}>
            <Upload size={18} /> Match CV
            <input type="file" hidden onChange={handleUpload} accept=".pdf" />
          </label>
        </div>
      </nav>

      {/* Stats Section */}
      {stats && activeTab === 'browse' && (
        <section className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', marginBottom: '60px' }}>
           <StatCard label="Total Internships" value={stats.total_internships} icon={<Briefcase color="#818cf8"/>} />
           <StatCard label="Active Companies" value={stats.total_companies} icon={<Trophy color="#f472b6"/>} />
           <StatCard label="Remote Roles" value={stats.remote_count} icon={<Globe color="#2dd4bf"/>} />
           <StatCard label="Cities Tracked" value={stats.by_city.length} icon={<MapPin color="#fbbf24"/>} />
        </section>
      )}

      {/* Dynamic Content */}
      <AnimatePresence mode="wait">
        {activeTab === 'browse' ? (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '32px' }}>
              <h2 style={{ fontSize: '1.8rem' }}>Discovery Engine</h2>
              <button 
                onClick={handleScrape} 
                disabled={scrapeStatus.is_running}
                className="btn-primary" 
                style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid var(--glass-border)', fontSize: '0.8rem' }}>
                <RefreshCw size={14} className={scrapeStatus.is_running ? 'animate-spin' : ''} />
                {scrapeStatus.is_running ? 'Scraping Live...' : 'Refresh Live Data'}
              </button>
            </div>
            
            {/* Filters */}
            <div style={{ display: 'flex', gap: '12px', marginBottom: '40px', flexWrap: 'wrap' }}>
              {['', 'Lahore', 'Karachi', 'Islamabad', 'Remote'].map(c => (
                <button key={c} 
                  onClick={() => setFilter({ ...filter, city: c })}
                  className={`glass-card ${filter.city === c ? 'pill-active' : ''}`}
                  style={{ padding: '10px 20px', borderRadius: '30px', cursor: 'pointer', border: '1px solid var(--glass-border)' }}>
                  {c || 'All Pakistan'}
                </button>
              ))}
            </div>

            <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))' }}>
              {internships.map(job => <InternshipCard key={job.id} job={job} />)}
              {internships.length === 0 && !loading && (
                <div style={{ gridColumn: '1/-1', textAlign: 'center', padding: '100px', opacity: 0.5 }}>
                  <AlertCircle size={48} style={{ marginBottom: '16px' }} />
                  <p>No internships found in the logs. Try refreshing data.</p>
                </div>
              )}
            </div>
          </motion.div>
        ) : (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
             <div style={{ textAlign: 'center', marginBottom: '60px' }}>
               <h2 style={{ fontSize: '2.5rem', marginBottom: '12px' }}>Resume Matcher AI <span style={{ color: '#a855f7' }}>Matching</span></h2>
               <p style={{ color: 'var(--text-muted)' }}>We've analyzed your CV and compared it against {stats?.total_internships} opportunities across Pakistan.</p>
             </div>

             {loading ? (
               <div style={{ textAlign: 'center', padding: '100px' }}>
                 <Zap size={48} className="animate-pulse" style={{ color: '#6366f1', marginBottom: '24px' }} />
                 <p>Analyzing CV and generating embeddings...</p>
               </div>
             ) : matchingResults && (
               <div className="grid">
                 {matchingResults.matches.map(job => <InternshipCard key={job.id} job={job} showMatch />)}
               </div>
             )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function StatCard({ label, value, icon }: any) {
  return (
    <div className="glass-card" style={{ padding: '24px', display: 'flex', alignItems: 'center', gap: '20px' }}>
      <div style={{ padding: '12px', borderRadius: '12px', background: 'rgba(255,255,255,0.05)' }}>{icon}</div>
      <div>
        <h4 style={{ fontSize: '1.5rem', fontWeight: 700 }}>{value}</h4>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>{label}</p>
      </div>
    </div>
  )
}

function InternshipCard({ job, showMatch }: { job: Internship, showMatch?: boolean }) {
  return (
    <motion.div layout className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h3 style={{ fontSize: '1.2rem', marginBottom: '4px' }}>{job.title}</h3>
          <p style={{ color: 'var(--primary)', fontWeight: 600, fontSize: '0.9rem' }}>{job.company}</p>
        </div>
        {showMatch && (
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#4ade80' }}>{Math.round(job.match_score || 0)}%</div>
            <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', fontWeight: 700 }}>MATCH SCORE</div>
          </div>
        )}
      </div>

      <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          <MapPin size={14} /> {job.city}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          {job.is_remote ? <Globe size={14} color="#4ade80" /> : <Briefcase size={14} />} 
          {job.is_remote ? 'Remote' : 'On-site'}
        </div>
      </div>

      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        {job.skills_required.slice(0, 4).map(s => <span key={s} className="badge">{s}</span>)}
        {showMatch && job.matching_skills?.slice(0, 3).map(s => (
          <span key={s} className="badge badge-green" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <CheckCircle2 size={10} /> {s}
          </span>
        ))}
      </div>

      <a href={job.url} target="_blank" rel="noopener noreferrer" className="btn-primary" style={{ marginTop: 'auto', justifyContent: 'center', fontSize: '0.9rem' }}>
        View Opportunity
      </a>
    </motion.div>
  )
}

function Shield(props: any) {
  return (
    <svg {...props} width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    </svg>
  )
}

export default App
