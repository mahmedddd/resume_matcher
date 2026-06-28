"use client";

import React, { useState, useEffect, useMemo } from 'react';
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
  Zap,
  Sparkles,
  Bookmark,
  FileText,
  Activity,
  GraduationCap,
  Code,
  BookOpen
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { internshipApi } from '@/lib/api';
import InternshipCard from '@/components/InternshipCard';
import EmailModal from '@/components/EmailModal';
import JobDetailModal from '@/components/JobDetailModal';
import AgentConsole from '@/components/AgentConsole';

// --- Types ---
interface Internship {
  id: number;
  title: string;
  company: string;
  city: string;
  is_remote: boolean;
  description: string;
  skills_required: string[];
  url: string;
  source: string;
  match_score?: number;
  matching_skills?: string[];
  missing_skills?: string[];
}

interface Application {
  id: number;
  internship_id: number;
  status: string;
  applied_at: string;
  company: string;
  job_title: string;
  notes?: string;
  questions_pending?: string[];
}

interface Profile {
  id: number;
  full_name: string;
  email: string;
  phone: string;
  linkedin_url: string;
  github_url: string;
  portfolio_url: string;
  education?: { degree: string; institution: string; year: string }[];
  skills?: string[];
  experience?: { title: string; company: string; duration: string; description: string }[];
  projects?: { name: string; description: string; technologies: string[] }[];
}

export default function Home() {
  const [internships, setInternships] = useState<Internship[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'browse' | 'match' | 'saved' | 'history' | 'profile'>('browse');
  const [matchingResults, setMatchingResults] = useState<any>(null);
  const [filter, setFilter] = useState({ city: '', search: '', remote_only: false });
  const [searchInput, setSearchInput] = useState('');
  const [scrapeStatus, setScrapeStatus] = useState({ is_running: false, last_result: {} });
  const [selectedJobForEmail, setSelectedJobForEmail] = useState<Internship | null>(null);
  const [viewingJob, setViewingJob] = useState<any>(null);
  const [savedJobs, setSavedJobs] = useState<Internship[]>([]);
  const [applications, setApplications] = useState<Application[]>([]);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [qaModal, setQaModal] = useState<{ appId: number; questions: string[] } | null>(null);
  const [qaAnswers, setQaAnswers] = useState<Record<string, string>>({});

  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        const saved = localStorage.getItem('savedJobs');
        if (saved) setSavedJobs(JSON.parse(saved));
      } catch (e) { console.error("Stored jobs parse error", e); }
    }
  }, []); // Only run once on mount

  useEffect(() => {
    fetchStats();
    fetchInternships();
    fetchProfile();
    fetchApplications();
    checkScrapeStatus();
    const interval = setInterval(checkScrapeStatus, 3000);
    return () => clearInterval(interval);
  }, [filter]); // Re-fetch on filter change

  // Persist saved jobs to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('savedJobs', JSON.stringify(savedJobs));
  }, [savedJobs]);

  const hasActiveApplications = useMemo(
    () => applications.some(a => [
      'Applying', 'Pending', 'Awaiting You', 'Needs LinkedIn', 'Paused'
    ].includes(a.status)),
    [applications]
  );

  // Real-time polling for active applications
  useEffect(() => {
    if (activeTab !== 'history') return;
    if (!hasActiveApplications) return;
    const poll = setInterval(() => { fetchApplications(); }, 10000);
    return () => clearInterval(poll);
  }, [activeTab, hasActiveApplications]);

  const fetchStats = async () => {
    try {
      const res = await internshipApi.getStats();
      setStats(res.data);
    } catch (e) { console.error(e); }
  };

  const fetchInternships = async () => {
    setLoading(true);
    try {
      const res = await internshipApi.getInternships(filter);
      setInternships(res.data.items);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  const fetchProfile = async () => {
    try {
      const res = await internshipApi.getProfile();
      setProfile(res.data);
    } catch (e) { console.error(e); }
  };

  const fetchApplications = async () => {
    try {
      const res = await internshipApi.getApplications();
      setApplications(res.data);
    } catch (e) { console.error(e); }
  };

  const checkScrapeStatus = async () => {
    try {
      const res = await internshipApi.getScrapeStatus();
      setScrapeStatus(res.data);
    } catch (e) { console.error(e); }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    setActiveTab('match');
    try {
      const res = await internshipApi.uploadCV(formData);
      setMatchingResults(res.data);
      fetchProfile(); // Refresh profile after CV upload
    } catch (e) {
      console.error(e);
      alert("Failed to match CV. Check backend console.");
    }
    setLoading(false);
  };

  const handleSubmitAnswers = async () => {
    if (!qaModal) return;
    try {
      await internshipApi.submitAnswers(qaModal.appId, qaAnswers);
      setQaModal(null);
      setQaAnswers({});
      fetchApplications();
    } catch (e) {
      console.error("Submit answers failed:", e);
      alert("Failed to submit answers. Please try again.");
    }
  };

  const handleScrape = async () => {
    try {
      await internshipApi.triggerScrape();
      setScrapeStatus({ ...scrapeStatus, is_running: true });
    } catch (e) { console.error(e); }
  };

  const handleSaveJob = (job: Internship) => {
    setSavedJobs(prev => {
      if (prev.some(j => j.id === job.id)) return prev;
      return [...prev, job];
    });
  };

  const handleUnsaveJob = (jobId: number) => {
    setSavedJobs(prev => prev.filter(j => j.id !== jobId));
  };

  const handleApply = async (jobId: number) => {
    try {
      await internshipApi.applyForInternship(jobId);
      await fetchApplications();
      setActiveTab('history'); // Navigate to Agent Console
    } catch (e) {
      console.error("Application failed:", e);
      alert("Failed to initiate application. Please try again.");
    }
  };

  const isJobSaved = (jobId: number) => savedJobs.some(job => job.id === jobId);

  return (
    <main className="max-w-[1240px] mx-auto px-6 py-10 min-h-screen">
      {/* Header */}
      <nav className="flex flex-col md:flex-row justify-between items-center gap-6 mb-16 p-4 glass-card border-white/40 sticky top-6 z-50">
        <div className="flex items-center gap-4">
          <div className="bg-gradient-to-br from-[#800020] via-[#9B111E] to-[#D4AF37] p-2.5 rounded-2xl shadow-glow">
            <Zap className="text-white fill-white" size={28} />
          </div>
          <div>
            <h1 className="text-2xl font-black tracking-tight text-[#1C1917]">
              SkillSync <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#800020] to-[#D4AF37]">PK</span>
            </h1>
            <p className="text-[10px] text-slate-500 font-bold tracking-[0.2em] uppercase">
              Production AI Agent • Pakistan Base
            </p>
          </div>
        </div>

        <div className="flex bg-slate-100/50 p-1.5 rounded-2xl border border-slate-200 shadow-sm backdrop-blur-md">
          <button
            onClick={() => { console.log('Tab change: browse'); setActiveTab('browse'); }}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all ${activeTab === 'browse' ? 'btn-primary shadow-glow' : 'text-slate-500 hover:text-slate-900 hover:bg-white/50'}`}
          >
            <Globe size={18} /> Discovery
          </button>
          <div className="relative">
            <label className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold cursor-pointer transition-all ${activeTab === 'match' ? 'btn-primary shadow-glow' : 'text-slate-500 hover:text-slate-900 hover:bg-white/50'}`}>
              <Upload size={18} /> CV Match
              <input type="file" hidden onChange={handleUpload} accept=".pdf" />
            </label>
          </div>
          <button
            onClick={() => { console.log('Tab change: saved'); setActiveTab('saved'); }}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all ${activeTab === 'saved' ? 'btn-primary shadow-glow' : 'text-slate-500 hover:text-slate-900 hover:bg-white/50'}`}
          >
            <Bookmark size={18} /> Saved
          </button>
          <button
            onClick={() => { console.log('Tab change: history'); setActiveTab('history'); }}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all ${activeTab === 'history' ? 'btn-primary shadow-glow' : 'text-slate-500 hover:text-slate-900 hover:bg-white/50'}`}
          >
            <Activity size={18} /> Agent Console
          </button>
          <button
            onClick={() => { console.log('Tab change: profile'); setActiveTab('profile'); }}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all ${activeTab === 'profile' ? 'btn-primary shadow-glow' : 'text-slate-500 hover:text-slate-900 hover:bg-white/50'}`}
          >
            <FileText size={18} /> Profile
          </button>
        </div>
      </nav>

      {/* Stats Section */}
      <AnimatePresence>
        {stats && activeTab === 'browse' && (
          <motion.section
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-16"
          >
            <StatCard label="Live Roles" value={stats.total_internships} icon={<Briefcase size={22} className="text-[#800020]" />} trend="+12% today" />
            <StatCard label="Tech Companies" value={stats.total_companies} icon={<Trophy size={22} className="text-[#D4AF37]" />} trend="Top Tier" />
            <StatCard label="Remote Work" value={stats.remote_count} icon={<Globe size={22} className="text-emerald-600" />} trend="WFH Basis" />
            <StatCard label="PK Cities" value={stats.by_city.length} icon={<MapPin size={22} className="text-amber-600" />} trend="Nationwide" />
          </motion.section>
        )}
      </AnimatePresence>

      {/* Main Content Area */}
      <AnimatePresence mode="wait">
        {activeTab === 'browse' ? (
          <motion.div
            key="browse"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.3 }}
          >
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
              <div>
                <h2 className="text-4xl font-black mb-2 flex items-center gap-3 text-[#1C1917] tracking-tight">
                  Marketplace <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#800020] to-[#D4AF37]">Discovery</span>
                  <span className="badge badge-burgundy">Real-time</span>
                </h2>
                <p className="text-slate-500 font-medium">Curated internship opportunities from across the Pakistan tech ecosystem.</p>
              </div>
              <button
                onClick={handleScrape}
                disabled={scrapeStatus.is_running}
                className="btn-secondary flex items-center gap-2 text-xs py-2 disabled:opacity-50"
              >
                <RefreshCw size={14} className={scrapeStatus.is_running ? 'animate-spin' : ''} />
                {scrapeStatus.is_running ? 'Agent Synchronizing...' : 'Re-sync Data'}
              </button>
            </div>

            {/* Search Box */}
            <div className="relative mb-6">
              <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#A8A29E]" />
              <input
                type="text"
                placeholder="Search by title, company or skill…"
                value={searchInput}
                onChange={e => {
                  setSearchInput(e.target.value);
                  setFilter(f => ({ ...f, search: e.target.value }));
                }}
                className="w-full pl-11 pr-5 py-3 bg-white border border-slate-200 rounded-2xl text-sm font-medium text-[#57534E] placeholder:text-slate-300 focus:outline-none focus:ring-2 focus:ring-[#800020]/20 focus:border-[#800020]/40 shadow-sm transition-all"
              />
            </div>

            {/* City Filters */}
            <div className="flex gap-3 mb-12 overflow-x-auto pb-4 custom-scrollbar">
              {['All', 'Lahore', 'Karachi', 'Islamabad', 'Remote'].map(c => (
                <button
                  key={c}
                  onClick={() => setFilter(f => ({ ...f, city: c === 'All' ? '' : c }))}
                  className={`px-8 py-2.5 rounded-2xl text-xs font-bold border transition-all whitespace-nowrap ${(filter.city === c || (c === 'All' && !filter.city))
                      ? 'btn-primary shadow-glow'
                      : 'bg-white text-slate-500 border-slate-200 hover:border-[#800020]/40 hover:text-[#800020] shadow-sm'
                    }`}
                >
                  {c.toUpperCase()}
                </button>
              ))}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
              {internships.map(job => (
                <InternshipCard
                  key={job.id}
                  job={job}
                  onCardClick={(j) => setViewingJob(j)}
                  onDraftEmail={(j) => setSelectedJobForEmail(j)}
                  onAutoApply={() => handleApply(job.id)}
                />
              ))}
              {internships.length === 0 && !loading && (
                <div className="col-span-full py-24 glass-card flex flex-col items-center justify-center text-center opacity-40">
                  <AlertCircle size={48} className="mb-4" />
                  <p className="text-lg font-bold">Resume Matcher logs are empty.</p>
                  <p className="text-sm">Run a fresh scrape to populate the marketplace.</p>
                </div>
              )}
            </div>
          </motion.div>
        ) : activeTab === 'match' ? (
          <motion.div
            key="match"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
          >
            <div className="max-w-4xl mx-auto text-center mb-12">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#FFFBFB] border border-[#FEE2E2] text-[#800020] text-[0.65rem] font-bold uppercase tracking-wider mb-6 shadow-sm">
                <Sparkles size={12} /> Semantic Search Active
              </div>
              <h2 className="text-5xl font-black mb-4 tracking-tight text-[#1C1917] leading-tight">Your Personalized <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#800020] to-[#D4AF37]">Matches</span></h2>
              <p className="text-slate-500 text-lg font-medium leading-relaxed">
                We've cross-referenced your CV embeddings against {stats?.total_internships || 'hundreds of'} openings
                using high-dimensional vector similarity.
              </p>
            </div>

            {loading ? (
              <div className="py-24 text-center">
                <div className="relative w-20 h-20 mx-auto mb-8">
                  <div className="absolute inset-0 border-4 border-[#800020]/20 rounded-full"></div>
                  <div className="absolute inset-0 border-4 border-[#800020] border-t-transparent rounded-full animate-spin"></div>
                  <Zap size={32} className="absolute inset-0 m-auto text-[#800020] animate-pulse" />
                </div>
                <p className="text-xl font-bold text-slate-300">Extracting CV Skills...</p>
                <p className="text-sm text-slate-500 mt-2 italic">Building your semantic profile via Gemini Flash v2</p>
              </div>) : matchingResults ? (
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-8">
                  <div className="col-span-full flex flex-col lg:flex-row gap-6 mb-8">
                    {matchingResults.cv_skills.projects?.length > 0 && (
                      <div className="glass-card p-6 flex flex-col flex-1 border-emerald-100 shadow-sm bg-white/40">
                        <h4 className="text-[10px] font-black tracking-widest text-slate-400 uppercase flex items-center gap-2 mb-4 shrink-0">
                          <Sparkles size={14} className="text-emerald-500" /> Structural Projects
                        </h4>
                        <div className="flex flex-col gap-4 text-sm text-slate-600 max-h-52 overflow-y-auto pr-2 custom-scrollbar">
                          {matchingResults.cv_skills.projects.map((proj: any, i: number) => (
                            <div key={i} className="pl-4 border-l-2 border-emerald-500/20 py-1 hover:border-emerald-500/50 transition-colors">
                              <div className="font-bold text-slate-800">{proj.name}</div>
                              <div className="text-[10px] text-emerald-600 font-bold mb-1 opacity-80">{proj.technologies?.join(" • ")}</div>
                              <div className="text-xs text-slate-500 leading-relaxed font-medium">{proj.description}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    {matchingResults.cv_skills.work_experience?.length > 0 && (
                      <div className="glass-card p-6 flex flex-col flex-1 border-[#FEE2E2] shadow-sm bg-white/40">
                        <h4 className="text-[10px] font-black tracking-widest text-[#A8A29E] uppercase flex items-center gap-2 mb-4 shrink-0">
                          <Briefcase size={14} className="text-[#800020]" /> Work Geometry
                        </h4>
                        <div className="flex flex-col gap-4 text-sm text-[#57534E] max-h-52 overflow-y-auto pr-2 custom-scrollbar">
                          {matchingResults.cv_skills.work_experience.map((exp: any, i: number) => (
                            <div key={i} className="pl-4 border-l-2 border-[#800020]/20 py-1 hover:border-[#800020]/50 transition-colors">
                              <div className="font-bold text-[#1C1917]">{exp.title} <span className="text-[#A8A29E] font-medium">@ {exp.company}</span></div>
                              <div className="text-[10px] text-[#800020] font-bold mb-1 opacity-80">{exp.duration}</div>
                              {exp.achievements?.map((ach: string, j: number) => (
                                <div key={j} className="text-xs text-[#57534E] flex items-start gap-2 mt-1.5 font-medium leading-relaxed">
                                  <span className="text-[#800020] shrink-0 mt-1">●</span> {ach}
                                </div>
                              ))}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                  {matchingResults.matches.map((job: any) => (
                    <InternshipCard
                      key={job.id}
                      job={job}
                      showMatch
                      onCardClick={(j) => setViewingJob(j)}
                      onDraftEmail={(j) => setSelectedJobForEmail(j)}
                      onAutoApply={() => handleApply(job.id)}
                    />
                  ))}
                </div>
              ) : (
              <div className="max-w-md mx-auto py-20 text-center glass-card border-dashed border-slate-300 bg-white/40">
                <Upload size={48} className="mx-auto mb-4 text-[#A8A29E]" />
                <p className="text-[#57534E] font-bold mb-6">No matching data found. Upload your CV to trigger the NLP engine.</p>
                <label className="btn-primary inline-flex cursor-pointer shadow-glow">
                  Upload PDF CV
                  <input type="file" hidden onChange={handleUpload} accept=".pdf" />
                </label>
              </div>
            )}
          </motion.div>
        ) : activeTab === 'saved' ? (
          <motion.div
            key="saved"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.3 }}
          >
            <h2 className="text-3xl font-black mb-8 text-[#1C1917]">Your Saved Internships</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
              {savedJobs.length > 0 ? (
                savedJobs.map(job => (
                  <InternshipCard
                    key={job.id}
                    job={job}
                    onCardClick={(j) => setViewingJob(j)}
                    onDraftEmail={(j) => setSelectedJobForEmail(j)}
                    onAutoApply={() => handleApply(job.id)}
                  />
                ))
              ) : (
                <div className="col-span-full py-24 glass-card flex flex-col items-center justify-center text-center opacity-40">
                  <Bookmark size={48} className="mb-4" />
                  <p className="text-lg font-bold">No saved internships yet.</p>
                  <p className="text-sm">Browse the marketplace and save jobs you're interested in!</p>
                </div>
              )}
            </div>
          </motion.div>
        ) : activeTab === 'history' ? (
          <motion.div
            key="history"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.3 }}
          >
            <AgentConsole
              applications={applications}
              onSubmitAnswers={async (appId, answers) => {
                try {
                  await internshipApi.submitAnswers(appId, answers);
                  await fetchApplications();
                } catch (e) { console.error('Submit answers failed:', e); }
              }}
              onRefresh={fetchApplications}
            />
          </motion.div>
        ) : activeTab === 'profile' ? (
          <motion.div
            key="profile"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.3 }}
          >
            <h2 className="text-4xl font-black mb-8 text-[#1C1917] tracking-tight">Your <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#800020] to-[#D4AF37]">Talent Profile</span></h2>
            {profile ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="glass-card p-8 flex flex-col gap-6 bg-white/60 border-white/80 shadow-medium">
                  <h3 className="text-2xl font-black text-[#1C1917] flex items-center gap-3">
                    <FileText size={24} className="text-[#800020]" /> Personal Identity
                  </h3>
                  <div className="space-y-4">
                    <p className="flex justify-between border-b border-slate-100 pb-2"><span className="font-bold text-[#A8A29E] text-xs uppercase tracking-widest">Full Name</span> <span className="font-bold text-[#57534E]">{profile.full_name}</span></p>
                    <p className="flex justify-between border-b border-slate-100 pb-2"><span className="font-bold text-[#A8A29E] text-xs uppercase tracking-widest">Email</span> <span className="font-bold text-[#57534E]">{profile.email}</span></p>
                    <p className="flex justify-between border-b border-slate-100 pb-2"><span className="font-bold text-[#A8A29E] text-xs uppercase tracking-widest">Phone</span> <span className="font-bold text-[#57534E]">{profile.phone}</span></p>
                    <p className="flex justify-between border-b border-slate-100 pb-2"><span className="font-bold text-[#A8A29E] text-xs uppercase tracking-widest">LinkedIn</span> <a href={profile.linkedin_url?.startsWith('http') ? profile.linkedin_url : `https://${profile.linkedin_url}`} target="_blank" rel="noopener noreferrer" className="text-[#800020] hover:underline font-bold">View Profile</a></p>
                    <p className="flex justify-between border-b border-slate-100 pb-2"><span className="font-bold text-[#A8A29E] text-xs uppercase tracking-widest">GitHub</span> <a href={profile.github_url?.startsWith('http') ? profile.github_url : `https://${profile.github_url}`} target="_blank" rel="noopener noreferrer" className="text-[#800020] hover:underline font-bold">Repository</a></p>
                  </div>
                </div>

                <div className="glass-card p-8 flex flex-col gap-6 bg-white/60 border-white/80 shadow-medium">
                  <h3 className="text-2xl font-black text-[#1C1917] flex items-center gap-3">
                    <GraduationCap size={24} className="text-[#D4AF37]" /> Academic Path
                  </h3>
                  {profile.education && profile.education.length > 0 ? (
                    profile.education.map((edu: any, i: number) => (
                      <div key={i} className="border-l-4 border-[#D4AF37]/20 pl-4 py-1 hover:border-[#D4AF37]/50 transition-colors">
                        <p className="font-black text-[#1C1917]">{edu.degree}</p>
                        <p className="text-[#57534E] font-bold text-xs">{edu.institution}</p>
                        <p className="text-xs text-[#800020] font-black mt-1 uppercase">{edu.year}</p>
                      </div>
                    ))
                  ) : (
                    <p className="text-[#57534E]">No education details found.</p>
                  )}

                  <h3 className="text-2xl font-black text-[#1C1917] flex items-center gap-3 mt-6">
                    <Code size={24} className="text-[#800020]" /> Verified Skills
                  </h3>
                  {profile.skills && profile.skills.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {profile.skills.map((skill: any, i: number) => (
                        <span key={i} className="badge badge-burgundy">
                          {skill}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="text-[#57534E]">No skills found.</p>
                  )}
                </div>

                <div className="glass-card p-8 flex flex-col gap-6 lg:col-span-2 bg-white/60 border-white/80 shadow-medium">
                  <h3 className="text-2xl font-black text-[#1C1917] flex items-center gap-3">
                    <Briefcase size={24} className="text-[#800020]" /> Professional Geometry
                  </h3>
                  {profile.experience && profile.experience.length > 0 ? (
                    profile.experience.map((exp: any, i: number) => (
                      <div key={i} className="border-l-4 border-[#800020]/20 pl-6 py-2 hover:border-[#800020]/50 transition-all group">
                        <p className="font-black text-[#1C1917] text-lg group-hover:text-[#800020] transition-colors">{exp.title} <span className="text-[#A8A29E] font-medium">@ {exp.company}</span></p>
                        <p className="text-xs text-[#800020] font-black uppercase tracking-widest mt-1">{exp.duration}</p>
                        <p className="text-[#57534E] mt-3 font-medium leading-relaxed">{exp.description}</p>
                      </div>
                    ))
                  ) : (
                    <p className="text-[#57534E]">No work experience found.</p>
                  )}
                </div>

                <div className="glass-card p-8 flex flex-col gap-6 lg:col-span-2 bg-white/60 border-white/80 shadow-medium">
                  <h3 className="text-2xl font-black text-[#1C1917] flex items-center gap-3">
                    <BookOpen size={24} className="text-emerald-500" /> Innovation Deck
                  </h3>
                  {profile.projects && profile.projects.length > 0 ? (
                    profile.projects.map((proj: any, i: number) => (
                      <div key={i} className="border-l-4 border-emerald-500/20 pl-6 py-2 hover:border-emerald-500/50 transition-all group">
                        <p className="font-black text-[#1C1917] text-lg group-hover:text-emerald-600 transition-colors">{proj.name}</p>
                        <p className="text-xs text-emerald-600 font-black uppercase tracking-widest mt-1">Tech Stack: {proj.technologies?.join(' • ')}</p>
                        <p className="text-[#57534E] mt-3 font-medium leading-relaxed">{proj.description}</p>
                      </div>
                    ))
                  ) : (
                    <p className="text-[#57534E]">No projects found.</p>
                  )}
                </div>
              </div>
            ) : (
              <div className="py-24 glass-card flex flex-col items-center justify-center text-center opacity-40">
                <FileText size={48} className="mb-4" />
                <p className="text-lg font-bold">No profile data available.</p>
                <p className="text-sm">Upload your CV in the "CV Match" tab to populate your profile.</p>
              </div>
            )}
          </motion.div>
        ) : null}
      </AnimatePresence>

      {/* Human-in-Loop Q&A Modal */}
      {qaModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-900/40 backdrop-blur-xl p-4">
          <div className="bg-white/90 border border-white/40 rounded-[2.5rem] p-10 max-w-2xl w-full shadow-2xl relative overflow-hidden glass-card">
            <div className="absolute top-0 left-0 w-full h-1.5 bg-gradient-to-r from-[#800020] via-[#9B111E] to-[#D4AF37]"></div>
            <div className="flex items-start gap-6 mb-8">
              <div className="p-4 bg-amber-50 rounded-2xl border border-amber-100 text-[#D4AF37] shrink-0 shadow-sm">
                <Zap size={28} className="fill-[#D4AF37]" />
              </div>
              <div>
                <h3 className="text-2xl font-black text-[#1C1917] mb-2 tracking-tight">Personal Voice Required</h3>
                <p className="text-[#57534E] text-sm font-medium leading-relaxed">The agent has encountered open-ended questions that require your unique perspective. Your responses will be used to complete the application autonomously.</p>
              </div>
            </div>
            <div className="flex flex-col gap-6 max-h-[400px] overflow-y-auto pr-2 custom-scrollbar">
              {qaModal.questions.map((q, i) => (
                <div key={i} className="group">
                  <label className="block text-[10px] font-black text-[#800020] uppercase tracking-widest mb-2 px-1">Question {i + 1}: {q}</label>
                  <textarea
                    className="w-full bg-slate-50 border border-slate-200 rounded-2xl p-5 text-sm text-[#57534E] focus:border-[#800020]/40 focus:ring-4 focus:ring-[#800020]/5 outline-none resize-none min-h-[120px] transition-all font-medium placeholder:text-slate-400"
                    placeholder="Your personal response..."
                    value={qaAnswers[q] || ""}
                    onChange={(e) => setQaAnswers(prev => ({ ...prev, [q]: e.target.value }))}
                  />
                </div>
              ))}
            </div>
            <div className="flex justify-between items-center mt-10">
              <button onClick={() => { setQaModal(null); setQaAnswers({}); }} className="px-6 py-3 text-sm font-bold text-[#A8A29E] hover:text-[#1C1917] transition-colors">Discard & Close</button>
              <button
                onClick={handleSubmitAnswers}
                disabled={qaModal.questions.some(q => !qaAnswers[q]?.trim())}
                className="btn-primary px-10 py-4 shadow-glow flex items-center gap-3 disabled:opacity-30"
              >
                <Zap size={18} className="fill-white" />
                <span className="font-black">Submit & Resume Agent</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* AI Outreach Modal */}
      {selectedJobForEmail && (
        <EmailModal
          isOpen={!!selectedJobForEmail}
          onClose={() => setSelectedJobForEmail(null)}
          job={selectedJobForEmail}
          userSkills={activeTab === 'match' ? (matchingResults?.cv_skills?.skills || []) : []}
        />
      )}

      {/* Embedded Job Detail Modal */}
      {viewingJob && (
        <JobDetailModal
          job={viewingJob}
          isSaved={isJobSaved(viewingJob.id)}
          onClose={() => setViewingJob(null)}
          onSave={(j) => isJobSaved(j.id) ? handleUnsaveJob(j.id) : handleSaveJob(j)}
          onApply={() => handleApply(viewingJob.id)}
          onDraftEmail={(j) => setSelectedJobForEmail(j)}
        />
      )}
    </main>
  );
}

function StatCard({ label, value, icon, trend }: any) {
  return (
    <div className="glass-card p-6 flex flex-col gap-1 relative overflow-hidden group bg-white/40 border-white/60 shadow-medium hover:-translate-y-1 transition-all duration-300">
      <div className="absolute -right-4 -bottom-4 opacity-[0.05] group-hover:scale-125 transition-transform duration-500 text-slate-900">
        {icon}
      </div>
      <div className="bg-slate-100 w-12 h-12 rounded-2xl flex items-center justify-center mb-4 border border-slate-200 group-hover:bg-white transition-colors shadow-sm text-[#800020]">
        {icon}
      </div>
      <h4 className="text-3xl font-black text-[#1C1917] tracking-tight">{value}</h4>
      <div className="flex justify-between items-center text-[10px] mt-1">
        <span className="text-[#57534E] font-bold uppercase tracking-wider">{label}</span>
        <span className="text-[#800020] font-black bg-[#FFFBFB] px-2 py-0.5 rounded-full border border-[#FEE2E2]">{trend}</span>
      </div>
    </div>
  )
}
