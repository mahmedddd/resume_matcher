"use client";

import { MapPin, Globe, Briefcase, CheckCircle2, Zap, Mail, Building, ShieldCheck, AlertTriangle, ShieldAlert, Clock, ExternalLink } from 'lucide-react';
import { motion } from 'framer-motion';

interface ReputationAnalysis {
  status: 'green' | 'yellow' | 'red';
  summary: string;
}

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
  deadline?: string;
  salary?: string;
  match_score?: number;
  matching_skills?: string[];
  missing_skills?: string[];
  reputation?: ReputationAnalysis;
}

export default function InternshipCard({ 
  job, 
  showMatch, 
  onCardClick,
  onDraftEmail,
  onAutoApply
}: { 
  job: Internship; 
  showMatch?: boolean;
  onCardClick?: (job: Internship) => void;
  onDraftEmail?: (job: Internship) => void;
  onAutoApply?: (job: Internship) => void;
}) {
  return (
    <motion.div 
      layout 
      className="glass-card p-6 flex flex-col gap-6 h-full relative group bg-white/40 border-white/60 shadow-medium hover:shadow-xl transition-all duration-300"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      whileHover={{ y: -5 }}
    >
      <div className="absolute top-0 left-0 w-full h-1.5 bg-gradient-to-r from-[#800020] via-[#9B111E] to-[#D4AF37] opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
      
      {/* Clickable Card Body - opens detail modal */}
      <div className="flex flex-col gap-5 cursor-pointer" onClick={() => onCardClick?.(job)}>
        <div className="flex justify-between items-start gap-4">
          <div>
            <h3 className="text-xl font-black text-[#1C1917] mb-1.5 leading-tight tracking-tight group-hover:text-[#800020] transition-all">{job.title}</h3>
            <p className="flex items-center gap-1.5 text-[#57534E] font-bold text-sm">
              <Building size={14} className="text-[#800020]" />
              {job.company}
            </p>
          </div>
          {showMatch && (
            <div className="shrink-0 text-right bg-[#FFFBFB] p-2.5 rounded-2xl border border-[#FEE2E2] shadow-sm">
              <div className="text-2xl font-black text-transparent bg-clip-text bg-gradient-to-b from-[#800020] to-[#E11D48] leading-none">
                {Math.round(job.match_score || 0)}<span className="text-sm">%</span>
              </div>
              <div className="text-[10px] text-[#800020]/60 font-black tracking-widest mt-1 uppercase">
                Match
              </div>
            </div>
          )}
        </div>

        {job.reputation && (
          <div className={`p-4 rounded-2xl border flex gap-3 items-start transition-all ${
            job.reputation.status === 'green' ? 'bg-emerald-50/50 border-emerald-100 text-emerald-800' :
            job.reputation.status === 'yellow' ? 'bg-amber-50/50 border-amber-100 text-amber-800' :
            'bg-red-50/50 border-red-100 text-red-800'
          }`}>
            <div className="shrink-0 mt-0.5">
              {job.reputation.status === 'green' && <ShieldCheck size={20} className="text-emerald-500" />}
              {job.reputation.status === 'yellow' && <AlertTriangle size={20} className="text-amber-500" />}
              {job.reputation.status === 'red' && <ShieldAlert size={20} className="text-red-500" />}
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] font-black uppercase tracking-widest mb-0.5 opacity-60">
                {job.reputation.status === 'green' ? 'Security Clearance: Verified' :
                 job.reputation.status === 'yellow' ? 'System Intel: Mixed' : 'Threat Level: high-risk'}
              </span>
              <span className="text-xs font-bold leading-relaxed">{job.reputation.summary}</span>
            </div>
          </div>
        )}

        <div className="flex gap-4 flex-wrap bg-slate-50/50 py-2.5 px-4 rounded-xl border border-slate-100">
          <div className="flex items-center gap-1.5 text-[11px] font-bold text-[#57534E] truncate max-w-[120px]" title={job.city}>
            <MapPin size={14} className="text-[#800020]/60 shrink-0" /> {job.city.toUpperCase() || 'PAKISTAN'}
          </div>
          <div className="flex items-center gap-1.5 text-[11px] font-bold text-[#57534E]">
            {job.is_remote ? <Globe size={14} className="text-emerald-500 shrink-0" /> : <Briefcase size={14} className="text-[#800020]/60 shrink-0" />} 
            {job.is_remote ? 'REMOTE' : 'ON-SITE'}
          </div>
          <div className="flex items-center gap-1.5 text-[11px] font-black text-rose-700">
             <Clock size={14} className="text-rose-500 shrink-0" /> {job.deadline || 'Check site for deadline'}
          </div>
          <div className="flex items-center gap-1.5 text-[11px] font-black text-emerald-700">
             <span className="text-emerald-500 shrink-0">$</span> {job.salary || 'Visit site for salary details'}
          </div>
        </div>

        <p className="text-sm text-[#57534E]/90 leading-relaxed line-clamp-2 italic border-l-2 border-[#800020]/20 pl-4 font-medium">
          "{job.description || `Exciting opportunity at ${job.company}. Access deep-intel to review full requirements and execute match sequence.`}"
        </p>

        <div className="flex flex-col gap-2.5">
          <span className="text-[10px] font-black text-[#A8A29E] uppercase tracking-widest px-1">Structural Requirements</span>
          <div className="flex gap-2 flex-wrap">
            {job.skills_required.map(s => (
              <span key={s} className="badge badge-burgundy">{s}</span>
            ))}
            {showMatch && job.matching_skills?.slice(0, 3).map(s => (
              <span key={s} className="badge badge-emerald flex items-center gap-1">
                <CheckCircle2 size={10} /> {s}
              </span>
            ))}
          </div>
        </div>
      </div>{/* end clickable body */}

      <div className="mt-auto pt-6 flex gap-2.5 border-t border-slate-100" onClick={(e) => e.stopPropagation()}>
        <a 
          href={job.url?.startsWith('http') ? job.url : job.url ? `https://${job.url}` : '#'} 
          onClick={(e) => { if (!job.url) e.preventDefault(); }}
          target="_blank" 
          rel="noopener noreferrer" 
          className="flex-1 bg-white hover:bg-[#FFFBFB] text-[#1C1917] border border-slate-200 flex items-center justify-center gap-2 text-xs py-3 rounded-2xl font-bold transition-all shadow-sm active:scale-95"
        >
          <ExternalLink size={14} className="text-[#800020]" /> {job.source.toLowerCase() === 'linkedin' ? 'LinkedIn' : 'View Posting'}
        </a>
        <button 
          onClick={() => onAutoApply?.(job)}
          className="px-6 py-3 btn-primary text-[11px] shadow-glow active:scale-95"
          title="Autonomous Application"
        >
          <Zap size={14} className="fill-white animate-pulse" /> 
        </button>
        <button 
          onClick={() => onDraftEmail?.(job)}
          className="p-3.5 rounded-2xl border border-slate-200 bg-white hover:bg-[#FFFBFB] hover:border-[#FEE2E2] transition-all text-[#A8A29E] hover:text-[#800020] shadow-sm active:scale-95"
          title="Draft AI Outreach Email"
        >
          <Mail size={18} />
        </button>
      </div>
    </motion.div>
  );
}
