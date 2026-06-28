"use client";

import { motion, AnimatePresence } from 'framer-motion';
import {
  X, MapPin, Globe, Briefcase, ExternalLink,
  Bookmark, BookmarkCheck, Zap, Mail, Building2,
  CalendarDays, DollarSign, CheckCircle2, XCircle,
  ShieldCheck, AlertTriangle, ShieldAlert, Link2, Code
} from 'lucide-react';

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

interface JobDetailModalProps {
  job: Internship | null;
  isSaved: boolean;
  onClose: () => void;
  onSave: (job: Internship) => void;
  onApply: (job: Internship) => void;
  onDraftEmail: (job: Internship) => void;
}

export default function JobDetailModal({
  job, isSaved, onClose, onSave, onApply, onDraftEmail
}: JobDetailModalProps) {
  if (!job) return null;

  const repConfig = job.reputation ? {
    green: { bg: 'bg-green-50', border: 'border-green-200', text: 'text-green-700', label: 'Verified Safe' },
    yellow: { bg: 'bg-amber-50', border: 'border-amber-200', text: 'text-amber-700', label: 'Proceed with Caution' },
    red: { bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700', label: '⚠️ High Risk: Potential Scam' },
  }[job.reputation.status] : null;

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 z-50 flex items-center justify-center p-4 md:p-8"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        {/* Backdrop */}
        <motion.div
          className="absolute inset-0 bg-[#1C1917]/50 backdrop-blur-sm"
          onClick={onClose}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        />

        {/* Modal Panel */}
        <motion.div
          className="relative w-full max-w-3xl max-h-[90vh] overflow-y-auto rounded-[2rem] bg-white/95 border border-white/60 shadow-2xl custom-scrollbar"
          style={{ backdropFilter: 'blur(20px)' }}
          initial={{ opacity: 0, scale: 0.93, y: 24 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.93, y: 24 }}
          transition={{ type: 'spring', damping: 28, stiffness: 320 }}
        >
          {/* Top Accent Bar */}
          <div className="absolute top-0 left-0 right-0 h-1.5 bg-gradient-to-r from-[#800020] via-[#9B111E] to-[#D4AF37] rounded-t-[2rem]" />

          {/* Close button */}
          <button
            onClick={onClose}
            className="absolute top-5 right-5 z-10 p-2 rounded-full bg-slate-100 border border-slate-200 hover:bg-[#FEE2E2] hover:border-[#800020]/30 transition-all text-slate-400 hover:text-[#800020]"
          >
            <X size={18} />
          </button>

          <div className="p-8 md:p-10">
            {/* Header */}
            <div className="flex items-start gap-5 mb-6 pr-10">
              <div className="p-3.5 bg-[#FFFBFB] rounded-2xl border border-[#FEE2E2] shadow-sm">
                <Building2 size={28} className="text-[#800020]" />
              </div>
              <div className="flex-1 min-w-0">
                <h2 className="text-2xl md:text-3xl font-black text-[#1C1917] leading-tight mb-1.5 tracking-tight">
                  {job.title}
                </h2>
                <p className="text-[#800020] font-bold text-lg">{job.company}</p>
                {job.match_score !== undefined && (
                  <div className="mt-2 inline-flex items-center gap-2 px-3 py-1 bg-[#FFFBFB] border border-[#FEE2E2] rounded-full text-[#800020] text-sm font-black shadow-sm">
                    <CheckCircle2 size={14} />
                    {Math.round(job.match_score)}% Match
                  </div>
                )}
              </div>
            </div>

            {/* Meta Ribbon */}
            <div className="flex flex-wrap gap-3 mb-8 p-5 bg-slate-50/80 rounded-2xl border border-slate-200/60">
              <div className="flex items-center gap-2 text-sm font-bold text-[#57534E]">
                <MapPin size={15} className="text-[#800020]" /> {job.city || 'Pakistan'}
              </div>
              <div className="flex items-center gap-2 text-sm font-bold text-[#57534E]">
                {job.is_remote
                  ? <Globe size={15} className="text-emerald-600" />
                  : <Briefcase size={15} className="text-[#800020]" />}
                {job.is_remote ? 'Remote' : 'On-Site'}
              </div>
              {job.deadline && (
                <div className="flex items-center gap-2 text-sm font-black text-rose-700 bg-rose-50 px-3 py-1.5 rounded-xl border border-rose-200">
                  <CalendarDays size={15} className="text-rose-500" /> Apply by: {job.deadline}
                </div>
              )}
              {job.salary && (
                <div className="flex items-center gap-2 text-sm font-black text-emerald-700 bg-emerald-50 px-3 py-1.5 rounded-xl border border-emerald-200">
                  <DollarSign size={15} className="text-emerald-500" /> {job.salary}
                </div>
              )}
              <div className="flex items-center gap-1.5 text-[0.65rem] font-black text-[#A8A29E] uppercase tracking-widest ml-auto border-l border-slate-200 pl-4">
                <Link2 size={11} className="text-[#A8A29E]" />
                <span>Source: {job.source}</span>
              </div>
            </div>

            {/* Reputation */}
            {job.reputation && repConfig && (
              <div className={`flex gap-3 items-start p-4 rounded-2xl border mb-6 ${repConfig.bg} ${repConfig.border}`}>
                <div className="shrink-0 mt-0.5">
                  {job.reputation.status === 'green' && <ShieldCheck size={20} className="text-green-600" />}
                  {job.reputation.status === 'yellow' && <AlertTriangle size={20} className="text-amber-600" />}
                  {job.reputation.status === 'red' && <ShieldAlert size={20} className="text-red-600" />}
                </div>
                <div>
                  <p className={`text-xs font-black uppercase tracking-widest mb-1 ${repConfig.text}`}>
                    Reputation: {repConfig.label}
                  </p>
                  <p className={`text-sm font-medium leading-relaxed ${repConfig.text}`}>{job.reputation.summary}</p>
                </div>
              </div>
            )}

            {/* Requirements / Skills */}
            {job.skills_required && job.skills_required.length > 0 && (
              <div className="mb-8">
                <h3 className="text-xs font-black uppercase tracking-widest text-[#A8A29E] mb-3 flex items-center gap-2 px-1">
                  <Code size={13} className="text-[#800020]" /> Core Requirements
                </h3>
                <div className="flex flex-wrap gap-2">
                  {job.skills_required.map(s => (
                    <span key={s} className="badge badge-burgundy">{s}</span>
                  ))}
                </div>
              </div>
            )}

            {/* Match Analysis */}
            {(job.matching_skills?.length || job.missing_skills?.length) ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
                {job.matching_skills && job.matching_skills.length > 0 && (
                  <div className="bg-green-50 border border-green-200 rounded-2xl p-4">
                    <p className="text-[0.65rem] font-black uppercase tracking-widest text-green-700 mb-3 flex items-center gap-2">
                      <CheckCircle2 size={12} /> Your Matching Skills
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {job.matching_skills.map(s => (
                        <span key={s} className="badge badge-emerald flex items-center gap-1">
                          <CheckCircle2 size={10} /> {s}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                {job.missing_skills && job.missing_skills.length > 0 && (
                  <div className="bg-red-50 border border-red-200 rounded-2xl p-4">
                    <p className="text-[0.65rem] font-black uppercase tracking-widest text-red-700 mb-3 flex items-center gap-2">
                      <XCircle size={12} /> Skills to Develop
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {job.missing_skills.slice(0, 8).map(s => (
                        <span key={s} className="px-2.5 py-1 bg-red-100 border border-red-200 rounded-lg text-xs font-bold text-red-700">
                          {s}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : null}

            {/* Description */}
            <div className="mb-8">
              <h3 className="text-xs font-black uppercase tracking-widest text-[#A8A29E] mb-4 flex items-center gap-2 px-1">
                <Briefcase size={13} className="text-[#800020]" /> About the Role
              </h3>
              <div className="text-[#57534E] text-[0.925rem] leading-[1.75] bg-slate-50/60 p-6 rounded-2xl border border-slate-200/60 whitespace-pre-wrap font-medium">
                {job.description || `This role at ${job.company} is an exciting opportunity for growth. Click "View Posting" to read the full job description on ${job.source}.`}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-wrap gap-3 pt-6 border-t border-slate-100">
              <a
                href={job.url?.startsWith('http') ? job.url : `https://${job.url}`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex-1 flex items-center justify-center gap-2 px-6 py-3.5 bg-white hover:bg-[#FFFBFB] text-[#1C1917] border border-slate-200 hover:border-[#800020]/40 rounded-2xl font-black text-sm transition-all shadow-sm"
              >
                <ExternalLink size={16} className="text-[#800020]" />
                {job.source === 'linkedin' ? 'View on LinkedIn' : 'View Full Posting'}
              </a>

              <button
                onClick={() => onSave(job)}
                className={`flex items-center justify-center gap-2 px-5 py-3.5 rounded-2xl font-black text-sm border transition-all ${
                  isSaved
                    ? 'bg-[#FFFBFB] border-[#FEE2E2] text-[#800020] shadow-sm'
                    : 'bg-white border-slate-200 text-[#A8A29E] hover:bg-[#FFFBFB] hover:border-[#FEE2E2] hover:text-[#800020]'
                }`}
              >
                {isSaved ? <BookmarkCheck size={16} /> : <Bookmark size={16} />}
                {isSaved ? 'Saved' : 'Save Job'}
              </button>

              <button
                onClick={() => onApply(job)}
                className="btn-primary flex items-center justify-center gap-2 px-5 py-3.5 shadow-glow"
              >
                <Zap size={16} className="fill-white" /> Auto Apply
              </button>

              <button
                onClick={() => onDraftEmail(job)}
                className="flex items-center justify-center gap-2 px-5 py-3.5 bg-[#FFFBFB] hover:bg-white text-[#800020] border border-[#FEE2E2] rounded-2xl font-black text-sm transition-all"
              >
                <Mail size={16} /> Draft Email
              </button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
