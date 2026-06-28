"use client";

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Zap, CheckCircle2, XCircle, Clock, AlertTriangle,
  Building2, Send, ChevronDown, ChevronUp, Activity,
  RotateCcw, Loader2, Pause, Play, StopCircle,
  Linkedin, ClipboardList, Eye, EyeOff, ExternalLink,
  Info
} from 'lucide-react';
import { internshipApi } from '@/lib/api';

interface Application {
  id: number;
  internship_id: number;
  status: string;
  applied_at: string;
  company: string;
  job_title: string;
  notes?: string;
  apply_url?: string;
  questions_pending?: string[];
  is_cancelled?: boolean;
  is_paused?: boolean;
}

interface AgentConsoleProps {
  applications: Application[];
  onSubmitAnswers: (appId: number, answers: Record<string, string>) => Promise<void>;
  onRefresh: () => void;
}

// ─── Status configuration ────────────────────────────────────────────
const STATUS_CONFIG: Record<string, {
  label: string; color: string; bg: string; border: string;
  icon: React.ReactNode; pulse: boolean;
}> = {
  Applying: {
    label: 'Agent Working…', color: 'text-[#800020]',
    bg: 'bg-[#FFFBFB]', border: 'border-[#FEE2E2]',
    icon: <Loader2 size={14} className="animate-spin text-[#800020]" />, pulse: true,
  },
  Pending: {
    label: 'Queued', color: 'text-slate-500',
    bg: 'bg-slate-50', border: 'border-slate-200',
    icon: <Clock size={14} className="text-slate-400" />, pulse: false,
  },
  'Awaiting You': {
    label: 'Action Required', color: 'text-amber-700',
    bg: 'bg-amber-50', border: 'border-amber-300',
    icon: <AlertTriangle size={14} className="text-amber-600" />, pulse: true,
  },
  'Needs LinkedIn': {
    label: 'LinkedIn Needed', color: 'text-blue-700',
    bg: 'bg-blue-50', border: 'border-blue-300',
    icon: <Linkedin size={14} className="text-blue-600" />, pulse: true,
  },
  'Manual Required': {
    label: 'Apply Manually', color: 'text-orange-700',
    bg: 'bg-orange-50', border: 'border-orange-200',
    icon: <ClipboardList size={14} className="text-orange-600" />, pulse: false,
  },
  Paused: {
    label: 'Paused', color: 'text-slate-600',
    bg: 'bg-slate-100', border: 'border-slate-300',
    icon: <Pause size={14} className="text-slate-500" />, pulse: false,
  },
  Applied: {
    label: 'Applied ✓', color: 'text-emerald-700',
    bg: 'bg-emerald-50', border: 'border-emerald-200',
    icon: <CheckCircle2 size={14} className="text-emerald-600" />, pulse: false,
  },
  Failed: {
    label: 'Failed', color: 'text-red-700',
    bg: 'bg-red-50', border: 'border-red-200',
    icon: <XCircle size={14} className="text-red-500" />, pulse: false,
  },
  Cancelled: {
    label: 'Cancelled', color: 'text-slate-500',
    bg: 'bg-slate-50', border: 'border-slate-200',
    icon: <StopCircle size={14} className="text-slate-400" />, pulse: false,
  },
};

const TIMELINE_STEPS = [
  { key: 'queued', label: 'Queued' },
  { key: 'agent',  label: 'Agent Start' },
  { key: 'form',   label: 'Form Fill' },
  { key: 'submit', label: 'Submit' },
  { key: 'done',   label: 'Complete' },
];

function getTimelineIndex(status: string, notes?: string) {
  const log = (notes || '').toLowerCase();
  switch (status) {
    case 'Pending':          return 0;
    case 'Applying':
      // Infer sub-step from agent log text
      if (log.includes('submit') || log.includes('sending') || log.includes('clicking submit')) return 3;
      if (log.includes('fill') || log.includes('typing') || log.includes('field') || log.includes('form')) return 2;
      return 1; // default: Agent Start
    case 'Awaiting You':     return 3;
    case 'Needs LinkedIn':   return 1;
    case 'Manual Required':  return 3;
    case 'Paused':           return 2;
    case 'Applied':          return 4;
    case 'Failed':           return 4;
    case 'Cancelled':        return 1;
    default:                 return 0;
  }
}

// SQLite returns datetime without 'Z', so JS would parse as local time.
// Append 'Z' to force UTC interpretation.
function normalizeUtc(isoString: string): string {
  if (!isoString) return isoString;
  const s = isoString.replace(' ', 'T');
  return s.endsWith('Z') || s.includes('+') ? s : s + 'Z';
}

function useRelativeTime(isoString: string) {
  const [rel, setRel] = useState('');
  useEffect(() => {
    function update() {
      const diff = Math.floor((Date.now() - new Date(normalizeUtc(isoString)).getTime()) / 1000);
      if (diff < 60) setRel(`${diff}s ago`);
      else if (diff < 3600) setRel(`${Math.floor(diff / 60)}m ago`);
      else if (diff < 86400) setRel(`${Math.floor(diff / 3600)}h ${Math.floor((diff % 3600) / 60)}m ago`);
      else setRel(new Date(normalizeUtc(isoString)).toLocaleDateString());
    }
    update();
    const t = setInterval(update, 10000);
    return () => clearInterval(t);
  }, [isoString]);
  return rel;
}

function ElapsedTimer({ isoString }: { isoString: string }) {
  const [elapsed, setElapsed] = useState('');
  useEffect(() => {
    function update() {
      const diff = Math.floor((Date.now() - new Date(normalizeUtc(isoString)).getTime()) / 1000);
      const h = Math.floor(diff / 3600);
      const m = Math.floor((diff % 3600) / 60);
      const s = diff % 60;
      setElapsed(h > 0 ? `${h}h ${m}m ${s}s` : m > 0 ? `${m}m ${s}s` : `${s}s`);
    }
    update();
    const t = setInterval(update, 1000);
    return () => clearInterval(t);
  }, [isoString]);
  return <span className="font-mono">{elapsed}</span>;
}

// ─── Inline Q&A Form ─────────────────────────────────────────────────
function InlineQAForm({ app, onSubmit }: {
  app: Application;
  onSubmit: (answers: Record<string, string>) => Promise<void>;
}) {
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);
  const questions = app.questions_pending || [];
  const allAnswered = questions.every(q => answers[q]?.trim());

  return (
    <div className="mt-4 flex flex-col gap-4">
      {questions.map((q, i) => (
        <div key={i}>
          <label className="block text-[10px] font-black text-[#800020] uppercase tracking-widest mb-1.5 px-1">
            Q{i + 1}: {q}
          </label>
          <textarea
            className="w-full bg-white border border-slate-200 focus:border-[#800020]/40 focus:ring-4 focus:ring-[#800020]/5 rounded-2xl p-4 text-sm text-[#57534E] outline-none resize-none min-h-[90px] font-medium placeholder:text-slate-300 transition-all"
            placeholder="Your personal response…"
            value={answers[q] || ''}
            onChange={e => setAnswers(prev => ({ ...prev, [q]: e.target.value }))}
          />
        </div>
      ))}
      <button
        onClick={async () => { setSubmitting(true); await onSubmit(answers); setSubmitting(false); }}
        disabled={!allAnswered || submitting}
        className="btn-primary flex items-center gap-2 self-start px-8 py-3 shadow-glow disabled:opacity-40"
      >
        {submitting ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
        <span className="font-black">{submitting ? 'Resuming Agent…' : 'Submit & Resume Agent'}</span>
      </button>
    </div>
  );
}

// ─── LinkedIn Credentials Form ────────────────────────────────────────
function LinkedInCredForm({ app, onDone }: {
  app: Application;
  onDone: () => void;
}) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPw, setShowPw] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async () => {
    if (!email.trim() || !password.trim()) { setError('Both fields are required.'); return; }
    setSubmitting(true); setError('');
    try {
      await internshipApi.setLinkedInCredentials(app.id, {
        linkedin_email: email.trim(),
        linkedin_password: password.trim(),
      });
      onDone();
    } catch (e: any) {
      setError(e?.response?.data?.detail || 'Failed to save credentials.');
    } finally { setSubmitting(false); }
  };

  return (
    <div className="mt-4 flex flex-col gap-4">
      <div className="flex items-start gap-2 p-3 bg-blue-50 border border-blue-200 rounded-xl text-xs text-blue-700 font-medium">
        <Info size={14} className="shrink-0 mt-0.5" />
        Your credentials are stored locally on your machine only and are never transmitted externally.
      </div>

      <div>
        <label className="block text-[10px] font-black text-blue-700 uppercase tracking-widest mb-1.5 px-1">
          LinkedIn Email
        </label>
        <input
          type="email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          placeholder="you@email.com"
          className="w-full bg-white border border-slate-200 focus:border-blue-400 focus:ring-4 focus:ring-blue-500/10 rounded-2xl px-4 py-3 text-sm text-[#57534E] font-medium outline-none transition-all"
        />
      </div>

      <div>
        <label className="block text-[10px] font-black text-blue-700 uppercase tracking-widest mb-1.5 px-1">
          LinkedIn Password
        </label>
        <div className="relative">
          <input
            type={showPw ? 'text' : 'password'}
            value={password}
            onChange={e => setPassword(e.target.value)}
            placeholder="••••••••"
            className="w-full bg-white border border-slate-200 focus:border-blue-400 focus:ring-4 focus:ring-blue-500/10 rounded-2xl px-4 py-3 pr-12 text-sm text-[#57534E] font-medium outline-none transition-all"
          />
          <button
            type="button"
            onClick={() => setShowPw(v => !v)}
            className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 transition-colors"
          >
            {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
          </button>
        </div>
      </div>

      {error && <p className="text-xs text-red-600 font-bold px-1">{error}</p>}

      <button
        onClick={handleSubmit}
        disabled={submitting}
        className="flex items-center gap-2 self-start px-8 py-3 rounded-2xl text-sm font-black text-white bg-[#0A66C2] hover:bg-[#004182] transition-all shadow-md disabled:opacity-40"
      >
        {submitting ? <Loader2 size={16} className="animate-spin" /> : <Linkedin size={16} />}
        {submitting ? 'Saving & Resuming…' : 'Save & Resume Easy Apply'}
      </button>
    </div>
  );
}

// ─── Manual Instructions Panel ────────────────────────────────────────
function ManualInstructionsPanel({ app }: { app: Application }) {
  const steps = (app.notes || '').split('\n').filter(Boolean);
  return (
    <div className="mt-4 flex flex-col gap-4">
      <div className="bg-orange-50 border border-orange-200 rounded-2xl p-5">
        <p className="text-[10px] font-black text-orange-700 uppercase tracking-widest mb-3">
          Manual Application Guide
        </p>
        <div className="flex flex-col gap-2">
          {steps.map((step, i) => (
            <div key={i} className="flex items-start gap-2 text-sm text-orange-800 font-medium">
              <span className="shrink-0 text-orange-500">›</span>
              <span>{step}</span>
            </div>
          ))}
        </div>
      </div>
      {app.apply_url && (
        <a
          href={app.apply_url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 self-start px-6 py-3 rounded-2xl text-sm font-black text-white bg-orange-600 hover:bg-orange-700 transition-all shadow-md"
        >
          <ExternalLink size={16} /> Open Application Page
        </a>
      )}
    </div>
  );
}

// ─── Application Row ──────────────────────────────────────────────────
function ApplicationRow({ app, onSubmit, onRefresh }: {
  app: Application;
  onSubmit: (appId: number, answers: Record<string, string>) => Promise<void>;
  onRefresh: () => void;
}) {
  const [expanded, setExpanded] = useState(
    ['Awaiting You', 'Needs LinkedIn', 'Manual Required', 'Paused'].includes(app.status)
  );
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  const cfg = STATUS_CONFIG[app.status] || STATUS_CONFIG['Pending'];
  const tlIdx = getTimelineIndex(app.status, app.notes);
  const relTime = useRelativeTime(app.applied_at);
  const isActive = ['Applying', 'Pending'].includes(app.status);
  const isFailed = ['Failed', 'Cancelled'].includes(app.status);
  const canPause = ['Applying', 'Pending'].includes(app.status);
  const canResume = app.status === 'Paused';
  const canCancel = !['Applied', 'Cancelled', 'Failed'].includes(app.status);

  const doAction = async (label: string, fn: () => Promise<any>) => {
    setActionLoading(label);
    try { await fn(); await new Promise(r => setTimeout(r, 400)); onRefresh(); }
    catch (e: any) { alert(e?.response?.data?.detail || `${label} failed.`); }
    finally { setActionLoading(null); }
  };

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`glass-card overflow-hidden border ${
        app.status === 'Awaiting You'   ? 'border-amber-300 bg-amber-50/40' :
        app.status === 'Needs LinkedIn' ? 'border-blue-300 bg-blue-50/30' :
        app.status === 'Manual Required'? 'border-orange-200 bg-orange-50/30' :
        app.status === 'Paused'         ? 'border-slate-300 bg-slate-50/40' :
        'bg-white/60 border-white/60'
      }`}
    >
      {/* Row Header */}
      <div
        role="button"
        tabIndex={0}
        className="w-full flex items-center gap-4 p-5 text-left hover:bg-slate-50/50 transition-colors cursor-pointer outline-none focus-visible:ring-2 focus-visible:ring-[#800020]/20"
        onClick={() => setExpanded(e => !e)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === 'Space') {
            e.preventDefault();
            setExpanded(prev => !prev);
          }
        }}
      >
        <div className={`p-2.5 rounded-xl border ${cfg.bg} ${cfg.border} shrink-0 ${cfg.pulse ? 'animate-pulse' : ''}`}>
          {cfg.icon}
        </div>
        <div className="flex-1 min-w-0">
          <p className="font-black text-[#1C1917] text-sm leading-tight truncate">{app.job_title}</p>
          <p className="text-[#A8A29E] font-bold text-xs mt-0.5 flex items-center gap-1">
            <Building2 size={10} /> {app.company}
          </p>
        </div>

        {/* Pause / Stop controls inline in header */}
        {(canPause || canResume || canCancel) && (
          <div className="flex items-center gap-1.5 shrink-0" onClick={e => e.stopPropagation()} onKeyDown={e => e.stopPropagation()}>
            {canPause && (
              <button
                title="Pause agent"
                disabled={!!actionLoading}
                onClick={() => doAction('Pause', () => internshipApi.pauseApplication(app.id))}
                className="p-2 rounded-xl bg-slate-100 hover:bg-amber-100 hover:text-amber-700 text-slate-500 border border-slate-200 hover:border-amber-300 transition-all text-xs font-bold"
              >
                {actionLoading === 'Pause' ? <Loader2 size={13} className="animate-spin" /> : <Pause size={13} />}
              </button>
            )}
            {canResume && (
              <button
                title="Resume agent"
                disabled={!!actionLoading}
                onClick={() => doAction('Resume', () => internshipApi.resumeApplication(app.id))}
                className="p-2 rounded-xl bg-emerald-50 hover:bg-emerald-100 text-emerald-600 border border-emerald-200 transition-all"
              >
                {actionLoading === 'Resume' ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}
              </button>
            )}
            {canCancel && (
              <button
                title="Cancel application"
                disabled={!!actionLoading}
                onClick={() => {
                  if (confirm('Cancel this application?')) {
                    doAction('Cancel', () => internshipApi.cancelApplication(app.id));
                  }
                }}
                className="p-2 rounded-xl bg-red-50 hover:bg-red-100 text-red-500 border border-red-200 transition-all"
              >
                {actionLoading === 'Cancel' ? <Loader2 size={13} className="animate-spin" /> : <StopCircle size={13} />}
              </button>
            )}
          </div>
        )}

        <span className={`badge ${cfg.bg} ${cfg.border} ${cfg.color} border text-[10px] font-black shrink-0`}>
          {cfg.label}
        </span>
        <div className="text-slate-300 shrink-0">
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </div>

      {/* Expanded Panel */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-6 border-t border-slate-100/80">
              {/* Timeline */}
              <div className="flex items-center gap-1 py-5 overflow-x-auto custom-scrollbar">
                {TIMELINE_STEPS.map((step, i) => {
                  const done = i < tlIdx || (i === tlIdx && app.status === 'Applied');
                  const active = i === tlIdx && app.status !== 'Applied';
                  const failed = isFailed && i === tlIdx;
                  return (
                    <div key={step.key} className="flex items-center gap-1 shrink-0">
                      <div className="flex flex-col items-center gap-1.5">
                        <div className={`w-7 h-7 rounded-full flex items-center justify-center border-2 transition-all ${
                          failed  ? 'bg-red-100 border-red-400' :
                          done    ? 'bg-[#800020] border-[#800020]' :
                          active  ? 'bg-[#FFFBFB] border-[#800020] animate-pulse' :
                                    'bg-slate-100 border-slate-200'
                        }`}>
                          {failed ? <XCircle size={14} className="text-red-500" /> :
                           done   ? <CheckCircle2 size={14} className="text-white" /> :
                           active ? <div className="w-2.5 h-2.5 rounded-full bg-[#800020]" /> :
                                    <div className="w-2 h-2 rounded-full bg-slate-300" />}
                        </div>
                        <span className={`text-[9px] font-black uppercase tracking-wider whitespace-nowrap ${
                          done || active ? 'text-[#800020]' : 'text-slate-400'
                        }`}>{step.label}</span>
                      </div>
                      {i < TIMELINE_STEPS.length - 1 && (
                        <div className={`w-8 h-0.5 mb-5 shrink-0 ${i < tlIdx ? 'bg-[#800020]' : 'bg-slate-200'}`} />
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Agent log */}
              {app.notes && app.status !== 'Manual Required' && (
                <div className="bg-slate-50 border border-slate-200 rounded-2xl p-4 mb-4">
                  <p className="text-[10px] font-black text-[#A8A29E] uppercase tracking-widest mb-1.5">Agent Log</p>
                  <p className="text-sm text-[#57534E] font-medium italic">{app.notes}</p>
                </div>
              )}

              <div className="flex items-center justify-between mb-4">
                <p className="text-[10px] font-bold text-[#A8A29E] uppercase tracking-wider">
                  Initiated: {relTime}
                </p>
                {isActive && (
                  <p className="text-[10px] font-bold text-[#800020] uppercase tracking-wider flex items-center gap-1">
                    <Loader2 size={10} className="animate-spin" />
                    Running: <ElapsedTimer isoString={app.applied_at} />
                  </p>
                )}
              </div>

              {/* ── Awaiting human Q&A ── */}
              {app.status === 'Awaiting You' && app.questions_pending && app.questions_pending.length > 0 && (
                <div className="bg-amber-50 border border-amber-200 rounded-2xl p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <AlertTriangle size={18} className="text-amber-600 shrink-0" />
                    <div>
                      <p className="font-black text-amber-800 text-sm">Agent Paused — Your Input Required</p>
                      <p className="text-amber-700 text-xs font-medium mt-0.5">
                        The agent found questions that need your personal perspective. Answer below to resume.
                      </p>
                    </div>
                  </div>
                  <InlineQAForm app={app} onSubmit={(answers) => onSubmit(app.id, answers)} />
                </div>
              )}

              {/* ── Needs LinkedIn credentials ── */}
              {app.status === 'Needs LinkedIn' && (
                <div className="bg-blue-50 border border-blue-200 rounded-2xl p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <Linkedin size={18} className="text-[#0A66C2] shrink-0" />
                    <div>
                      <p className="font-black text-blue-800 text-sm">LinkedIn Credentials Required</p>
                      <p className="text-blue-700 text-xs font-medium mt-0.5">
                        This job uses LinkedIn Easy Apply. Enter your credentials to let the agent apply for you.
                      </p>
                    </div>
                  </div>
                  <LinkedInCredForm app={app} onDone={onRefresh} />
                </div>
              )}

              {/* ── Manual Required ── */}
              {app.status === 'Manual Required' && (
                <div className="bg-orange-50 border border-orange-200 rounded-2xl p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <ClipboardList size={18} className="text-orange-600 shrink-0" />
                    <div>
                      <p className="font-black text-orange-800 text-sm">Manual Application Required</p>
                      <p className="text-orange-700 text-xs font-medium mt-0.5">
                        The agent couldn&apos;t auto-submit this one. Follow the steps below to apply.
                      </p>
                    </div>
                  </div>
                  <ManualInstructionsPanel app={app} />
                </div>
              )}

              {/* ── Paused ── */}
              {app.status === 'Paused' && (
                <div className="flex items-center gap-3 p-4 bg-slate-100 border border-slate-200 rounded-2xl">
                  <Pause size={18} className="text-slate-500 shrink-0" />
                  <div className="flex-1">
                    <p className="font-black text-slate-700 text-sm">Agent is Paused</p>
                    <p className="text-slate-500 text-xs font-medium mt-0.5">
                      Click Resume above to continue the application, or Cancel to stop it entirely.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ─── Main Console ─────────────────────────────────────────────────────
export default function AgentConsole({ applications, onSubmitAnswers, onRefresh }: AgentConsoleProps) {
  const active    = applications.filter(a => ['Applying', 'Pending', 'Awaiting You', 'Needs LinkedIn', 'Paused'].includes(a.status));
  const attention = applications.filter(a => ['Awaiting You', 'Needs LinkedIn', 'Manual Required'].includes(a.status));
  const manual    = applications.filter(a => a.status === 'Manual Required');
  const completed = applications.filter(a => ['Applied', 'Failed', 'Cancelled'].includes(a.status));

  return (
    <div className="flex flex-col gap-8">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-4xl font-black text-[#1C1917] tracking-tight">
            Agent <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#800020] to-[#D4AF37]">Console</span>
          </h2>
          <p className="text-slate-500 font-medium mt-1">Real-time view of your autonomous application agent.</p>
        </div>
        <button onClick={onRefresh} className="btn-secondary flex items-center gap-2 text-xs py-2 px-4">
          <RotateCcw size={14} /> Refresh
        </button>
      </div>

      {/* Stats strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {[
          { label: 'Active',          count: active.length,    color: 'text-[#800020]',   bg: 'bg-[#FFFBFB] border-[#FEE2E2]' },
          { label: 'Needs Attention', count: attention.length, color: 'text-amber-700',   bg: 'bg-amber-50 border-amber-200' },
          { label: 'Manual',          count: manual.length,    color: 'text-orange-700',  bg: 'bg-orange-50 border-orange-200' },
          { label: 'Completed',       count: completed.length, color: 'text-emerald-700', bg: 'bg-emerald-50 border-emerald-200' },
        ].map(s => (
          <div key={s.label} className={`glass-card p-4 border ${s.bg} flex flex-col gap-1`}>
            <p className={`text-3xl font-black ${s.color}`}>{s.count}</p>
            <p className="text-xs font-bold text-[#A8A29E] uppercase tracking-widest">{s.label}</p>
          </div>
        ))}
      </div>

      {/* Attention banner */}
      {attention.length > 0 && (
        <div className="flex items-center gap-3 p-4 bg-amber-50 border border-amber-300 rounded-2xl shadow-sm">
          <div className="p-2 bg-amber-100 rounded-xl animate-pulse">
            <Zap size={20} className="fill-amber-600 text-amber-600" />
          </div>
          <div>
            <p className="font-black text-amber-800 text-sm">
              {attention.length} application{attention.length > 1 ? 's' : ''} need{attention.length === 1 ? 's' : ''} your attention
            </p>
            <p className="text-amber-700 text-xs font-medium">
              Expand the application below to respond and resume the agent.
            </p>
          </div>
        </div>
      )}

      {/* Active Applications */}
      {active.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-4">
            <Activity size={16} className="text-[#800020]" />
            <h3 className="text-xs font-black uppercase tracking-widest text-[#A8A29E]">Active Agent Feed</h3>
          </div>
          <div className="flex flex-col gap-3">
            {active.map(app => (
              <ApplicationRow key={app.id} app={app} onSubmit={onSubmitAnswers} onRefresh={onRefresh} />
            ))}
          </div>
        </div>
      )}

      {/* Manual Required (separate section) */}
      {manual.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-4">
            <ClipboardList size={16} className="text-orange-600" />
            <h3 className="text-xs font-black uppercase tracking-widest text-[#A8A29E]">Manual Applications</h3>
          </div>
          <div className="flex flex-col gap-3">
            {manual.map(app => (
              <ApplicationRow key={app.id} app={app} onSubmit={onSubmitAnswers} onRefresh={onRefresh} />
            ))}
          </div>
        </div>
      )}

      {/* Completed */}
      {completed.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-4">
            <CheckCircle2 size={16} className="text-emerald-600" />
            <h3 className="text-xs font-black uppercase tracking-widest text-[#A8A29E]">Completed Applications</h3>
          </div>
          <div className="flex flex-col gap-3">
            {completed.map(app => (
              <ApplicationRow key={app.id} app={app} onSubmit={onSubmitAnswers} onRefresh={onRefresh} />
            ))}
          </div>
        </div>
      )}

      {/* Empty state */}
      {applications.length === 0 && (
        <div className="py-28 glass-card flex flex-col items-center justify-center text-center opacity-40">
          <Zap size={48} className="mb-4" />
          <p className="text-lg font-bold">No applications yet.</p>
          <p className="text-sm">Hit Auto Apply on any internship card to start the agent.</p>
        </div>
      )}
    </div>
  );
}
