"use client";

import { useState } from 'react';
import { X, Copy, Check, Send, Sparkles, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { internshipApi } from '@/lib/api';

export default function EmailModal({ 
  isOpen, 
  onClose, 
  job, 
  userSkills 
}: { 
  isOpen: boolean; 
  onClose: () => void; 
  job: any; 
  userSkills: string[] 
}) {
  const [draft, setDraft] = useState('');
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const generateDraft = async () => {
    setLoading(true);
    try {
      const res = await internshipApi.draftEmail({
        cv_skills: userSkills,
        job_title: job.title,
        company: job.company,
        job_description: job.description || job.title
      });
      setDraft(res.data.draft);
    } catch (e) {
      console.error(e);
      setDraft("Failed to generate draft. Please try again.");
    }
    setLoading(false);
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(draft);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xl">
          <motion.div 
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className="glass-card w-full max-w-2xl overflow-hidden flex flex-col max-h-[90vh] bg-white/90 border-white/60 shadow-2xl"
          >
            {/* Header */}
            <div className="p-8 border-b border-slate-100 flex justify-between items-center bg-slate-50/50 backdrop-blur-sm">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-2xl bg-[#FFFBFB] text-[#800020] border border-[#FEE2E2] shadow-sm"><Sparkles size={24}/></div>
                <div>
                  <h2 className="text-2xl font-black text-[#1C1917] tracking-tight">AI Outreach <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#800020] to-[#D4AF37]">Architect</span></h2>
                  <p className="text-[10px] text-[#A8A29E] font-bold uppercase tracking-widest mt-0.5">Hyper-personalized for {job.company}</p>
                </div>
              </div>
              <button onClick={onClose} className="p-2.5 hover:bg-slate-100 text-[#A8A29E] hover:text-[#1C1917] rounded-full transition-all"><X size={20}/></button>
            </div>

            {/* Content */}
            <div className="p-8 overflow-y-auto flex-1 flex flex-col gap-6 bg-white/60">
              {!draft && !loading ? (
                <div className="text-center py-16">
                  <div className="w-20 h-20 bg-[#FFFBFB] rounded-[2rem] flex items-center justify-center mx-auto mb-6 border border-[#FEE2E2] shadow-sm">
                    <Sparkles className="text-[#800020]" size={36} />
                  </div>
                  <h3 className="text-2xl font-black text-[#1C1917] mb-2 tracking-tight">Ready to Execute?</h3>
                  <p className="text-[#57534E] font-medium mb-8 max-w-sm mx-auto leading-relaxed">
                    Our AI will synthesize your CV and the job requirements into a high-conversion outreach sequence.
                  </p>
                  <button onClick={generateDraft} className="btn-primary px-10 py-4 shadow-glow mx-auto">
                    Synthesize Outreach
                  </button>
                </div>
              ) : (
                <div className="flex flex-col gap-5">
                  <div className="flex justify-between items-center px-1">
                    <label className="text-[10px] font-black text-[#800020] tracking-widest uppercase">Synthesized Sequence</label>
                    <button 
                      onClick={copyToClipboard}
                      className="flex items-center gap-1.5 text-xs font-bold text-[#A8A29E] hover:text-[#1C1917] transition-colors"
                    >
                      {copied ? <Check size={14} className="text-emerald-500" /> : <Copy size={14} />}
                      {copied ? 'Copied to Clipboard' : 'Copy Sequence'}
                    </button>
                  </div>
                  
                  {loading ? (
                <div className="bg-slate-50/50 rounded-3xl p-12 border border-slate-100 flex flex-col items-center justify-center gap-6 min-h-[400px] shadow-inner">
                  <div className="relative">
                    <div className="absolute inset-0 border-4 border-[#800020]/20 rounded-full"></div>
                    <div className="absolute inset-0 border-4 border-[#800020] border-t-transparent rounded-full animate-spin"></div>
                    <Loader2 className="animate-spin text-[#800020]" size={40} />
                  </div>
                  <div className="text-center">
                    <p className="text-lg font-black text-[#1C1917] tracking-tight">Generating Intelligence...</p>
                    <p className="text-sm text-[#A8A29E] font-medium mt-1">Cross-referencing your CV geometry with job specs</p>
                  </div>
                </div>
                  ) : (
                    <textarea 
                      className="bg-slate-50 border border-slate-200 rounded-3xl p-8 text-sm leading-relaxed text-[#57534E] focus:outline-none focus:border-[#800020]/40 focus:ring-4 focus:ring-[#800020]/5 min-h-[400px] resize-none transition-all font-medium shadow-inner"
                      value={draft}
                      onChange={(e) => setDraft(e.target.value)}
                    />
                  )}
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="p-8 border-t border-slate-100 bg-slate-50/50 backdrop-blur-sm flex justify-between items-center">
               <button onClick={onClose} className="text-sm font-bold text-[#A8A29E] hover:text-[#1C1917] transition-colors px-4 py-2">
                 Discard Draft
               </button>
               {draft && (
                 <button className="btn-primary px-8 py-3.5 shadow-glow flex items-center gap-2">
                   <Send size={18} className="fill-white" />
                   <span className="font-black">Execute Transmission</span>
                 </button>
               )}
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
