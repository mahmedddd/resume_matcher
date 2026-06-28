"""
Patch script to add the Q&A modal and real-time polling to the History tab in page.tsx.
Also adds the Application interface update for the new fields.
"""
import re

with open('src/app/page.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# 1. Update the Application interface to include notes and questions_pending
old_interface = (
    'interface Application {\n'
    '  id: number;\n'
    '  internship_id: number;\n'
    '  status: string;\n'
    '  applied_at: string;\n'
    '  company: string;\n'
    '  job_title: string;\n'
    '}'
)
new_interface = (
    'interface Application {\n'
    '  id: number;\n'
    '  internship_id: number;\n'
    '  status: string;\n'
    '  applied_at: string;\n'
    '  company: string;\n'
    '  job_title: string;\n'
    '  notes?: string;\n'
    '  questions_pending?: string[];\n'
    '}'
)
if old_interface in content:
    content = content.replace(old_interface, new_interface)
    changes += 1
    print('✅ Application interface: UPDATED')
else:
    print('❌ Application interface: NOT FOUND')

# 2. Add qaModal state variables near the top of component
old_state = '  const [savedJobs, setSavedJobs] = useState<Internship[]>(() => {'
new_state = (
    '  const [qaModal, setQaModal] = useState<{ appId: number; questions: string[] } | null>(null);\n'
    '  const [qaAnswers, setQaAnswers] = useState<Record<string, string>>({});\n'
    '  const [savedJobs, setSavedJobs] = useState<Internship[]>(() => {'
)
if old_state in content:
    content = content.replace(old_state, new_state, 1)
    changes += 1
    print('✅ Q&A state added')
else:
    print('❌ Q&A state: NOT FOUND')

# 3. Add real-time polling useEffect after the existing savedJobs effect (near line 76)
old_effect = (
    '  // Persist saved jobs to localStorage whenever they change\n'
    '  useEffect(() => {\n'
    '    localStorage.setItem(\'savedJobs\', JSON.stringify(savedJobs));\n'
    '  }, [savedJobs]);'
)
new_effect = (
    '  // Persist saved jobs to localStorage whenever they change\n'
    '  useEffect(() => {\n'
    '    localStorage.setItem(\'savedJobs\', JSON.stringify(savedJobs));\n'
    '  }, [savedJobs]);\n\n'
    '  // Real-time polling for active applications\n'
    '  useEffect(() => {\n'
    '    if (activeTab !== \'history\') return;\n'
    '    const hasActive = applications.some(a => a.status === \'Applying\' || a.status === \'Pending\');\n'
    '    if (!hasActive) return;\n'
    '    const poll = setInterval(() => { fetchApplications(); }, 10000);\n'
    '    return () => clearInterval(poll);\n'
    '  }, [activeTab, applications]);'
)
if old_effect in content:
    content = content.replace(old_effect, new_effect, 1)
    changes += 1
    print('✅ Polling useEffect added')
else:
    print('❌ Polling useEffect: NOT FOUND')

# 4. Add submitAnswers handler after handleAutoApply
old_handler = '  const handleScrape = async () => {'
new_handler = (
    '  const handleSubmitAnswers = async () => {\n'
    '    if (!qaModal) return;\n'
    '    try {\n'
    '      await internshipApi.submitAnswers(qaModal.appId, qaAnswers);\n'
    '      setQaModal(null);\n'
    '      setQaAnswers({});\n'
    '      fetchApplications();\n'
    '    } catch (e) {\n'
    '      console.error(\"Submit answers failed:\", e);\n'
    '      alert(\"Failed to submit answers. Please try again.\");\n'
    '    }\n'
    '  };\n\n'
    '  const handleScrape = async () => {'
)
if old_handler in content:
    content = content.replace(old_handler, new_handler, 1)
    changes += 1
    print('✅ submitAnswers handler added')
else:
    print('❌ submitAnswers handler: NOT FOUND')

# 5. Add Q&A modal before the closing </main> tag
old_modal_area = (
    '      {/* AI Outreach Modal */}\n'
    '      {selectedJobForEmail && ('
)
new_modal_area = (
    '      {/* Human-in-Loop Q&A Modal */}\n'
    '      {qaModal && (\n'
    '        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-md p-4">\n'
    '          <div className="bg-[#0d1526] border border-amber-500/30 rounded-3xl p-8 max-w-2xl w-full shadow-[0_0_60px_rgba(245,158,11,0.15)] relative overflow-hidden">\n'
    '            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-amber-500/0 via-amber-500 to-amber-500/0"></div>\n'
    '            <div className="flex items-start gap-4 mb-6">\n'
    '              <div className="p-3 bg-amber-500/20 rounded-2xl border border-amber-500/30 text-amber-400 shrink-0">\n'
    '                <Zap size={24} className="fill-amber-400" />\n'
    '              </div>\n'
    '              <div>\n'
    '                <h3 className="text-xl font-black text-white mb-1">Your Input Required</h3>\n'
    '                <p className="text-amber-300/80 text-sm">The agent found open-ended questions it cannot answer for you. Please write your own responses — these questions require your personal voice.</p>\n'
    '              </div>\n'
    '            </div>\n'
    '            <div className="flex flex-col gap-5 max-h-80 overflow-y-auto pr-1 custom-scrollbar">\n'
    '              {qaModal.questions.map((q, i) => (\n'
    '                <div key={i}>\n'
    '                  <label className="block text-xs font-black text-amber-300 uppercase tracking-widest mb-2">Q{i+1}: {q}</label>\n'
    '                  <textarea\n'
    '                    className="w-full bg-white/5 border border-white/10 rounded-xl p-4 text-sm text-slate-200 focus:border-amber-400 outline-none resize-none min-h-[100px] custom-scrollbar"\n'
    '                    placeholder="Write your personal answer here..."\n'
    '                    value={qaAnswers[q] || ""}\n'
    '                    onChange={(e) => setQaAnswers(prev => ({ ...prev, [q]: e.target.value }))}\n'
    '                  />\n'
    '                </div>\n'
    '              ))}\n'
    '            </div>\n'
    '            <div className="flex justify-between mt-6">\n'
    '              <button onClick={() => { setQaModal(null); setQaAnswers({}); }} className="px-6 py-3 text-sm font-bold text-slate-400 hover:text-white border border-white/10 rounded-xl hover:bg-white/5 transition-all">Cancel</button>\n'
    '              <button\n'
    '                onClick={handleSubmitAnswers}\n'
    '                disabled={qaModal.questions.some(q => !qaAnswers[q]?.trim())}\n'
    '                className="px-8 py-3 bg-gradient-to-r from-amber-500 to-orange-500 text-white font-black rounded-xl text-sm shadow-[0_0_20px_rgba(245,158,11,0.4)] hover:shadow-[0_0_30px_rgba(245,158,11,0.6)] transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"\n'
    '              >\n'
    '                <Zap size={16} className="fill-white" /> Submit & Resume Agent\n'
    '              </button>\n'
    '            </div>\n'
    '          </div>\n'
    '        </div>\n'
    '      )}\n\n'
    '      {/* AI Outreach Modal */}\n'
    '      {selectedJobForEmail && ('
)
if old_modal_area in content:
    content = content.replace(old_modal_area, new_modal_area, 1)
    changes += 1
    print('✅ Q&A modal added')
else:
    print('❌ Q&A modal area: NOT FOUND')

# 6. Update application status card to show "Awaiting You" button
old_status = (
    "                          <div className={`badge ${app.status === 'Applied' ? 'badge-blue' : app.status === 'Rejected' ? 'badge-red' : 'badge-green'} px-4 py-1.5 shadow-lg`}>\n"
    '                            {app.status.toUpperCase()}\n'
    '                          </div>'
)
new_status = (
    "                          <div className={`badge px-4 py-1.5 shadow-lg ${\n"
    "                            app.status === 'Applied' ? 'badge-blue' :\n"
    "                            app.status === 'Failed' ? 'badge-red' :\n"
    "                            app.status === 'Awaiting You' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' :\n"
    "                            app.status === 'Applying' ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40 animate-pulse' :\n"
    "                            'badge-green'\n"
    "                          }`}>\n"
    '                            {app.status === \'Awaiting You\' ? (\n'
    '                              <button onClick={() => setQaModal({ appId: app.id, questions: app.questions_pending || [] })} className="flex items-center gap-1">\n'
    '                                <Zap size={11} className="fill-amber-400" /> AWAITING YOUR INPUT\n'
    '                              </button>\n'
    '                            ) : app.status.toUpperCase()}\n'
    '                          </div>'
)
if old_status in content:
    content = content.replace(old_status, new_status, 1)
    changes += 1
    print('✅ Application status badge updated')
else:
    print('❌ Application status badge: NOT FOUND')

with open('src/app/page.tsx', 'w', encoding='utf-8') as f:
    f.write(content)

print(f'\n✅ page.tsx saved — {changes}/6 patches applied.')
