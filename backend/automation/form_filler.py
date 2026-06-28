"""
Autonomous Form Filler - v2
Handles: standard forms, LinkedIn Easy Apply, dropdowns/radios,
         human Q&A pause, pause/cancel signals, manual fallback instructions.
"""
import asyncio
import json
import re
from typing import Optional

from playwright.async_api import async_playwright, Page, BrowserContext
import google.generativeai as genai
from config import settings

# Questions that require a human's personal perspective (not auto-fillable)
HUMAN_REQUIRED_KEYWORDS = [
    "why do you want", "tell us about yourself", "describe yourself",
    "why are you interested", "what makes you", "your strengths",
    "your weaknesses", "your goals", "cover letter", "motivation letter",
    "what excites you", "why us", "relevant experience", "how would you",
    "what experience", "describe a time", "why apply", "career goals",
    "long-term goal", "short-term goal", "anything else", "additional info",
    "supporting statement", "what can you bring", "personal statement",
]

LINKEDIN_DOMAINS = ["linkedin.com", "www.linkedin.com"]


def _is_linkedin(url: str) -> bool:
    return any(d in url for d in LINKEDIN_DOMAINS)


class AgentCancelledError(Exception):
    pass


class AgentPausedError(Exception):
    pass


class FormFiller:
    def __init__(self, profile_data: dict, app_id: int, db_check_fn=None):
        """
        profile_data: full_name, email, phone, linkedin_url, linkedin_email,
                      linkedin_password, github_url, portfolio_url,
                      human_answers (optional)
        app_id:       used for pause/cancel polling
        db_check_fn:  async callable(app_id) -> (is_paused, is_cancelled)
        """
        self.profile = dict(profile_data)
        self.human_answers: dict = self.profile.pop("human_answers", {}) or {}
        self.app_id = app_id
        self.db_check_fn = db_check_fn
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")

    # ------------------------------------------------------------------
    # Pause / cancel polling
    # ------------------------------------------------------------------
    async def _check_signals(self):
        """Raises AgentCancelledError or AgentPausedError if flagged in DB."""
        if not self.db_check_fn:
            return
        try:
            is_paused, is_cancelled = await self.db_check_fn(self.app_id)
            if is_cancelled:
                raise AgentCancelledError()
            if is_paused:
                raise AgentPausedError()
        except (AgentCancelledError, AgentPausedError):
            raise
        except Exception:
            pass  # DB read errors should not interrupt the agent

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    async def apply(self, apply_url: str) -> dict:
        print(f"[FormFiller] Starting -> {apply_url}")
        try:
            if _is_linkedin(apply_url):
                return await self._linkedin_flow(apply_url)
            else:
                return await self._generic_flow(apply_url)
        except AgentCancelledError:
            return {"success": False, "reason": "cancelled"}
        except AgentPausedError:
            return {"success": False, "reason": "paused"}
        except Exception as e:
            print(f"[FormFiller] Fatal error: {e}")
            return {"success": False, "reason": f"Unexpected error: {e}"}

    # ------------------------------------------------------------------
    # LinkedIn Easy Apply flow
    # ------------------------------------------------------------------
    async def _linkedin_flow(self, url: str) -> dict:
        email = self.profile.get("linkedin_email", "")
        password = self.profile.get("linkedin_password", "")

        if not email or not password:
            print("[LinkedIn] No credentials found - pausing for user.")
            return {"success": False, "reason": "linkedin_credentials_required"}

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/122 Safari/537.36"
                ),
            )
            page = await context.new_page()
            try:
                # Step 1 - Login
                await self._check_signals()
                login_result = await self._linkedin_login(page, email, password)
                if not login_result:
                    return {
                        "success": False,
                        "reason": "LinkedIn login failed - check your credentials.",
                    }

                # Step 2 - Navigate to job
                await self._check_signals()
                await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                await asyncio.sleep(3)

                # Step 3 - Click Easy Apply
                await self._check_signals()
                easy_apply_clicked = await self._click_easy_apply(page)
                if not easy_apply_clicked:
                    # Fall back to manual if Easy Apply button not found
                    return {
                        "success": False,
                        "reason": "manual_required",
                        "apply_url": url,
                        "instructions": (
                            "1. Open the link below in your browser\n"
                            "2. Click 'Apply' or 'Easy Apply'\n"
                            "3. Fill in your details and submit"
                        ),
                    }

                # Step 4 - Fill multi-step form
                await self._check_signals()
                result = await self._fill_linkedin_steps(page, context)
                return result

            except (AgentCancelledError, AgentPausedError):
                raise
            except Exception as e:
                return {"success": False, "reason": f"LinkedIn error: {e}"}
            finally:
                await browser.close()

    async def _linkedin_login(self, page: Page, email: str, password: str) -> bool:
        try:
            await page.goto(
                "https://www.linkedin.com/login", timeout=20000, wait_until="domcontentloaded"
            )
            await page.fill("#username", email)
            await page.fill("#password", password)
            await page.click("button[type='submit']")
            await asyncio.sleep(4)
            # Verify login succeeded
            if "feed" in page.url or "mynetwork" in page.url or "jobs" in page.url:
                print("[LinkedIn] Login successful.")
                return True
            # Check for wrong password / captcha page
            if "checkpoint" in page.url or "challenge" in page.url:
                print("[LinkedIn] Login blocked by checkpoint/captcha.")
                return False
            # Try checking for error message
            error_el = page.locator(".form__error--is-shown, #error-for-username")
            if await error_el.count() > 0:
                print("[LinkedIn] Login error shown on page.")
                return False
            print(f"[LinkedIn] Post-login URL: {page.url}")
            return True
        except Exception as e:
            print(f"[LinkedIn] Login exception: {e}")
            return False

    async def _click_easy_apply(self, page: Page) -> bool:
        easy_apply_selectors = [
            "button.jobs-apply-button",
            "button:has-text('Easy Apply')",
            "button:has-text('Apply')",
            ".jobs-apply-button",
        ]
        for sel in easy_apply_selectors:
            try:
                el = page.locator(sel).first
                if await el.is_visible(timeout=3000):
                    await el.click()
                    await asyncio.sleep(2)
                    print(f"[LinkedIn] Clicked Easy Apply via: {sel}")
                    return True
            except Exception:
                continue
        return False

    async def _fill_linkedin_steps(self, page: Page, context: BrowserContext) -> dict:
        """Navigate LinkedIn Easy Apply multi-step modal, filling fields at each step."""
        max_steps = 10
        for step_num in range(max_steps):
            await self._check_signals()
            await asyncio.sleep(1.5)

            # Check if modal is still open
            modal = page.locator(".jobs-easy-apply-modal, [data-test-modal]")
            if not await modal.is_visible(timeout=3000):
                # Modal closed - likely submitted
                print(f"[LinkedIn] Modal closed after step {step_num} - application likely submitted.")
                return {"success": True, "reason": "LinkedIn Easy Apply submitted successfully."}

            # Fill all visible text inputs
            await self._fill_visible_text_inputs(page)
            await self._check_signals()

            # Fill selects/dropdowns
            await self._fill_visible_selects(page)
            await self._check_signals()

            # Fill radios
            await self._fill_visible_radios(page)
            await self._check_signals()

            # Fill textareas (check for human-required first)
            human_qs = await self._collect_human_questions(page)
            unanswered = [q for q in human_qs if q["question"] not in self.human_answers]
            if unanswered:
                return {
                    "success": False,
                    "reason": "human_required",
                    "questions": [q["question"] for q in unanswered],
                }
            # Fill answered textareas
            await self._fill_textareas_with_answers(page)

            # Try to click Next / Review / Submit
            clicked_action = await self._click_next_or_submit(page)

            if clicked_action == "submitted":
                return {"success": True, "reason": "LinkedIn Easy Apply submitted successfully."}
            elif clicked_action == "next":
                continue  # go to next step
            else:
                # No next/submit button found
                return {
                    "success": False,
                    "reason": "manual_required",
                    "apply_url": page.url,
                    "instructions": (
                        "The agent filled your details but could not find the final Submit button.\n"
                        "1. Open the link below\n"
                        "2. Log into LinkedIn\n"
                        "3. Locate the Easy Apply button and finish the last step manually."
                    ),
                }

        return {"success": False, "reason": "LinkedIn form exceeded maximum steps - please apply manually."}

    async def _click_next_or_submit(self, page: Page) -> str:
        """Tries to click Submit, then Review, then Next. Returns 'submitted', 'next', or 'none'."""
        # Submit first
        for sel in [
            "button[aria-label='Submit application']",
            "button:has-text('Submit application')",
            "button:has-text('Submit')",
        ]:
            try:
                el = page.locator(sel).first
                if await el.is_visible(timeout=1000):
                    await el.click()
                    await asyncio.sleep(2)
                    print(f"[LinkedIn] Clicked submit: {sel}")
                    return "submitted"
            except Exception:
                pass

        # Review / Next
        for sel in [
            "button[aria-label='Continue to next step']",
            "button:has-text('Next')",
            "button:has-text('Review')",
            "button:has-text('Continue')",
        ]:
            try:
                el = page.locator(sel).first
                if await el.is_visible(timeout=1000):
                    await el.click()
                    await asyncio.sleep(1.5)
                    print(f"[LinkedIn] Clicked next/review: {sel}")
                    return "next"
            except Exception:
                pass

        return "none"

    # ------------------------------------------------------------------
    # Generic form flow (non-LinkedIn)
    # ------------------------------------------------------------------
    async def _generic_flow(self, url: str) -> dict:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/122 Safari/537.36"
                ),
            )
            page = await context.new_page()
            try:
                await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                await asyncio.sleep(3)
                await self._check_signals()

                # -- Check if the page redirected to LinkedIn login --
                current_url = page.url
                if _is_linkedin(current_url):
                    print(f"[FormFiller] Redirected to LinkedIn: {current_url}")
                    await browser.close()
                    # Route through the LinkedIn-aware flow
                    return await self._linkedin_flow(apply_url)

                # -- Check for 'Apply with LinkedIn' / 'Sign in with LinkedIn' buttons --
                li_btn_selectors = [
                    "a[href*='linkedin.com']",
                    "button:has-text('Apply with LinkedIn')",
                    "button:has-text('Sign in with LinkedIn')",
                    "a:has-text('Apply with LinkedIn')",
                    "a:has-text('LinkedIn')",
                ]
                for sel in li_btn_selectors:
                    try:
                        el = page.locator(sel).first
                        if await el.is_visible(timeout=1500):
                            li_href = await el.get_attribute('href') or ''
                            if 'linkedin.com' in li_href or 'linkedin' in (await el.inner_text()).lower():
                                print(f"[FormFiller] LinkedIn Apply button found: {li_href}")
                                await browser.close()
                                target = li_href if li_href else apply_url
                                return await self._linkedin_flow(target)
                    except Exception:
                        continue

                # Detect all inputs
                inputs = await self._extract_all_fields(page)
                if not inputs:
                    # No fields found - provide manual instructions
                    return {
                        "success": False,
                        "reason": "manual_required",
                        "apply_url": url,
                        "instructions": (
                            "The agent could not detect a standard application form.\n"
                            "The page may require a login or use a complex application system.\n\n"
                            "Steps to apply manually:\n"
                            "1. Open the link below\n"
                            "2. Look for an 'Apply' or 'Apply Now' button\n"
                            "3. Create an account if required\n"
                            "4. Fill in your details and submit"
                        ),
                    }

                print(f"[FormFiller] Detected {len(inputs)} fields.")
                await self._check_signals()

                # Check for human-required textarea questions
                human_qs = [
                    inp for inp in inputs
                    if inp.get("tag") == "textarea"
                    and any(
                        kw in (inp.get("labelText") or inp.get("placeholder") or "").lower()
                        for kw in HUMAN_REQUIRED_KEYWORDS
                    )
                ]
                unanswered = [
                    q for q in human_qs
                    if (q.get("labelText") or q.get("placeholder") or "") not in self.human_answers
                ]
                if unanswered:
                    return {
                        "success": False,
                        "reason": "human_required",
                        "questions": [
                            q.get("labelText") or q.get("placeholder") or "Open question"
                            for q in unanswered
                        ],
                    }

                await self._check_signals()

                # Fill standard text inputs via LLM mapping
                mapping = await self._get_llm_mapping(inputs)
                filled_count = 0
                if mapping:
                    filled_count += await self._apply_text_mapping(page, mapping)

                # Fill selects
                select_inputs = [i for i in inputs if i.get("tag") == "select"]
                filled_count += await self._fill_selects_llm(page, select_inputs)
                await self._check_signals()

                # Fill radios
                filled_count += await self._fill_radios_llm(page)
                await self._check_signals()

                # Fill human-answered textareas
                for q in human_qs:
                    q_text = q.get("labelText") or q.get("placeholder") or ""
                    answer = self.human_answers.get(q_text, "")
                    if not answer:
                        continue
                    selector = f"#{q['id']}" if q.get("id") else (f"[name='{q['name']}']" if q.get("name") else "")
                    if selector:
                        try:
                            if await page.is_visible(selector):
                                await page.fill(selector, answer)
                                filled_count += 1
                        except Exception as fe:
                            print(f"[FormFiller] textarea fill error: {fe}")

                print(f"[FormFiller] Filled {filled_count} fields total.")
                await self._check_signals()

                # Submit
                submitted, submit_reason = await self._submit_form(page)
                if submitted:
                    return {"success": True, "reason": f"Form submitted ({filled_count} fields filled)."}
                else:
                    return {
                        "success": False,
                        "reason": "manual_required",
                        "apply_url": url,
                        "instructions": (
                            f"The agent filled {filled_count} field(s) but could not click Submit automatically.\n\n"
                            "Steps to complete manually:\n"
                            "1. Open the link below\n"
                            "2. Your details may need to be re-entered\n"
                            "3. Review the form and click Submit/Apply"
                        ),
                    }

            except (AgentCancelledError, AgentPausedError):
                raise
            except Exception as e:
                return {
                    "success": False,
                    "reason": "manual_required",
                    "apply_url": url,
                    "instructions": (
                        f"The agent encountered an error: {str(e)[:200]}\n\n"
                        "Steps to apply manually:\n"
                        "1. Open the link below\n"
                        "2. Fill in your details\n"
                        "3. Submit the application"
                    ),
                }
            finally:
                await browser.close()

    # ------------------------------------------------------------------
    # Field extraction helpers
    # ------------------------------------------------------------------
    async def _extract_all_fields(self, page: Page) -> list:
        return await page.eval_on_selector_all(
            "input:not([type='hidden']):not([type='submit']):not([type='button']):not([type='file']),"
            "textarea, select",
            """
            elements => elements.map(el => {
                let labelText = '';
                if (el.id) {
                    const label = document.querySelector(`label[for='${el.id}']`);
                    if (label) labelText = label.innerText;
                }
                if (!labelText) {
                    const parentLabel = el.closest('label');
                    if (parentLabel) labelText = parentLabel.innerText;
                }
                if (!labelText) {
                    let prev = el.previousElementSibling;
                    while (prev) {
                        if (['LABEL','P','SPAN','DIV','H1','H2','H3','H4'].includes(prev.tagName)) {
                            const txt = prev.innerText?.trim();
                            if (txt && txt.length < 200) { labelText = txt; break; }
                        }
                        prev = prev.previousElementSibling;
                    }
                }
                // For selects, collect options
                const options = el.tagName === 'SELECT'
                    ? Array.from(el.options).map(o => ({ value: o.value, text: o.text.trim() }))
                    : [];
                return {
                    tag: el.tagName.toLowerCase(),
                    type: el.type || '',
                    id: el.id || '',
                    name: el.name || '',
                    placeholder: el.placeholder || '',
                    labelText: labelText.trim(),
                    options,
                };
            })
            """,
        )

    async def _fill_visible_text_inputs(self, page: Page):
        """Fill text/email/tel/number inputs from profile."""
        field_map = {
            "email": self.profile.get("email", ""),
            "phone": self.profile.get("phone", ""),
            "name": self.profile.get("full_name", ""),
            "full_name": self.profile.get("full_name", ""),
            "linkedin": self.profile.get("linkedin_url", ""),
            "github": self.profile.get("github_url", ""),
            "portfolio": self.profile.get("portfolio_url", ""),
        }
        inputs = await page.eval_on_selector_all(
            "input[type='text'], input[type='email'], input[type='tel'], input[type='number'], input:not([type])",
            """els => els.map(el => ({
                id: el.id || '', name: el.name || '',
                placeholder: (el.placeholder || '').toLowerCase(),
                label: (() => {
                    if (el.id) {
                        const l = document.querySelector(`label[for='${el.id}']`);
                        if (l) return l.innerText.toLowerCase();
                    }
                    return '';
                })()
            }))""",
        )
        for inp in inputs:
            combined = f"{inp['label']} {inp['placeholder']} {inp['name']}".lower()
            value = ""
            if "email" in combined:
                value = self.profile.get("email", "")
            elif "phone" in combined or "tel" in combined or "mobile" in combined:
                value = self.profile.get("phone", "")
            elif "name" in combined:
                value = self.profile.get("full_name", "")
            elif "linkedin" in combined:
                value = self.profile.get("linkedin_url", "")
            elif "github" in combined:
                value = self.profile.get("github_url", "")
            elif "portfolio" in combined or "website" in combined:
                value = self.profile.get("portfolio_url", "")
            if not value:
                continue
            selector = f"#{inp['id']}" if inp.get("id") else (f"[name='{inp['name']}']" if inp.get("name") else "")
            if selector:
                try:
                    if await page.is_visible(selector, timeout=1000):
                        await page.fill(selector, str(value))
                        await asyncio.sleep(0.15)
                except Exception:
                    pass

    async def _fill_visible_selects(self, page: Page):
        """Fill visible select elements by picking the best option with LLM."""
        selects = await page.eval_on_selector_all(
            "select",
            """els => els.map(el => ({
                id: el.id || '', name: el.name || '',
                label: (() => {
                    if (el.id) {
                        const l = document.querySelector(`label[for='${el.id}']`);
                        if (l) return l.innerText.trim();
                    }
                    return '';
                })(),
                options: Array.from(el.options).map(o => ({ value: o.value, text: o.text.trim() }))
            }))""",
        )
        for sel_info in selects:
            if not sel_info.get("options"):
                continue
            selector = f"#{sel_info['id']}" if sel_info.get("id") else (f"[name='{sel_info['name']}']" if sel_info.get("name") else "")
            if not selector:
                continue
            best_value = await self._llm_pick_option(
                sel_info.get("label", "") or sel_info.get("name", ""),
                sel_info["options"],
            )
            if best_value:
                try:
                    if await page.is_visible(selector, timeout=1000):
                        await page.select_option(selector, value=best_value)
                        await asyncio.sleep(0.2)
                except Exception as e:
                    print(f"[FormFiller] Select error for {selector}: {e}")

    async def _fill_visible_radios(self, page: Page):
        """For each radio group, click the best option using LLM reasoning."""
        radio_groups: dict = {}
        radios = await page.eval_on_selector_all(
            "input[type='radio']",
            """els => els.map(el => ({
                id: el.id || '', name: el.name || '', value: el.value || '',
                label: (() => {
                    if (el.id) {
                        const l = document.querySelector(`label[for='${el.id}']`);
                        if (l) return l.innerText.trim();
                    }
                    const parent = el.closest('label');
                    if (parent) return parent.innerText.trim();
                    return '';
                })()
            }))""",
        )
        for r in radios:
            name = r.get("name", "unknown")
            radio_groups.setdefault(name, []).append(r)

        for group_name, options in radio_groups.items():
            labels = [r.get("label") or r.get("value") for r in options]
            best_label = await self._llm_pick_radio(group_name, labels)
            if not best_label:
                continue
            # Find the matching radio and click
            for r in options:
                r_label = r.get("label") or r.get("value") or ""
                if r_label.strip().lower() == best_label.strip().lower():
                    selector = f"#{r['id']}" if r.get("id") else f"input[type='radio'][name='{group_name}'][value='{r['value']}']"
                    try:
                        if await page.is_visible(selector, timeout=1000):
                            await page.click(selector)
                            await asyncio.sleep(0.2)
                    except Exception as e:
                        print(f"[FormFiller] Radio click error {selector}: {e}")
                    break

    async def _collect_human_questions(self, page: Page) -> list:
        """Find textarea fields that need personal human answers."""
        textareas = await page.eval_on_selector_all(
            "textarea",
            """els => els.map(el => ({
                id: el.id || '', name: el.name || '', placeholder: el.placeholder || '',
                label: (() => {
                    if (el.id) {
                        const l = document.querySelector(`label[for='${el.id}']`);
                        if (l) return l.innerText.trim();
                    }
                    return '';
                })()
            }))""",
        )
        result = []
        for ta in textareas:
            combined = f"{ta.get('label', '')} {ta.get('placeholder', '')}".lower()
            if any(kw in combined for kw in HUMAN_REQUIRED_KEYWORDS):
                result.append({
                    "question": ta.get("label") or ta.get("placeholder") or "Open question",
                    "id": ta.get("id"),
                    "name": ta.get("name"),
                })
        return result

    async def _fill_textareas_with_answers(self, page: Page):
        human_qs = await self._collect_human_questions(page)
        for q in human_qs:
            answer = self.human_answers.get(q["question"], "")
            if not answer:
                continue
            selector = f"#{q['id']}" if q.get("id") else (f"[name='{q['name']}']" if q.get("name") else "")
            if selector:
                try:
                    if await page.is_visible(selector, timeout=1000):
                        await page.fill(selector, answer)
                        await asyncio.sleep(0.2)
                except Exception:
                    pass

    async def _submit_form(self, page: Page) -> tuple[bool, str]:
        submit_selectors = [
            "button[type='submit']",
            "input[type='submit']",
            "button:has-text('Apply Now')",
            "button:has-text('Apply')",
            "button:has-text('Submit Application')",
            "button:has-text('Submit')",
            "button:has-text('Send Application')",
            "a:has-text('Apply Now')",
            ".apply-button",
            "#apply-button",
        ]
        for sel in submit_selectors:
            try:
                el = page.locator(sel).first
                if await el.is_visible(timeout=1000):
                    await el.click()
                    await asyncio.sleep(2)
                    print(f"[FormFiller] Submitted via: {sel}")
                    return True, sel
            except Exception:
                continue
        return False, ""

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------
    async def _get_llm_mapping(self, extracted_inputs: list) -> list:
        allowed_keys = [k for k in self.profile.keys()
                        if k not in ("human_answers", "linkedin_password")]
        prompt = f"""
You are an autonomous form-filling agent. Map HTML form fields to user profile keys.

PROFILE KEYS AVAILABLE: {json.dumps(allowed_keys)}
USER PROFILE VALUES: {json.dumps({k: v for k, v in self.profile.items() if k not in ('human_answers', 'linkedin_password')})}

HTML FIELDS:
{json.dumps([i for i in extracted_inputs if i.get('tag') in ('input',)], indent=2)}

Return ONLY a JSON array. Each item: {{ "id": "...", "name": "...", "mapped_key": "..." }}
Only include fields with a confident match to a profile key. Omit everything else.
"""
        try:
            resp = await asyncio.to_thread(self.model.generate_content, prompt)
            text = resp.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            return json.loads(text)
        except Exception as e:
            print(f"[FormFiller] LLM mapping error: {e}")
            return []

    async def _apply_text_mapping(self, page: Page, mapping: list) -> int:
        filled = 0
        for field in mapping:
            selector = ""
            if field.get("id"):
                selector = f"#{field['id']}"
            elif field.get("name"):
                selector = f"[name='{field['name']}']"
            if not selector:
                continue
            value = self.profile.get(field.get("mapped_key", ""), "")
            if not value:
                continue
            try:
                if await page.is_visible(selector, timeout=1000):
                    await page.fill(selector, str(value))
                    filled += 1
                    await asyncio.sleep(0.15)
            except Exception as fe:
                print(f"[FormFiller] Fill error {selector}: {fe}")
        return filled

    async def _fill_selects_llm(self, page: Page, select_inputs: list) -> int:
        filled = 0
        for sel_info in select_inputs:
            options = sel_info.get("options", [])
            if not options:
                continue
            label = sel_info.get("labelText") or sel_info.get("name") or ""
            selector = f"#{sel_info['id']}" if sel_info.get("id") else (f"[name='{sel_info['name']}']" if sel_info.get("name") else "")
            if not selector:
                continue
            best = await self._llm_pick_option(label, options)
            if best:
                try:
                    if await page.is_visible(selector, timeout=1000):
                        await page.select_option(selector, value=best)
                        filled += 1
                        await asyncio.sleep(0.2)
                except Exception as e:
                    print(f"[FormFiller] Select LLM error {selector}: {e}")
        return filled

    async def _fill_radios_llm(self, page: Page) -> int:
        await self._fill_visible_radios(page)
        return 0  # count not tracked here; radios are filled inside

    async def _llm_pick_option(self, label: str, options: list) -> Optional[str]:
        """Ask LLM which dropdown option best suits the user's profile."""
        if not options:
            return None
        # Simple heuristics first (avoids LLM call for common fields)
        label_lower = label.lower()
        for opt in options:
            opt_text_lower = (opt.get("text") or "").lower()
            if "yes" in label_lower and opt_text_lower in ("yes", "true", "1"):
                return opt.get("value")
            if "pakistan" in opt_text_lower and ("country" in label_lower or "location" in label_lower):
                return opt.get("value")
            if "bachelor" in opt_text_lower and "education" in label_lower:
                return opt.get("value")

        prompt = f"""
You are filling a job application form for this user:
Name: {self.profile.get('full_name', '')}
Email: {self.profile.get('email', '')}
Skills: {', '.join(self.profile.get('skills', [])[:10]) if self.profile.get('skills') else 'AI, Python, Backend'}

The field label is: "{label}"
The available options are:
{json.dumps(options, indent=2)}

Return ONLY the "value" field (exact string) of the most appropriate option. Nothing else.
"""
        try:
            resp = await asyncio.to_thread(self.model.generate_content, prompt)
            value = resp.text.strip().strip('"\'')
            # Validate the returned value exists
            valid_values = [o.get("value") for o in options]
            if value in valid_values:
                return value
            # Try partial match
            for v in valid_values:
                if value.lower() in (v or "").lower():
                    return v
        except Exception as e:
            print(f"[FormFiller] LLM option pick error: {e}")
        return None

    async def _llm_pick_radio(self, group_name: str, labels: list) -> Optional[str]:
        """Ask LLM which radio option best suits the user."""
        if not labels:
            return None
        # Heuristics
        for lbl in labels:
            l = (lbl or "").lower()
            if l in ("yes", "true", "i am", "authorized", "eligible"):
                return lbl

        prompt = f"""
Job application radio group: "{group_name}"
Options: {json.dumps(labels)}

User: {self.profile.get('full_name', '')}, applying for an internship in Pakistan.
Return ONLY the exact text of the best option. Nothing else.
"""
        try:
            resp = await asyncio.to_thread(self.model.generate_content, prompt)
            answer = resp.text.strip().strip('"\'')
            if answer in labels:
                return answer
        except Exception as e:
            print(f"[FormFiller] LLM radio error: {e}")
        return labels[0] if labels else None
