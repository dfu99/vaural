# Head Scientist — Product Project Critiques
## 2026-03-22

---

## 1. SMSAissistant — Location-Based Reminders via SMS

**VERDICT: Shelve. Fatal assumptions.**

**Market:** Dead on arrival. Apple Reminders, Google Tasks already do location-based alerts natively. Google Keep just killed its location reminders, consolidating onto OS tools. The "SMS delivery" differentiator is something users didn't ask for — they already have push notifications.

**Fatal Assumptions:**
- Users put locations in Google Calendar events. Most don't — "Dentist" not "123 Main St."
- SMS over push notifications. SMS costs money (Twilio) and is a step backward.
- iOS background location is reliable. It's not — iOS aggressively kills background processes.

**Implementation Traps:** SQLite on Fly.io auto-stop machines = data loss on cold starts. Claude API in hot path for every location ping = expensive. Google OAuth app verification = weeks-long review.

**MVP Gap:** Enormous. Current state is a demo that sends "you are at [address]." Needs calendar parsing, geofencing, auth, and a reason users would pay Twilio fees monthly.

**Pivot if reviving:** Client-side geofence-to-push-notification that parses calendar events locally. No backend, no Twilio, no Google OAuth.

---

## 2. hemingway — AI Scientific Writing Tool

**VERDICT: Pivot the pitch, not the product. Text2Figure is the only moat.**

**Market:** Crowded. Paperpal (250M+ papers), Jenni AI (1M+ users), SciSpace, Elicit, Scite. OpenAI just shipped Prism — free LaTeX-native scientific writing workspace that directly overlaps.

**The One Differentiator:** Text2Figure from real experimental data. Nothing else generates figures from *your own data files* in natural language. That is the wedge.

**Fatal Assumptions:**
- Researchers want a standalone editor. Most write in Overleaf or Word.
- Focus Guard solves a real problem. Unproven — nobody measured if it changes writing quality.
- Local file access is fine. Enterprise/institutional IT will block subprocess execution.

**Implementation Traps:** Text2Figure executes arbitrary Claude-generated Python with 30s timeout — sandbox escape vector. 668KB JS bundle hurts cold load. SQLite = single-user, collaborative editing requires full migration.

**Action:** Strip generalist framing. Make Text2Figure the hero. Get 3 external researchers to use it on real papers. Prism's launch is a 6-month countdown clock.

---

## 3. dippy-WAN — AI Avatar Gesture Videos for Language Learning

**VERDICT: Pivot to pre-rendered asset library. Real-time inference is economically impossible.**

**Market:** Obliterated. HeyGen, Synthesia, D-ID, Colossyan, ByteDance OmniHuman — 20+ SaaS products with lip sync, full-body motion, 175 languages. Alibaba released WAN 2.2 S2V (Aug 2025) — open-source audio-driven talking avatars. You're building a wrapper on the same model family they shipped finished.

**Fatal Assumptions:**
- Gesture-from-text is a substitute for audio-driven lip sync. It isn't — HeyGen drives from speech audio. dippy-WAN generates charades gestures with no audio coupling.
- 86.5% cache hit rate. Simulated on 50-word vocabulary with Zipf access. Real usage has thousands of sentences.
- Inference cost is manageable. WAN 14B on A100 = 10 min/sentence, $1.50/hr. A 20-sentence lesson = $4.50 in compute.

**Pivot:** Integrate with alinakai immediately. Single avatar + 100 sentences. Pre-generate all clips offline. Ship as curated content library, not real-time API.

---

## 4. alinakai-claude — RL-Powered Language Learning

**VERDICT: Pivot pitch from consumer app to research demo / SDK.**

**Market:** Duolingo has 50.5M DAU and $748M revenue. They used AI to ship 148 new courses in early 2025. Lingvist, LingoLooper, Taalhammer all occupy the AI-SRS lane.

**The One Differentiator:** RL policy that learns which tokens to surface per user. This is technically novel and rare. But it's buried under a generic-looking practice UI.

**Fatal Assumptions:**
- Stable token proficiency profiles exist for the RL policy to act on. With mastery rate at 2.6% after 500K steps, the policy barely works in simulation.
- Users tolerate text-input-only practice. Competitors do voice, roleplay, gamification.
- LLM generation works. It doesn't — Ollama CPU-only, 40+ refill failures. The differentiating feature is the fallback.

**Implementation Traps:** Policy scores collapsed (all users 0.547-0.560). GPT-4o-mini grading in critical path with no local fallback. No mobile. No users.

**Pivot:** Position the RL token-targeting as an SDK/backend intelligence layer for language app developers. As a consumer app, 12-18 months from defensible differentiation. As a research demo of RL-driven vocabulary scheduling, already publishable.

---

## 5. vibevelop — DAG-Based AI Project Management

**VERDICT: Pivot or kill. VM-per-component is a 3-year infrastructure play with no buyer.**

**Market:** Linear shipped "Agentic Backlog" monitoring Slack/GitHub. Notion has "Agentic Sync." ClickUp Brain is embedded everywhere. No enterprise will replace Linear+GitHub for a self-hosted Go dashboard.

**Fatal Assumptions:**
- Enterprises want another self-hosted platform. They want SaaS with SSO.
- Interface contracts need a new tool. OpenAPI/Protobuf/Backstage already exist.
- "Cognitive scoping" (restricting agent context) is a product. Cursor and Claude Code already let you scope manually.

**Implementation Traps:** Docker-per-component at scale = infra overhead that kills adoption. Distributed state across isolated VMs = Kubernetes-hard. Only 2 completed objectives, both pre-alpha.

**Pivot worth saving:** The black-box doc engine. Reframe as "Backstage but AI-native" — component catalog where AI auto-generates interface docs from existing code. Sell to microservices teams.

---

## Summary

| Project | Verdict | Key Action |
|---------|---------|-----------|
| SMSAissistant | Shelve | Fatal market + assumption gaps |
| hemingway | Pivot pitch | Make Text2Figure the hero, get 3 external users |
| dippy-WAN | Pivot to offline | Pre-render 100 clips, integrate with alinakai |
| alinakai | Pivot to SDK | RL token-targeting as backend intelligence layer |
| vibevelop | Pivot or kill | Save doc engine, kill VM orchestration |
