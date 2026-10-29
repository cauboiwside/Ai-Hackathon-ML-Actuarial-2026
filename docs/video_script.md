# Video script (target 4:45; only the first 5 minutes are graded)

Format: screen recording. Show the thing named in [brackets]; speak the text. Speak at a normal pace; don't rush. Numbers in {braces} get filled in from the final results.

---

**[0:00 — title slide: team name, both names, "Actuarial pricing across markets with AI assistance"]**

Hi, I'm Toby, and this is [PARTNER]. We priced motor, buildings and contents for a developed market, then adapted the approach to an emerging market with two years of history. I'll cover four things: how the model works, where AI helped and where we had to catch it, what changed for the new market, and the guardrails.

**[0:25 — repo README / pipeline diagram: scan → target → models → validation → level → tests]**

The pipeline is the same in both parts. A data-quality scan first. Then claims matched to policy rows and capped. Then three gradient-boosting models per line: a Tweedie model matching the scoring metric, a frequency-times-severity pair, and a per-peril version, which is the structure a home-pricing team would actually use. We validate with cross-validation grouped by policy, so a policy's years never straddle train and test, and we blend the models with weights fitted on held-out predictions.

**[0:55 — decision log, scrolled slowly; stop on the exposure row]**

The most important file in the repo is the decision log: forty-odd entries, each with who proposed it and how it was checked. Three examples of what that caught.

**[1:10 — the exposure-by-band table with the impossible A/E]**

First, the data had planted problems. About nine hundred rows per line carried exposure in days instead of years, so one row counted 365 times. We found it because actual-to-expected was above one in every exposure band but below one overall, which is arithmetically impossible unless rows fall outside the bands. Fixing it, and a scrambled construction-type code, was worth more than any modelling change.

**[1:35 — claim rate by customer_service_contacts chart]**

Second, leakage. Customer service contacts looked like the best predictor by far. But zero contacts means zero claims in every row, because people contact the insurer *after* they claim. It's not knowable at quote time, so we excluded it, and in Part 2 we confirmed the test column had been scrubbed anyway.

**[1:55 — experiment table: adjusted target vs raw target]**

Third, the AI was wrong and the experiment caught it. My first approach weighted each late-reported claim individually to get to ultimate. A controlled test showed that *hurt* accuracy: the weights added noise. We reversed it and apply reserving adjustments at the aggregate level instead. That's the pattern throughout: the AI proposes, the experiment decides, the log records.

**[2:25 — Part 2 comparison table: what changed between markets]**

Part 2. The new market had five percent of the data, no credit band, no mileage, no telematics, banded ages, a different region system, twelve percent missing values, and a 2025 book that is seventy-six percent new business from a broker, versus fourteen percent in the history.

**[2:50 — model table: Transfer / Local / Offset / Pooled with D² by line]**

We didn't start from scratch. We tested four ways to use the developed market: transfer the model and re-level it; train local only; train the local model on *deviations* from the transferred prior, which is credibility weighting; and pool both markets with a market flag. On the home lines the developed prior was worth twice the local data. The best transfer turned out to be the frequency-severity model: frequency relativities travel across markets, severity levels don't, and that structure keeps them apart.

**[3:25 — new-business A/E table]**

The biggest single risk was new business. Locally it cost about one and a half times renewals, on very thin exposure. We applied a credibility-weighted uplift and reported it as the main uncertainty in the price.

**[3:45 — gender paragraph in the write-up]**

The new market's data included gender. We treated that as a regulatory decision, not a modelling one: an insurer coming from a market where gender rating is prohibited shouldn't adopt it abroad. We measured the cost of leaving it out. It was zero. Age band and vehicle already carry the signal.

**[4:05 — tests passing in the terminal; guardrails list]**

Guardrails. Seven automated tests on every submission: format, every row once, non-negative, level within range, exposure in years, categorical levels consistent. The pricing tool reports the unbiased expected cost separately from the metric-optimal prediction, because the Tweedie score rewards a different number than a pricing team should use. And every leaderboard-derived level is disclosed as what it is: the public sample used as a stand-in for a year of experience.

**[4:30 — results slide: Part 1 0.1111 → 0.1294, 8th of 37; Part 2 {final}]**

Results: Part 1 went from thirty-third to eighth. Part 2 is {rank} on the public board. What we'd want before using this for real: development history for bodily injury, a year of actual emerging-market experience to settle the new-business level, and real persistence in claims, which this synthetic data doesn't have. Thanks.

**[4:45 — end]**
