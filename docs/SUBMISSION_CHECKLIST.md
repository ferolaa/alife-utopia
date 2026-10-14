# Sapienza DLAI 2025/26 submission checklist

Checked on 14 October 2026 (Europe/Rome) against the official
[course repository](https://github.com/erodola/DLAI-s2-2026),
[guidelines.pdf](https://github.com/erodola/DLAI-s2-2026/blob/main/guidelines.pdf) and
[report template](https://github.com/erodola/DLAI-s2-2026/blob/main/template.zip).
Observed course main commit: 9bc0fd141fe6d56c12ca2f51f715ee381ff70ada.
These are authoritative sources, not an inferred project-specific deadline.

## Verified official requirements

- [x] BYOP project format; artificial life is an explicitly allowed topic. Prior topic
  approval/email is not required when the topic fits the listed areas (guidelines §1).
- [x] Two-person groups are permitted without prior permission; project scope is assessed
  relative to team size (guidelines §4).
- [ ] Supply a project repository link. Public/private are both allowed; private repositories
  must be shared with the professor and teaching assistants (guidelines §2).
- [ ] Write the official report using the mandatory unmodified template. The base limit
  is 2 pages plus 1 per additional student: **three main-text pages for this two-person team**.
  AI-use statement, references and appendix are excluded (guidelines §§2,4; template).
  Do not change margins/spacing/style to evade the limit. Excess main text is ignored.
- [ ] Include a mandatory honest AI-use statement **before references**. Appendix, if used,
  follows references. Specify actual assistance with code, algorithms, experiment scripts,
  debugging, data processing, plots and writing as applicable. Do not declare merely
  spelling assistance: the documented work includes substantial AI-assisted development.
  Students must understand, verify and explain all assisted content (template AI section).
- [ ] Email the required material to **rodola@di.uniroma1.it**, **strano@di.uniroma1.it**,
  and **solombrino@di.uniroma1.it**, with exact subject **[DLAI 2026] Project delivery**
  (guidelines §3). Do not send to different addresses. No email has been sent here.
- [ ] Both students confirm registration on Infostud for the selected exam session.
  Only registered, timely submitted projects are graded; do not request early grading
  (guidelines §5 and course README).

The course page states project = 100% of the grade. Oral examination is optional,
with at most 3 points added or subtracted; instructors may require it when necessary.
Prepare to explain code, REINFORCE, matching, statistical units and limitations.

## Deadlines and student-confirmation items

The course README currently lists:

| Exam session | Project cutoff |
|---|---|
|19 June 2026|12 June 2026|
|17 July 2026|10 July 2026|
|18 September 2026|11 September 2026|

All listed dates precede this check. **No later session/cutoff is published in the
checked sources.** The previously discussed remaining work time is not an official
deadline. Confirm the intended session, its cutoff and any later announcement through
the course/Infostud or instructors; do not infer a new date or assume late acceptance.

Students must confirm: both names/student IDs, selected session, registration,
repository access by graders, deadline, final template version and actual AI-use
scope. The template offers a typical report structure, not a mandatory section order.
Use references for Calhoun and policy gradients; do not present the simulator as biology.
The official report and AI statement have not been written by this task.

## Repository visibility and safe delivery

GitHub HEAD/default was inspected and resolves to **main**, not the completed research
branch. Safest immediate action, requiring no repository-setting change: submit both
an explicit branch URL and an immutable final submission commit URL, and link the
correct branch directly from the report/email. Current preserved experiments:

- https://github.com/ferolaa/alife-utopia/tree/breakdown-mechanisms
- https://github.com/ferolaa/alife-utopia/tree/878838689d5e6d6fa177b0b46864f1f7b44353f7

After documentation is reviewed and separately committed, update the immutable link
to that final documentation-inclusive commit. The completed branch README contains
an explicit branch notice. The root URL still opens old main until settings change.

For the clearest root landing page, an owner can later approve changing only GitHub's
default branch to breakdown-mechanisms. This does not merge or modify main, but can
change the default PR target and other branch-dependent workflows. Review those
settings first. Do not merge research into main just to change the landing page.
**No default-branch change, commit, push or submission is authorized by this task.**

## Final delivery checks

- [x] Completed science/checkpoints preserved at 8788386; 286 tests passed at preservation.
- [ ] Review and separately preserve the documentation changes; use the final immutable SHA.
- [ ] Check report page count, template compliance, figure readability and working links.
- [ ] Include AI declaration and references in the prescribed order.
- [ ] Check every graded scientific claim against stored analysis; report posthumous success,
  unequal follow-up, programmed mechanisms and nonsignificant adjusted comparisons.
- [ ] Confirm session/cutoff/Infostud, all recipient addresses and access, then submit.
