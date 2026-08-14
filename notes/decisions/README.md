# decisions/ — one fork, and the branch not taken

`status: active | superseded | reversed`

Written **when the fork is taken, before the work** — a decision note written afterwards is a
justification, and reads like one.

What it must carry:

- **The fork.** Both branches, stated so that the rejected one still sounds reasonable. If it does
  not, you have written an advert.
- **What decided it** — the measurement, the constraint, or the judgement call. "Judgement" is an
  honest answer; "it was obvious" is not.
- **What would reverse it.** The condition under which the other branch becomes right.

When it reverses, do not edit the note: set `status: reversed`, add `superseded_by:`, and write the
new one. The pair is the useful object — this repository has already re-taken the same fork twice
without noticing, and a reversed note is what makes that visible.
