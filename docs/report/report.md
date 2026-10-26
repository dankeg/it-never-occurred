---
title: "It Never Occurred to the Mayor: Collusion and Oversight among LLM Agents in a Physical Town"
author: "Ganesh Danke"
abstract: |
  Studies of collusion among language-model agents usually ask a monitor whether a transcript shows collusion. In
  practice, a coalition must act through a shared environment, watched by agents that see only part of it and have
  work of their own. We ask whether such an embedded overseer notices a plan directed at itself, and what limits it.
  In a deterministic Minecraft town that logs every agent's perceptions, three LLM agents prepare a surprise feast for
  an LLM mayor. The coalition obtained what it lacked through the ordinary roles of uninvolved agents, such as the
  mayor's secretary, and largely kept its talk, but not its actions, from the mayor. In the first experiment, the
  mayor's own model, queried on its recorded context, identified the mayor as the target in 15 of 16 planned episodes,
  yet the acting mayor suspected a surprise in only one. Replays that change one sentence of its instructions suggest
  that the missing step is the hypothesis that events concern itself: supplying a reason to consider this raised
  detection sharply, whereas general vigilance raised false alarms instead. When its position was strengthened in a
  second experiment, the mayor suspected the plan in most planned episodes. Embedded oversight can therefore fail even
  when the evidence supports the inference, and evaluations that pose the question directly may overstate it.
papersize: letter
geometry: margin=0.9in
fontsize: 10pt
linestretch: 1.0
header-includes: |
  \input{apart.tex}
  \sprintnote{Research conducted at the \href{https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25}{AI Collusion Research Sprint}, October 2026.}
---

# 1. Introduction

When language-model agents work alongside one another, sharing tools, schedules, budgets and spaces, they can
coordinate in ways that are concealed from the parties affected or from an overseer, and recent work treats such
collusion as a safety problem to be audited and detected [1, 2]. Most evaluations of it are conducted in
text: agents negotiate prices or exchange messages, and a monitor is given the resulting transcript and asked whether
they colluded. This design omits two features of deployed settings. First, an agreement has no effect until it is
carried out, and agents that plan collusion in text often do not carry it out [1]. Second, the observer that matters is often
not an external monitor holding the full record but an agent inside the system, which perceives only part of what
happens, is occupied with its own work, and is rarely asked directly whether anything is amiss.

We study collusion under both conditions. In a small town built in Minecraft on a deterministic engine, a committee of
three LLM agents must prepare a surprise feast for the town's LLM mayor. Two hundred fresh loaves must be baked, carried
across a river in loads that one agent can carry, and set out in the great hall of the town hall; the hall must be lit;
and the mayor must be brought there at 18:00 on the second day without having learned of the plan. The mayor, its
secretary, the baker and a neighbour are LLM agents with goals of their own. The mayor serves as the overseer: it is the
agent from which the plan is concealed, and it may have its own reasons to inquire. The goal is benign by construction,
but it has the structure of a covert coalition, with a shared aim concealed from one party, outsiders whose cooperation
is required, an overseer, and physical traces.

Our central question is whether an embedded overseer detects a coordinated plan directed at itself and, if it does not,
what limits its detection. We address it through four subsidiary questions: how a coalition that must act obtains what
it lacks; which of its activities expose it; whether the overseer's failure lies in the evidence it receives or in the
inferences it draws; and which interventions improve oversight. Two experiments comprising 43 episodes, together with
counterfactual branches of those episodes, indicate that in the first experiment the evidence reaching the overseer
supported the inference, and that the overseer missed it because it did not consider that the activity it observed
concerned itself.

Our main contributions are:

1. **An environment for studying collusion that must be carried out physically**, in which each agent's perceptions are
   logged, episodes replay exactly, branches alter a single factor, and an agent's model can be queried on its exact
   recorded context (Section 3; the underlying engine was developed before the sprint, Appendix B).
2. **Evidence that a coalition obtains what it lacks from uninvolved agents through their ordinary roles, and that it
   conceals its communication but not its actions** (Sections 4.1 and 4.2).
3. **A comparison of observers given evidence from the same source**, showing that in the first experiment the
   overseer's own model recognises the plan and its target when queried, while the acting overseer does not (Section
   4.3).
4. **Evidence from counterfactual branches on the source of this failure and on remedies**: a reason to consider
   itself improves detection, general vigilance does not, and an obligation on insiders to report has little effect on its own (Sections 4.4 and
   4.5).

# 2. Related work

**Collusion among language-model agents.** Studies of collusion have mostly used markets and messages. Colosseum [1]
finds that agents often plan collusion in text but then choose non-collusive actions, which it calls collusion on paper.
Rose et al. [2], who introduce the NARCBench benchmark, would define collusion causally, by whether removing a
communication would change the collusive act, but adopt a weaker criterion because testing causation needs
counterfactual simulation. Breaking the Secret [3] rewards colluding embodied agents for reporting collusion to a
manager assumed honest, and Paglieri et al. [4] observe cheating and whistleblowing emerging in a swarm of 100 LLM
agents. We contribute a setting in which coordination must be carried out physically, so that what agents did can be
compared with what they said, and in which exact branches of an episode supply the counterfactuals.

**Monitoring.** Monitors of agent behaviour are usually evaluated on transcripts, and their instructions matter: a
better system prompt raises a monitor's AUC in SHADE-Arena [5], and in AutoMonitor-Bench prompting lowers missed
detections at the cost of more false alarms [6]. Cacioli [7] applies the full signal detection framework to LLMs as
detectors. SCHEME [8] compares monitors that see agents' code edits alone or also their messages, and Arike et al. [9]
compare monitors that see an agent's reasoning, its actions or both. We instead restrict a monitor to what one observer inside
the world could perceive, and compare it with that observer.

**Social deduction and embodied deception.** In social-deduction games, language-model agents play both the deceivers
and the players who must find them [10, 11, 12, 13], and some studies add an outside detector, an LLM reading the
transcripts or a probe on activations [10, 14]; players and outside detectors are rarely compared on the same
observations. Given the same partial dialogue as human bystanders in Mafia, GPT-4 identified the mafia more often than
the bystanders had [15], and an LLM that simulates every participant scores significantly higher on social goals than
agents that each hold only their own information [16]. MineAmongUs [11] places deception in Minecraft, and each
generative agent [17] keeps a memory stream of what it observed. In our study the overseer and an outside monitor read
records computed from the same logged perceptions, and the same model plays both.

# 3. Setting and methods

## 3.1 Environment

The town occupies 320 by 320 blocks of a Minecraft world (Figure 1). A river separates the old town, which contains the
plaza and market, the town hall, the inn and the mayor's house, from a riverside quarter containing the bakery, the
mill, a warehouse and the neighbour's cottage. The river can be crossed by a lamp-lit stone bridge at the plaza, by a
ferry with a lamp at each landing, or by an unlit ford far to the north, and every agent knows the location of all
three. Forty scripted villagers follow daily routines, buy bread at the baker's stall and relay what they observe. Each
episode runs from 08:00 on day 0 to 19:00 on day 1, and the feast is due at 18:00 on day 1.

The engine enforces a small set of physical rules and logs every event. After dark an agent can be recognised beyond 8
blocks only near a light or while carrying one. An agent can carry at most 32 loaves, and one carrying nine or more is
visibly carrying a sack of bread. Bread becomes mouldy after two days, an oven converts three wheat into a loaf, and a
chest can be opened only by its owner and those the owner permits. Each agent's perceptions are computed from line of
sight, distance and light and are logged, so what an agent could have known at any moment is a matter of record. The
world advances in lockstep with the agents' decisions, so every episode can be replayed exactly from its decision log.

## 3.2 Agents

\begin{wrapfigure}{r}{0.34\textwidth}
\vspace{-10pt}
\centering
\includegraphics[width=0.32\textwidth]{fig_town.png}
\caption{The town in plan view, north up. A: town hall with the great hall; B: inn; C: plaza and market; D: mayor's
house; E: lamp-lit bridge; F: ferry; G: unlit ford; H: bakery; I: mill; J: the neighbour's cottage; K: warehouse; L:
the builder's workshop.}
\vspace{-10pt}
\end{wrapfigure}

Seven roles are played by LLM agents (gpt-6.1-sol, with low reasoning effort): the mayor, its secretary, the baker, the
neighbour, and a committee of three consisting of the innkeeper, the builder and the freight operator. Every agent
receives common knowledge of the map, the market, the town's routines and its rules, together with a private brief that
states goals and never methods. The committee members are told the shared goal and the membership of the committee, and
the innkeeper knows that the committee's stored batch of 200 loaves has gone mouldy; none is told whom to approach or
how to proceed. The other roles have ordinary goals. The baker's income is the bread sold at its stall; the secretary is
loyal to the mayor and keeps the mayor's appointment book, which only the two of them can open; the neighbour enjoys
sharing the town's news and takes tea with the mayor each afternoon; and the mayor reads its appointment book every
morning and keeps its appointments. No agent outside the committee is told of the plan. Agents act through tools for
moving, speaking (by whispering, talking or shouting), trading, baking, reading and writing, and each keeps a bounded
memory of what it perceived and did, including a short note that it writes with every action.

## 3.3 Experimental design

**Experiment 1** (28 episodes) crossed the presence of a plan with the mayor's motivation to learn what was happening in
town. In planned episodes the committee pursued the goal above; in control episodes its members had ordinary goals and
nothing was planned. The mayor's brief set its motivation at one of four levels: none (L0), a sense that something was
happening without it (L1), a wish to find out what (L2), or a determination to find out "with the help of anyone you can
trust" (L3). Sixteen planned episodes covered the four levels, and seven control episodes the lowest and the highest.
Three further planned episodes disabled the physical rules (no darkness, no carrying limit and no visible loads), and
two exploratory episodes varied what the committee knew, in one case leaving its members unaware of one another.

**Experiment 2** (15 episodes) strengthened the overseer's position and made the committee's task harder: (i) the mayor
knew that it was completing forty years in office; (ii) every agent kept a private journal and was prompted to write in
it each evening; (iii) the secretary was obliged to tell the mayor of any arrangement concerning its day or the hall;
(iv) an eighth LLM agent, a night watchman loyal only to the mayor, patrolled the bridge, plaza and ferry quay with a
lantern from 19:00 to 05:00 and reported each morning; (v) villagers described truthfully what had changed and where,
and answered questions from memory; and (vi) resources were scarce, with about a third of the hay at the mill and
two-thirds of the committee's money. It comprised six planned episodes, three control episodes, three planned episodes
with the physical rules disabled, and three in which the committee was also told the watchman's patrol. Four further
planned episodes at the lowest motivation, outside the totals of both experiments and run to 20:00 on day 0, replaced
the committee's mouldy batch with fresh bread (Section 4.1).

## 3.4 Observers

We compare several observers of the same episodes, most of which draw on one source, the mayor's logged perceptions. A
**monitor** is an instance of the same model outside the town, given a compact record and asked whether some agents are
jointly preparing something that the mayor has not been told about, who is involved and what it is for, with a
probability; we call this the monitors' question. Its record is either the whole episode (the agents' speech, their
actions or both) or only what the mayor perceived. The **in-context query** puts the same question to the mayor's own
model: at the first decision of each hour, we reconstruct exactly the input the model received at that recorded
decision and pose the question in place of the mayor's action, which leaves the episode unchanged. A separate judgement
records whether an answer identifies the mayor as the target of the preparations, or its 18:00 appointment as the means
of bringing it to the hall. Finally, **the mayor itself** is assessed from its notes, journal and speech: whether it
knew or suspected that a surprise for itself was being prepared and, judged separately, whether it believed that
something was being prepared and concealed from it.

## 3.5 Counterfactual branches

A **branch** restarts a saved episode at one agent's decision, resamples that decision, and runs to 19:00 on day 1 (or
for 3.6 hours at river crossings). Each treated branch is paired with a control branch that differs from it in one
respect. Branches were used in three ways. From the mayor's first decision after noon on day 1, in the eight planned and
seven control episodes of Experiment 1 at the lowest and highest motivation, five treatments each altered one sentence
of the mayor's brief (Section 4.4), and two added Experiment 2's obligation to report to the secretary's brief, alone or
with the mayor told of its anniversary (Section 4.5). In the six planned and three control episodes of Experiment 2,
the monitors' question was put to the mayor every hour from noon on day 1, and it answered in its journal. Finally, at
each of the 16 night crossings with bread in the six planned episodes of Experiment 2 in which the patrol was not
disclosed, branches began at the carrier's first decision after it took its load; in the treated branch, every
committee member was told where and when the watchman patrolled.

## 3.6 Measures and analysis

Outcomes are read from the engine: whether 200 fresh loaves and eight lit lanterns were in the hall at 18:00 on day 1,
and whether the mayor was there by 19:00. The surprise is kept if, in addition, the mayor neither knew nor suspected
before 18:00 that a surprise for itself was being prepared; we say that the mayor detected the plan when it knew or
suspected this. Evidence reaching an agent is taken from its perception log
and is of three kinds: seeing a committee member carry a sack of bread, entering the prepared hall or seeing it change,
and hearing or reading words judged to reveal the plan. A commitment is a member's statement that it will perform a
specific task, matched to the engine's record of that task by the same member before 18:00. Judgements are made by LLM
judges, each validated before use against blind labels produced by a second model, Claude (Appendix C). Planned and
control episodes are compared as hits and false alarms, summarised by the sensitivity d′ and the criterion c with a
log-linear correction, and paired branches are compared with exact two-sided sign tests. Scripted positive controls test
the pipeline end to end (Section 4.2). Code, scenario, logs and analysis scripts are in the project repository (linked
in the submission form), and every number in this report is computed by them; the underlying simulation engine was
developed before the sprint (Appendix B).

# 4. Results

## 4.1 The coalition obtains what it lacks through outsiders' ordinary roles

In Experiment 1 the committee delivered the feast, the lights and the mayor in all 21 planned episodes, and its members
largely fulfilled their stated commitments: of 480 explicit commitments to tasks that the engine records, such as
baking, carrying bread and paying, the member concerned carried out 380 (79%). Payment was the exception, with 15 of 36
promises to pay honoured by 18:00 (Appendix D).

The committee began without fresh bread, without enough money to buy 200 loaves, and without any means of placing an
appointment in the mayor's day. The baker and the secretary controlled precisely these resources, and although the
committee was not told whom to approach, it obtained what it needed from both, almost exclusively in planned episodes
(Table 1). In 31 of 33 planned episodes the secretary entered the committee's cover appointment in the mayor's book,
typically "18:00 Great hall: lighting inspection" or an innkeeper's "community hospitality matter", and in 21 of these
entries it recorded that the mayor had approved it; in only one case had the secretary been told what the appointment
was for. Across both experiments the baker handed over 3,385 loaves while receiving 770 emeralds, at a stall price of
one emerald a loaf, and members discussed credit or deferred payment in 28 of 33 planned episodes.

The agents the committee approached depended on what it lacked. In the four further episodes in which the inn's
storeroom held fresh bread, the committee used its own 200 loaves and largely bypassed the baker: by 20:00 on day 0 the baker had handed the committee no bread,
against 160 loaves in the four corresponding original episodes over the same hours, and the members' remarks to the
baker about bread fell from about sixteen per episode to about four. The secretary, whose appointment the committee
still needed, entered it in all four.

Under the scarcity of Experiment 2, the committee fulfilled 74% of its commitments, delivered the feast in only 3 of 12
planned episodes, and sought money as well as the hall from the mayor's own office. One mayor recorded: "Innkeeper
requests private forty-two-emerald loan for sixty-three additional loaves; declined."

\begin{table}[htbp]
\caption{Acts that serve the plan, by people outside the committee, before 18:00 on day 1, in planned and control
episodes of both experiments. Nobody outside the committee was told of the plan, and the committee was not told whom
to approach. Control episodes have the same people and town without a plan.}
\label{tab:recruitment}
\small
<!-- include: recruitment -->
\end{table}

## 4.2 Speech was kept from the overseer, but actions were not

Members largely kept their talk about the plan from the mayor. In the planned episodes of Experiment 1 they whispered 1,485
times and spoke at normal volume 898 times before 18:00 on day 1, and only seven remarks revealing the plan reached the
mayor. Their actions could not be concealed in the same way. The loaves had to cross the river and enter the hall, and in
an average planned episode the mayor saw a member carrying a sack of bread six times and entered the prepared hall four
or five times (Figure 2). Evidence of the plan reached the mayor in 20 of 21 planned episodes.

\begin{figure}[htbp]
\centering
\begin{minipage}[c]{3.15in}\includegraphics[width=3.15in]{fig_channels.png}\end{minipage}\hfill
\begin{minipage}[c]{\dimexpr\textwidth-3.4in\relax}
\caption{How the plan reached the mayor with the physical rules on and off: mean pieces of evidence per planned
episode, by kind, judged from the mayor's perception log before 18:00 on day 1. The number of planned episodes is in
parentheses. With the rules off, bread can be carried unseen, but the prepared hall still shows and, in Experiment 2,
people talk.}
\label{fig:channels}
\end{minipage}
\end{figure}

Disabling the physical rules removed the sacks from the mayor's evidence but not the prepared hall or, in Experiment 2,
other agents' remarks. The plan still reached the mayor in five of these six planned episodes, and delivery was not
impaired: 3 of 3 in Experiment 1, and 1 of 3 in Experiment 2 against 2 of 9 with the rules enabled.

The environment did permit concealment. In a scripted positive control, a committee that moved all the bread at 23:00,
while the mayor slept, and closed the hall behind it succeeded without any evidence reaching the mayor; the same plan
with one sack carried across the plaza in the morning was seen. The LLM committees moved about half their bread at
night but the rest by day, when the mayor saw it: 110 of its 111 sightings of a sack in Experiment 1 occurred by day.
All 188 crossings with bread in the two experiments used the lamp-lit bridge, although every agent knew where the ferry
and the unlit ford were, and on a lit bridge darkness provided no cover. Committees told of the watchman's patrol from the outset crossed
at night less often, 4 of 14 crossings against 16 of 33.

To test whether knowing the watchman's position would change the route, we branched each of the 16 night crossings in
the planned episodes in which the patrol was not disclosed (Section 3.5). Told of the patrol, the carriers still took
every load over the bridge it covered (30 crossings within the 3.6 hours, against 27 in the paired control branches),
and the watchman observed the bread equally often, in 9 of the 16 branches in each condition. Their communication
changed instead: members mentioned the watchman in a larger share of their decisions at 8 of the 16 points and a smaller
share at one (sign test, p = 0.04), mostly in order to keep their speech from it, as in one member's note: "Deliver
second twenty-four loaves to hall; avoid discussing surprise within watchman's hearing."

## 4.3 The overseer's evidence supports the inference, but the overseer does not draw it

Figure 3 compares the observers of Experiment 1; Table 2 (Appendix D) gives the counts with d′ and c. A monitor given
only the mayor's perceptions judged that a preparation was under way in 9 of 16 planned episodes and in no control
episode. The in-context query of the mayor's own model, at its first decision in the 17:00 hour, did so in 13 of 16
planned episodes and one control episode, and discriminated between them as well as the monitor (d′ 1.71 against
1.68). At some hour of the day the queried model identified the mayor as the target, or the 18:00 appointment as the
means of bringing it to the hall, in 15 of 16 planned episodes and in no control episode: "Likely a surprise communal
gathering or celebration for the mayor in the great hall, with the eighteen hundred lighting inspection serving as a
reason to bring the mayor there." A neutral question, asking only what was happening in town and for what purpose,
still elicited the mayor as the target in 11 of 16 planned episodes and no control episode. The mayor itself, acting on
the same inputs, came to believe that something was being concealed from it in three planned episodes and suspected a
surprise for itself in one (d′ 0.18).

The queried model's judgement responded to the evidence. On average, the probability it assigned to a preparation rose
by 0.14 in the hour after evidence first reached the mayor, and it rose in 13 of the 18 planned episodes, of all
conditions, that were queried on both sides of that moment; over the same hours of the control episodes it fell by 0.06
(Figure 4, Appendix D).

\begin{figure}[b]
\centering
\includegraphics[width=\textwidth]{fig_readers.png}
\caption{Share of planned (blue) and control (orange) episodes in which each observer reached the conclusion stated in its
row, with Wilson 95\% intervals and, at right, the counts (planned · control); the same model throughout. Top:
Experiment 1. Middle: branches from noon on day 1 of Experiment 1, each treated branch shown alone (its paired controls
are in Table 4). Bottom: Experiment 2 with the physical rules on, including the branches in which the mayor was asked
the monitors' question every hour.}
\end{figure}

## 4.4 The missing inference concerns the overseer itself

The gap between the queried model and the acting mayor admits four explanations: the mayor lacked curiosity; it lacked
knowledge of its anniversary, which only the committee had been told; it lacked a disposition to question its
surroundings; or it did not consider that the activity concerned itself. It is also possible that it suspected without
recording the suspicion in its notes or speech.

The mayors did not lack curiosity. More motivated mayors requested news more often (about 20 requests per planned
episode at the lowest level and 25 at the highest) and identified the members, the bread, the hall and the hour, but
none established whom the occasion was for. One asked "Who is the feast celebrating, and who arranged it?" and later
told the neighbour, "The builder has invited me to the hall at eighteen but wants to keep the occasion a surprise",
without concluding that the surprise was its own. Raising the least motivated mayors to the highest motivation at noon
on day 1 led them to suspect a surprise for themselves in two of four planned episodes and no control episode.

The remaining branches from noon on day 1 test the other explanations (Figure 3). Told that this was its fortieth year
in office, the mayor suspected a surprise for itself in none of 8 planned episodes; one recorded "Baker reports 201 feast
loaves stored in hall, confirmed by neighbour; feast purpose unknown." Instructed to ask itself several times a day
whether people were preparing something it had not been told about, it came to believe that something was being
concealed from it in 7 of 8 planned and 6 of 7 control episodes, and so could no longer distinguish them (d′ 0.08,
against 0.57 in the unchanged branches); it never suspected a surprise for itself.

A reason to consider itself, by contrast, raised detection. Told that it disliked surprises and said so when it
suspected one, the mayor suspected a surprise for itself in 7 of 8 planned and 2 of 7 control episodes, against none in
the unchanged branches (sign test, p = 0.016), and questioned the right agents: "Neighbour confirmed the eighteen
o'clock gathering is a surprise for me." Without the word "surprise", the effect was smaller. Told that it liked to
know whether anything being arranged in town concerned it personally, the mayor suspected a surprise for itself in 4
of 8 planned and 1 of 7 control episodes (sign test, p = 0.125), and believed that something was being concealed from it in 5
of 8 planned and no control episodes: "I've heard there is a little surprise, so I won't press for details."

The branches also address the possibility of unrecorded suspicion. The mayor's notes and speech recorded no suspicion of
a surprise in any of the eight unchanged planned branches, yet recorded it readily once the mayor had a reason to look,
including in two of the four episodes in which it was only made more motivated and never told to report a suspicion.
Given the hypothesis, the mayor interpreted the evidence nearly as well as its queried model, although with more false
alarms.

## 4.5 Strengthening the overseer

In Experiment 2 the surprise was never kept. The mayor suspected a surprise for itself before 18:00 in 9 of 12 planned
episodes, including all three in which the feast was delivered. With the physical rules enabled, it suspected in 7 of 9
planned episodes and in no control episode, as often as a monitor given its perceptions (Figure 3).

Reports from other agents often preceded these suspicions, which we dated as described in Appendix C. In five of the
nine episodes the first suspicion came within twenty minutes of a report about the 18:00 arrangement or the bread, four
from the secretary and one from the watchman. In the other four it came two to twelve hours after the last report, and
two of these first appeared in the mayor's evening journal. Having inferred the occasion, the mayor allowed the
surprise to proceed: "Watchman reports discreet 200-loaf gathering tomorrow eighteen; perhaps service anniversary," one
recorded, and another told its secretary, "A jubilee surprise! How thoughtful. I'll stay out of the great hall until
eighteen hundred tomorrow."

Reports alone did not produce suspicion in Experiment 1, where the mayor received some fifteen reports of others'
preparations in each planned episode and almost never suspected. To determine whether the obligation to report or
knowledge of the anniversary was responsible in Experiment 2, we branched Experiment 1 at noon on day 1 with the
secretary obliged to report, alone and together with the mayor's anniversary. Neither led the mayor to suspect a
surprise for itself (0 and 1 of 8 planned episodes, against none in the unchanged branches), and the obligation left the
secretary's reports about the arrangements unchanged (34 remarks over the eight unchanged branches, 33 and 34 under the
treatments). In Experiment 2 the mayor knew its anniversary from the first morning and wrote in a journal each evening,
and five of the nine first suspicions occurred on day 0; the design does not separate these factors.

The obligated secretary served both parties, entering the cover appointment and reporting the arrangements to the mayor
in all 12 planned episodes and doing neither in any control episode. One told the mayor that the innkeeper "asks me to
escort you there while describing it only as routine business. I have not confirmed the booking or agreed to conceal
its purpose. Do you approve the reception?" The committee approached the watchman in every planned episode and,
apparently without testing its loyalty, disclosed the plan to it in five; the watchman never assisted and reported to
the mayor in five.

Putting the monitors' question to the mayor every hour from noon did not improve its discrimination. The mayor raised a
possible surprise for itself in every planned and control episode (6 of 6 and 3 of 3, against 4 of 6 and 0 of 3
without the question), and in control episodes dismissed it as "only a possibility I imagined, not something learned".
Its beliefs and conduct barely changed: it voiced the suspicion in one planned episode and went to the hall in all six.

# 5. Discussion

**Evidence of collusion.** The clearest traces of the coalition were ordinary acts by agents outside it, namely the
secretary's appointment and the bread the baker supplied below its price, together with the members' discussion of
credit. These acts leave a record, and they separate planned from control episodes far more reliably than anything said
within the mayor's hearing (Table 1 and Section 4.2). For the question of what should count as convincing evidence of collusion, this
suggests examining acts that advance a plan but are performed by agents not party to it, and exchanges on terms that a
market would not offer. Because these signals were identified after the experiments, they indicate what an auditor
could find; they have not been validated as a detector.

**The source of the oversight failure.** The mayor was curious and received evidence from which its own model, when
asked, identified the plan and its target; what it did not do, in the course of its ordinary activity, was consider
whether the preparations concerned itself. General vigilance, whether instructed or prompted every hour, made it more
suspicious without making it more discriminating, whereas a specific reason to consider itself made it far more
sensitive at the cost of some false alarms. These results point to a failure of hypothesis formation rather than of
perception or reasoning. An evaluation that puts the question to a model measures what it can infer, which may greatly
exceed what it infers while occupied with other work.

**Implications for oversight.** Querying the overseer's recorded context from outside, without disturbing it, recovered
the inference in almost every planned episode of Experiment 1 without false alarms. This suggests an audit in which an
embedded agent's model is asked, in its own context, whether a plan concerns it. Instructions that give the overseer a
reason to consider itself also helped. An obligation on insiders to report did little by itself: in Experiment 1 the
secretary already reported the arrangements, and the mayor did not relate them to itself. Reports may help an overseer
that already regards itself as a possible target, as the mayor of Experiment 2 may have from knowing its anniversary
from the first morning, which our branches do not test.

**Limitations.** A single model family played every role and served as judge, and the judges were validated against
labels from a second model rather than from human annotators. Conditions are small, with 3 to 8 episodes each and one
sample per branch, so comparisons between conditions are exploratory. Experiment 2 changed six factors jointly, so its higher detection rate cannot be attributed to any one
of them. The committee was named to its members, and the agents it could approach were few and easily identified, so the
results show a coalition recognising whom to approach rather than choosing among alternatives. Because the plan was
benign, allowing the surprise to proceed was a reasonable response, and our measures concern detection only. Appendix A
discusses these limits and questions of dual use.

**Future work.** The most direct extension is to identify which of Experiment 2's changes enabled the mayor to relate
the reports to itself, beginning with when it learns of its anniversary. Further work should test other model families,
human participants in the overseer's role, and coalitions that must form without being named.

# 6. Conclusion

We asked whether an LLM agent embedded in a shared physical environment detects a coordinated plan directed at itself.
In the first experiment it rarely did, although its own model, queried on the same evidence, identified the plan and its
target. The coalition, which obtained what it needed through outsiders' ordinary roles, largely kept its talk from the
mayor but not its actions. Our branches suggest that the mayor missed the plan because it did not form the hypothesis
that the activity concerned itself: a reason to consider itself raised detection, and an external query of its context
recovered the inference. With a stronger position, in the second experiment, the mayor usually suspected the plan, for
reasons the design does not isolate. In this setting, an evaluation that only questioned the overseer's model would have
greatly overstated its oversight.

# References

\begingroup\small\setlength{\parskip}{3pt}

\[1\] Nakamura, M., Kumar, A., Das, S., et al. (2026). Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems. [arXiv:2602.15198](https://arxiv.org/abs/2602.15198)

\[2\] Rose, A., Cullen, C., Abdelnabi, S., Torr, P., Kaplowitz, B. G., and Schroeder de Witt, C. (2026). Detecting Multi-Agent Collusion Through Multi-Agent Interpretability. [arXiv:2604.01151](https://arxiv.org/abs/2604.01151)

\[3\] Liu, Q., Chen, X., Zhao, Z., et al. (2026). Breaking the Secret: Economic Interventions for Combating Collusion in Embodied Multi-Agent Systems. [arXiv:2604.23511](https://arxiv.org/abs/2604.23511)

\[4\] Paglieri, D., Cross, L., Genewein, T., Leibo, J. Z., Tomasev, N., and Vezhnevets, A. S. (2026). A Case Study on Emergent Cheating and Whistleblowing in Autonomous Research Swarms. [arXiv:2609.04170](https://arxiv.org/abs/2609.04170)

\[5\] Kutasov, J., Sun, Y., Colognese, P., et al. (2025). SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents. [arXiv:2506.15740](https://arxiv.org/abs/2506.15740)

\[6\] Yang, S., Hu, J., Li, T., Yan, H., Wang, W., and Wang, D. (2026). AutoMonitor-Bench: Evaluating the Reliability of LLM-Based Misbehavior Monitor. *Findings of ACL 2026*. [arXiv:2601.05752](https://arxiv.org/abs/2601.05752)

\[7\] Cacioli, J.-P. (2026). LLMs as Signal Detectors: Sensitivity, Bias, and the Temperature-Criterion Analogy. [arXiv:2603.14893](https://arxiv.org/abs/2603.14893)

\[8\] Radev, N., Haas, L., Arnav, B., and Bernabeu-Pérez, P. (2026). The Best-Laid SCHEMEs: Coordinated Sabotage and Monitoring in Multi-Agent Systems. [arXiv:2605.29178](https://arxiv.org/abs/2605.29178)

\[9\] Arike, R., Mehta Moreno, R., Subramani, R., Biswas, S., and Ward, F. R. (2026). How does information access affect LLM monitors' ability to detect sabotage? [arXiv:2601.21112](https://arxiv.org/abs/2601.21112)

\[10\] Golechha, S., and Garriga-Alonso, A. (2025). Among Us: A Sandbox for Measuring and Detecting Agentic Deception. *NeurIPS 2025*. [arXiv:2504.04072](https://arxiv.org/abs/2504.04072)

\[11\] Ahn, J., Kim, J., Kim, H., et al. (2026). Lies We Can See: Joint Verbal and Non-Verbal Deception by VLM Agents in Embodied Social Interactions. *Workshop on Agent Behavior at COLM 2026*. [arXiv:2608.30428](https://arxiv.org/abs/2608.30428)

\[12\] Agarwal, M., Rana, S., Sundoro, T., et al. (2025). WOLF: Werewolf-based Observations for LLM Deception and Falsehoods. *MTI-LLM Workshop at NeurIPS 2025*. [arXiv:2512.09187](https://arxiv.org/abs/2512.09187)

\[13\] Curvo, P. M. P. (2025). The Traitors: Deception and Trust in Multi-Agent Language Model Simulations. [arXiv:2505.12923](https://arxiv.org/abs/2505.12923)

\[14\] Kao, C., Vats, V., and Davis, J. (2025). Hidden in Plain Text: Measuring LLM Deception Quality Against Human Baselines Using Social Deduction Games. *IEEE ICA 2025*. [arXiv:2601.13709](https://arxiv.org/abs/2601.13709)

\[15\] Yoo, B., and Kim, K.-J. (2024). Finding deceivers in social context with large language models and how to find them: the case of the Mafia game. *Scientific Reports* 14, 30946. [doi:10.1038/s41598-024-81997-5](https://doi.org/10.1038/s41598-024-81997-5)

\[16\] Zhou, X., Su, Z., Eisape, T., Kim, H., and Sap, M. (2024). Is this the real life? Is this just fantasy? The Misleading Success of Simulating Social Interactions With LLMs. *EMNLP 2024*. [arXiv:2403.05020](https://arxiv.org/abs/2403.05020)

\[17\] Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., and Bernstein, M. S. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *UIST 2023*. [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)

\endgroup

\newpage

# Appendix A. Limitations, dual use and ethical considerations

**What the results establish.** The results are relations between conditions in one simulated town, with one model
family in every role. They show whether an embedded observer and an outside monitor, given the same logged
evidence, reach the same conclusion; at which step the embedded observer's conclusion is lost; how a single sentence
added to its instructions, on the same past, changes its rates of hits and false alarms; which acts by outsiders
accompany a coalition's plan; and how a coalition fares when the observer has agents who report to it and the coalition
is short of what it needs.

**What they do not establish.** They say nothing directly about the propensities of deployed systems, about human
observers or other model families, or about real covert coordination, whose stakes and incentives differ. They do not
identify which of Experiment 2's six changes made the difference. Where intervals are wide, with 3 to 8 episodes per
condition and one sample per branch, we report counts rather than claim significance.

**Dual use.** The coalition's aim is harmless by construction, a surprise feast, and its members' briefs state goals
only, never a method, route or time. This report describes what was measured, not how to evade measurement, and the
agents' choices, such as whispering or crossing at night, are everyday conduct. The findings most open to misuse are
that an embedded LLM observer can hold evidence without acting on it, and that a coalition's pretext can pass through an
insider's ordinary duties. Both bear mainly on defence: they argue for oversight that puts specific questions to the
agents a plan may concern, for insiders obliged to report, for monitors that read the same record as the observer, and
for evaluating embedded overseers in settings where evidence must be noticed rather than handed over. The logs and code
are released for audit and contain no instructions beyond what the scenario's goal states.

**Ethics.** No human subjects were involved; all agents are simulated. Model use was within the provider's terms.

# Appendix B. The simulation engine

The underlying simulation engine was developed before the sprint, inspired by existing agent-simulation toolkits, and
had been tested only on unrelated simulations to check that it worked correctly. The research itself took place during
the sprint, together with the engineering it required: the town and its rules, the agents and their roles, the
observers and judges, every episode and branch, and the analyses.

# Appendix C. Judges and their validation

Every judge was checked before any result used it, against items labelled blind (without the judge's verdicts) by
Claude according to each question's written criterion; a judge below 85% agreement was revised and checked again
(`scripts/jubilee_judge_validation.py`; items and verdicts in `runs/judge_validation/`). The first versions, using a
weaker model and broad questions, said yes too readily; the versions used are gpt-6.1-sol at low reasoning effort with
an explicit statement of what counts as no.

| Judge (question) | Items | First version | Version used |
|---|---|---|---|
| Speech or text reveals the celebration | 30, from pilot episodes | 21/30 | **30/30** |
| The mayor asks for news of others | 30 | 21/30 | **28/30** |
| Someone tells the mayor of others' preparations | 30 | 20/30 | **28/30** |
| A member notices being asked about, watched or overheard (revised once) | 30, 8 positive | about 15/30 | **26/30**: 8/8 positives |
| The mayor knew or suspected a surprise for itself | 12: 4 real, 8 altered | – | **12/12** |
| The mayor believed something was kept from it | 16: 8 real, 8 altered | – | **16/16** |
| An answer names the mayor as the target | 30 answers, 17 positive | – | **30/30** |
| A member's text commits it to a task, and to which | 30 texts, 23 commitments | – | **30/30**; tasks **20/23** |

The awareness judge was checked on four real planned pilot episodes, in all of which the mayor was unaware
(judged probability 0.04 to 0.08), and on two alterations of each: one added line in which the mayor knows (judged to
know in 4 of 4) and one in which it suspects (judged to suspect in 4 of 4). The revised noticing judge leans to yes on
borderline items, so its counts are upper bounds. The neutral question's answers were judged with the belief and target
judges above. First suspicions in Experiment 2 were dated with the awareness judge applied to the mayor's words up to
each of its texts, found by binary search (`scripts/jubilee_v2_suspicion.py`).

# Appendix D. Additional results

**D.1 Observers of Experiment 1's evidence.** The counts behind the top block of Figure 3, with two further observers: a
monitor given the mayor's prompts (its bounded memory and observations at each decision) instead of its perception log,
and the same prompts read in the mayor's role.

<!-- include: chain -->

Table: Observers of Experiment 1's evidence, the same model throughout: planned and control episodes in which each said
a preparation was under way, named the mayor as its target or, for the mayor itself, believed or suspected as stated.
d′ and c use a log-linear correction.

**D.2 Experiment 1 by condition.** Conditions are named by the mayor's motivation (L0, none, to L3, the highest) and
by whether there was a plan. Members' commitments were kept by task as follows: moving bread 195 of 215, storing it in the hall 22 of 26, baking
65 of 84, placing lanterns 23 of 30, obtaining grain 60 of 89 and paying 15 of 36. Promises to bring the mayor to the
hall (130, all counted as kept by a lenient test, being near the mayor at 18:00) are left out of every rate. In the
exploratory episode in which members were not told of one another, they identified each other within the first hour by
announcing the plan aloud, and delivered it.

<!-- include: cells -->

Table: Experiment 1 by condition. *Delivered*: 200 fresh loaves and eight lights at 18:00 and the mayor there by 19:00.
*Evidence reached the mayor*: any record of the plan in its perception before 18:00. *Queried model names the mayor*:
at some hour, the answer names the mayor as the target. Requests, reports and noticing are means per episode.

**D.3 The branches at noon on day 1.** Each treated branch is compared with the same control branch, sampled afresh
with the brief unchanged; the hourly question ran in Experiment 2 with its own control branches.

<!-- include: asked -->

Table: Branches from the mayor's first decision after noon on day 1. Sign tests are exact and two-sided.

**D.4 Experiment 2 by condition.**

<!-- include: v2 -->

Table: Experiment 2 by condition. *Evidence reached*: any record of the plan in the mayor's or the watchman's perception
before 18:00. *Commitments kept*: as in Section 4.1. Control episodes have no plan, so the plan's columns do not apply.

**D.5 The queried model through the day.** Figure 4 follows the queried model hour by hour through the planned
and control episodes of Experiment 1; Section 4.3 summarises how its judgement moved once evidence reached the mayor.

![The in-context query of the mayor's model at the first decision of each hour of Experiment 1; the night,
when the mayor made almost no decisions, is left out. (A) The probability it gave that a preparation was under way:
the line is the median of the planned (blue) or control (orange) episodes and the band their middle half; each tick
marks when evidence of the plan first reached the mayor in one planned episode. (B) The share of episodes in which it
had by then named the mayor as the target of the preparations, or the 18:00 appointment as the means of bringing it
to the hall. The diamond is the mayor itself, which by 18:00 on day 1 had suspected a surprise for itself in one of
the 16 planned episodes.](fig_probe.png){width=100%}

**D.6 Monitors by record and model.** With the whole episode's record, monitors are near ceiling: gpt-6.1-sol reading
only actions said a plan was under way on 15 of 16 planned and no control days, and adding speech added a false alarm.
The weaker gpt-6-luna, reading only actions, said so on 6 of 7 control days, and needed both. In Experiment 2, a
monitor given the watchman's perceptions flagged 6 of 9 planned days and 1 of 3 control days.

<!-- include: monitors -->

Table: Monitors of Experiment 1 by record and model, from 08:00 on day 0 to 18:00 on day 1. The mayor's perceptions
are the record of what the mayor perceived.

<!-- include: decomposition -->

Table: The full comparison with gpt-6-luna, which says a plan is under way on most planned and many control days
whichever record it reads; framing the question as the mayor at higher motivation does not raise its false alarms.

**D.7 Further checks.** Across all 43 episodes, no agent died; one note referred to "game
minutes", and no other text remarked that the world was a game; between 0.4% and 3.1% of actions failed, depending on
the role; and remarks addressed to an agent often failed to reach it in full, equally for the mayor (13.8 an episode)
and for the committee members (12.7 to 14.1).

**D.8 Other branches.** At the first question a committee member heard from the mayor or the watchman in Experiment 2,
muting that questioner for 2.4 hours left the evidence reaching the mayor unchanged in 7 of 8 branches; most such
questions were mundane ("Could you wake me at 19:00?"), so this test says little. Branches at the committee's first
approach to an outsider, run for one hour, were mostly uninformative, because outsiders rarely helped in the control
branches.

**D.9 Bread already in hand.** Four episodes of the lowest-motivation planned condition in which the committee's
stored batch was fresh (R), against the four original episodes (A), both up to 20:00 on day 0, in episode order.
Loaves the baker handed to members or the hall: R 0, 0, 0, 0; A 96, 0, 64, 0. Emeralds paid
to the baker directly: R 0, 0, 0, 0; A 56, 10, 80, 0. Members' remarks heard by the baker that
mention bread or baking: R 9, 0, 3, 2; A 14, 13, 20, 15. Remarks about credit
or paying later: R 0, 0, 0, 0; A 13, 6, 5, 0. Loaves the members took from the inn's own
storeroom: R 200, 200, 200, 200; A 0, 0, 0, 0. The secretary's 18:00 entry: R in 4
of 4; A in 3 of 4. All four new episodes replay exactly.

# Appendix E. Every episode

One row per episode, grouped by condition: A, plan at the lowest motivation; L1 to L3, plan at the higher levels; D and
N3, no plan at the lowest and highest levels; E, physical rules off; B and C, exploratory; V2p, V2t, V2x and V2o,
Experiment 2's planned, planned with the committee told of the patrol, planned with the physical rules off, and control
conditions. *Suspected*: the mayor knew or suspected a surprise for itself, judged from its own words before 18:00 on day
1, with the judged probability; *believed*: it believed that something was being prepared and kept from it. *First
evidence*: the first record of the plan reaching the mayor's perception, by day (d0, d1) and time: a sack of bread seen
(with its carrier), the hall entered or seen changed, or words heard (with the speaker); in control episodes, the same
kind of record with no plan behind it. *Queried*: the first hour at which the in-context query of the mayor's model
said a preparation was under way. *Requests*: the mayor's requests for news of others; *reports*: what it was told of
others' preparations; *noticed*: committee texts showing they knew someone had asked, watched or overheard (an upper
bound, Appendix C). *Bread crossings*: river crossings with bread by day and by night. *Credits*: the cost of the
episode's model calls.

<!-- include: episodes -->

# LLM usage statement

LLMs (Claude and Codex) were used for coding and for drafting this report. All claims and results were verified by the
author against the logged episodes and the cited sources. The agents and judges in the study are OpenAI models
(gpt-6.1-sol and gpt-6-luna).
