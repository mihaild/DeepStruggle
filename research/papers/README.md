# Papers — what the literature did, and what this project took from it

Reading notes, not summaries. Each file records what the paper actually does and which of its
choices this project adopted, rejected, or has not tested — so a design argument can cite a
precedent rather than an impression.

| paper | the part this project uses |
|:---|:---|
| [`paper_nash_policy_gradient.md`](paper_nash_policy_gradient.md) | NashPG itself: the PPO-style loss with a KL penalty against a frozen reference policy |
| [`paper_magnetic_mirror_descent.md`](paper_magnetic_mirror_descent.md) | why regularising toward a reference converges where self-play cycles |
| [`paper_kl_regularized_search.md`](paper_kl_regularized_search.md) | the one-step improvement `π′(a) ∝ π(a)·exp(Q̂(a)/τ)` that P3's expert iteration would distil |
| [`paper_deepnash_rnad.md`](paper_deepnash_rnad.md) | regularised Nash dynamics at scale, in a game with hidden information |
| [`paper_student_of_games.md`](paper_student_of_games.md) | search in imperfect information without a public belief state |
| [`paper_rebel.md`](paper_rebel.md) | belief-state search, the principled alternative to determinization |
| [`paper_off_belief_learning.md`](paper_off_belief_learning.md) | why naive determinization suffers strategy fusion, which `ai/search/dmcts.py` inherits |
| [`paper_cicero_diplomacy.md`](paper_cicero_diplomacy.md) | anchoring a policy to human play so it stays legible |
| [`paper_alphastar.md`](paper_alphastar.md) | league play and opponent pools — the ancestor of `--opponent-self-pool` |
| [`paper_suphx_mahjong.md`](paper_suphx_mahjong.md) | oracle-guided training, and how the privileged critic is weaned off |
| [`paper_douzero_perfectdou.md`](paper_douzero_perfectdou.md) | perfect-information distillation in a card game, the shape P3 proposes |

Where a paper's result was tested here rather than assumed, the test lives in
[`../findings/`](../findings/README.md) and the arm is registered in [`../runs.md`](../runs.md).
