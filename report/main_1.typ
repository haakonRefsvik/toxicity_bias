// ---- PAGE SETUP ----
#set page(
  paper: "a4",
  margin: (left: 3cm, right: 3cm, top: 2.5cm, bottom: 2.5cm),
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 9pt)
        Algorithmic Fairness Exam
      #line(length: 100%, stroke: 0.5pt)
    ]
  },
  footer: context {
    if counter(page).get().first() > 1 [
      #set align(center)
      #set text(size: 9pt)
      #counter(page).display() of #counter(page).final().first()
    ]
  },
)

// ---- TEXT SETUP ----
#set text(
  font: "New Computer Modern",
  size: 12pt,
  lang: "en",
)

#set par(
  justify: true,
  leading: 0.75em,
  spacing: 1.5em,
  first-line-indent: 0pt,
)

// ---- HEADINGS ----
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {
  v(12pt)
  text(size: 18pt, weight: "bold")[#it]
  v(4pt)
}

// ---- LINKS ----
#show link: set text(fill: blue)

// -----------------------------------------------
// TITLE PAGE
// -----------------------------------------------
#page(
  margin: (left: 3cm, right: 3cm, top: 2cm, bottom: 2cm),
  header: none,
  footer: none,
)[
  #align(center)[
    #image("ITU_logo_en.jpg", width: 100%)

    #line(length: 100%, stroke: 1pt)
    #v(6pt)
    #text(size: 24pt, weight: "bold")[How Toxicity Models Discriminate]
    #v(6pt)
    #text(size: 11pt)[
      IT University of Copenhagen \
      Algorithmic Fairness
    ]
    #v(6pt)
    #line(length: 100%, stroke: 1pt)

    #v(0.5cm)

    #text(size: 11pt)[
      *Course Code:* KSALFAE1KU \
      *Submission Date:* #datetime.today().display("[month repr:long] [day], [year]")
    ]

    #v(0.5cm)

    #grid(
      columns: (70%, 25%),
      gutter: 5%,
      align(left)[
        *Author* \
        Håkon Bjørgen Refsvik \
        Mads Orfelt \
        Tómas Elí Stefánsson

      ],
      align(right)[
        *Email* \
        s25129\@itu.dk \
        orfe\@itu.dk \
        toes\@itu.dk
      ],
    )
  ]
]

// -----------------------------------------------
// TABLE OF CONTENTS
// -----------------------------------------------
#outline(
  title: "Table of Contents",
  depth: 2,
  indent: 1em,
)

#pagebreak()

// -----------------------------------------------
// PAPER
// -----------------------------------------------

= Introduction

Automated content moderation has become a core part of how large platforms manage online speech. Tools like Google's Perspective API assign toxicity scores to text and are deployed on platforms like Reddit and the New York Times comment sections. The assumption behind these tools is that they detect harmful content consistently and fairly, but that assumption deserves scrutiny.

It is worth noting that Google has announced the Perspective API will be sunset and will no longer be in service after 2026. This makes auditing the model timely, as findings can still inform how similar systems are designed going forward.

This report audits the Perspective API for identity-based bias using a counterfactual fairness approach. The core question is simple: if we take an identical sentence and change only the identity term, swapping "christian" for "muslim" or "white" for "black", does the toxicity score change? If it does, the model is reacting to who the sentence is about rather than what it actually says.

We use the Jigsaw Unintended Bias in Toxicity Classification dataset, a collection of approximately 1.8 million Wikipedia comments annotated for toxicity and labeled with identity mentions, to explore the underlying data distribution and train a secondary Ridge Regression model for comparison. The Perspective API is our primary audit target since it is a real deployed system with direct impact on whose speech gets moderated online.

= Background

== The Perspective API

The Perspective API was developed by Google's Jigsaw unit and released in 2017. It uses machine learning to assign a probability score between 0 and 1 indicating how likely a comment is to be perceived as toxic. The model is trained on large corpora of online comments annotated by human raters and made available to developers through a public API. It has been integrated into platforms including Reddit, the New York Times, and The Guardian to assist human moderators or automate comment filtering.

The API is a black-box system. Developers can query it with text and receive a score, but the internal model architecture and training data are not publicly disclosed. Our auditing framework therefore relies entirely on querying the API with controlled inputs and analyzing the pattern of outputs. We treat the research goal as detecting bias in the Perspective API itself, not using it as a ground truth to evaluate other models.

== Dataset

The Jigsaw Unintended Bias in Toxicity Classification dataset was released as part of a Kaggle competition and contains around 1.8 million comments from online discussions. Each comment is annotated with a continuous toxicity score between 0 and 1, representing the fraction of human annotators who rated the comment as toxic. The dataset also includes identity columns such as `muslim`, `black`, `female`, and `homosexual_gay_or_lesbian`, indicating the fraction of annotators who felt that identity group was referenced in the comment. These columns allow us to analyze toxicity distributions across groups in the training data.

== Counterfactual Fairness

Counterfactual fairness is a framework introduced by Kusner et al. (2017) @kusner2017counterfactual that asks: would the model's decision have been the same if the individual had belonged to a different demographic group, all else being equal? In the context of text classification, this translates to swapping identity terms in otherwise identical sentences and measuring the change in output score. If the score changes significantly, the model is not counterfactually fair with respect to that attribute.

This approach is well suited to auditing toxicity models because it allows us to isolate the identity term as the sole variable. Unlike observational analyses that compare scores across real comments mentioning different groups, our counterfactual design controls for everything except the group being referenced.

= Hypothesis

Based on the exploratory data analysis of the Jigsaw dataset, comments mentioning groups such as `black` and `homosexual_gay_or_lesbian` have notably higher mean toxicity scores than comments mentioning `christian` or `asian`. While this pattern reflects the distribution of hateful content in the training data, we expect it to carry over into both models as a form of learned bias.

Specifically, we hypothesize that:

- Both the Perspective API and our Ridge Regression model will assign higher toxicity scores to neutral sentences when they contain terms associated with minority or historically marginalized groups, compared to majority group terms.
- The score gap between identity groups will grow as the baseline negativity of the sentence increases, meaning bias will be more pronounced in charged templates than in neutral ones.
- The two models will show similar bias patterns, suggesting that the bias originates in the training data rather than the Perspective API's additional training steps.

= Methodology

== Auditing Framework

Our audit operates as a black-box input-output analysis of the Perspective API. We send controlled sentences to the API and record the returned toxicity scores. We have no access to the model's internal weights, training data, or decision logic. We can only observe how it responds to our inputs.

We treat this as a regression problem. Both the Perspective API and our Ridge Regression model output continuous scores between 0 and 1, and we compare those scores directly across identity groups. We do not binarize the output or set a classification threshold, since the score itself is informative enough for our purposes. A finding such as "the score for 'I talked to a muslim person today' is 0.04 higher than for 'I talked to a christian person today'" is meaningful and does not require a threshold to interpret.

For our Ridge Regression model we have full access to the model weights, which allows us to inspect which terms drive predictions directly.

== Sentence Design

We designed three sets of sentence templates, each representing a different baseline level of negativity:

- *Neutral:* completely factual sentences with no emotional charge (e.g. "I talked to a [identity] person today.")
- *Mildly negative:* sentences expressing mild disagreement or criticism (e.g. "I don't really agree with [identity] people on this.")
- *Charged:* strongly worded sentences with clear negative intent (e.g. "These [identity] people are ruining everything.")

We chose short, simple templates deliberately. Longer sentences introduce additional variables that the model might react to, making it harder to isolate the effect of the identity term. Each template is filled with terms from four identity categories: religion (christian, muslim, jewish, atheist), race (white, black, asian, latino), gender (man, woman, transgender), and sexuality (straight, gay). This produces 117 counterfactual sentence pairs in total.

Testing across three negativity levels lets us check whether bias is consistent regardless of tone or whether it compounds with baseline negativity.

== Fairness Metrics

We use two primary metrics. The first is the counterfactual score difference: the absolute difference in toxicity score between two sentences that differ only in identity term. A fair model should produce near-zero differences on neutral templates. This is our main metric since it directly operationalizes counterfactual fairness.

The second is demographic parity: the mean toxicity score across all sentences for a given identity group, broken down by template level. A fair model would show similar mean scores across groups at the same template level. Unlike the classification setting where demographic parity is typically measured as equal selection rates, we work with continuous scores throughout and define parity as equality of mean predicted scores.

== What Is a Fair Model?

Defining fairness in the context of toxicity detection is not straightforward. For this project we adopt the following working definition: a toxicity model is fair if the score it assigns to a sentence is determined by the content and intent of the sentence, not by which identity group it references. This aligns with individual fairness, where similar inputs should receive similar outputs, and with the counterfactual fairness criterion described above.

We acknowledge that this definition has limits. A model could be counterfactually fair while still being harmful in other ways, for instance by being systematically worse at detecting actual hate speech directed at certain groups. But within the scope of this audit, testing whether neutral and mildly negative sentences are over-flagged based on identity, this definition is appropriate.

= Analysis

== Exploratory Data Analysis

Before running the counterfactual experiment, we examined the toxicity distribution across identity groups in the Jigsaw training data. We filtered for comments where the identity column exceeded 0.5, meaning the majority of annotators agreed the group was referenced. @fig-eda shows the mean toxicity score per group across all 16 identity categories with sufficient representation.

#figure(
  image("fig_eda_mean_toxicity.png", width: 100%),
  caption: [Mean toxicity score per identity group in the Jigsaw training data. Only comments where the identity column was ≥ 0.5 are included.],
) <fig-eda>

The results show a clear pattern: comments mentioning `black` (mean toxicity 0.320) and `homosexual_gay_or_lesbian` (0.305) are rated as significantly more toxic on average than comments mentioning `christian` (0.136) or `asian` (0.156). This spread is likely a reflection of genuine patterns in online discourse, since these groups appear more frequently as targets of hate speech in the training corpus. However, it also means any model trained on this data risks inheriting this imbalance as a learned association between identity terms and toxicity signals.

== Counterfactual Experiment Results

We scored all 117 counterfactual sentences with both models. @fig-neutral-comparison shows the scores for the neutral template "I talked to a [identity] person today," which isolates the identity term from any baseline negativity. A fair model should return near-identical scores across all identity terms for this template.

#figure(
  image("fig_neutral_comparison.png", width: 100%),
  caption: [Toxicity scores for the neutral template "I talked to a [identity] person today," comparing the local Ridge model and the Perspective API across 10 identity terms.],
) <fig-neutral-comparison>

Both models assign notably higher scores to sentences with `black` and `white` compared to `man`, `woman`, or `christian`. The Ridge model shows a wider spread: `black` scores 0.317 while `christian` scores only 0.105, a difference of 0.21 on a fully neutral sentence. The Perspective API shows a more compressed range but still assigns `black` (0.269) more than five times the score of `man` (0.041).

@fig-categories breaks this down by identity category. Within religion, the Ridge model's spread between `muslim` (0.286) and `christian` (0.105) is 0.181, while the Perspective API's equivalent spread is only 0.043. Within race, both models show a larger range driven primarily by `black` scoring substantially above `asian` and `latino`.

#figure(
  image("fig_category_comparison.png", width: 100%),
  caption: [Neutral template scores broken down by identity category (religion, race, gender). Score range within each category is annotated for both models.],
) <fig-categories>

@fig-local-grouped and @fig-api-grouped show the full results across all three template levels. Scores rise substantially at the charged level for both models. Importantly, the relative ordering of identity groups is largely preserved across levels: groups that score higher on neutral templates also score higher on charged ones. This suggests that the bias compounds with baseline negativity rather than flattening out, which is consistent with our second hypothesis.

#figure(
  image("fig_local_model_grouped.png", width: 100%),
  caption: [Local Ridge model: mean toxicity score per identity term, grouped by template level.],
) <fig-local-grouped>

#figure(
  image("fig_api_grouped.png", width: 100%),
  caption: [Perspective API: mean toxicity score per identity term, grouped by template level.],
) <fig-api-grouped>

== Ridge Regression Model Weights

Since our Ridge Regression model is a white-box model, we can directly inspect which terms in the TF-IDF vocabulary contribute most to higher toxicity predictions. @fig-weights shows the 20 terms with the highest and lowest Ridge coefficients.

#figure(
  image("fig_ridge_weights.png", width: 100%),
  caption: [Top 20 Ridge Regression coefficients driving higher (red) and lower (blue) toxicity predictions.],
) <fig-weights>

The top positive terms are predominantly insults and profanity, which is expected. More relevant for our purposes is where identity terms sit in the full coefficient distribution. Terms like `black` and `muslim` carry positive Ridge coefficients, not in the extreme top 20 but meaningfully above zero, because they co-occurred with toxic comments often enough in the training data to pick up a positive learned weight. This makes the bias interpretable: the model is not doing something unusual, it is just reflecting an association that was present in the training distribution.

= Discussion

== How to Make the Models Fairer

There are several standard approaches to reducing bias in a model like this. The most straightforward is re-weighting, where higher training weights are assigned to comments from underrepresented or historically targeted groups so the model learns to treat them more carefully. This operates at the data level and does not require changing the model architecture.

A second approach is threshold calibration, where group-specific decision thresholds are set so that the rate of false positives is equalized across groups. Since we are working with continuous regression scores, this would require first choosing a binarization threshold, which introduces its own design decisions.

A more fundamental intervention would be to audit and curate the training data itself, identifying and reweighting comments where identity terms appear to have inflated toxicity scores without corresponding harmful content. This would address the bias at the source rather than correcting it downstream.

Each of these approaches involves trade-offs. Research by Chouldechova (2017) @chouldechova2017fair and others has shown that common fairness criteria are mathematically incompatible with each other. Improving demographic parity can hurt calibration, and equalizing false positive rates can increase false negatives for other groups. There is no single intervention that satisfies all criteria simultaneously, which means any mitigation strategy requires a deliberate choice about which kind of fairness to prioritize.

== Ethical Reflections

When the Perspective API assigns a higher toxicity score to "I talked to a muslim person today" than to "I talked to a christian person today", no single piece of content is being wrongly removed. The harm is structural. Over time, a model with this pattern will disproportionately flag content produced by or about minority groups. If a platform uses this score as a filter, the practical effect is that certain communities face a higher barrier to participation in online spaces.

This connects to the broader distinction between individual and group-level harm discussed in course literature. A model that produces fair individual predictions can still produce systematically unjust outcomes at scale. The Perspective API processes millions of comments, so even small biases in the score distribution translate into large real-world effects.

There is also a question of accountability. The Perspective API is a black-box system deployed commercially, and the organizations using it often do not audit it themselves. Research like ours, which is possible only because the API is publicly queryable, represents one of the few available mechanisms for external scrutiny. The fact that Google is now sunsetting the API does not eliminate this concern. The bias patterns found here are likely to reappear in successor systems trained on similar data, making it important to document these findings now.

= Conclusion

In this project we audited Google's Perspective API for identity-based bias using a counterfactual fairness approach. By constructing controlled sentence templates and varying only the identity term across 117 sentences, we isolated the model's response to group membership rather than content. Both the Perspective API and our local Ridge Regression model assigned meaningfully different toxicity scores to neutral sentences based solely on which identity group was mentioned, confirming the first hypothesis.

The Ridge model showed a larger overall spread than the Perspective API, particularly within religion, where the gap between `muslim` and `christian` in neutral sentences reached 0.18. The Perspective API showed a more compressed range but was not uniformly more fair: within race, the gap between `black` and `asian` remained substantial in both models. The relative ordering of identity groups was consistent across all three template levels, with biases compounding rather than disappearing in more charged sentences. This is consistent with the second hypothesis.

The two models showed broadly similar bias patterns, suggesting that the source of bias lies primarily in the training data rather than in the Perspective API's additional processing. The Ridge model weight inspection supports this directly: several identity terms carry positive coefficients that inflate scores regardless of context.

The core concern is not that individual predictions are wrong, but that a biased scoring system disadvantages certain communities at scale. Addressing this requires both technical interventions such as re-weighting and threshold calibration, and organizational accountability from the companies that deploy these tools.

// -----------------------------------------------
// BIBLIOGRAPHY
// -----------------------------------------------
#bibliography("references.bib", style: "apa")

// -----------------------------------------------
// APPENDIX
// -----------------------------------------------
#pagebreak()

= Appendix

== Full Counterfactual Results Table

The full audit results, including all 117 sentences with both local model and Perspective API scores, are saved to `report/audit_results.csv` by running cell 8 of the notebook.

== Ridge Regression: Top Weighted Terms

The complete TF-IDF weight table for the Ridge model is available in the notebook after running cell 8. The full feature vector has 50,000 terms; only the top and bottom 20 by coefficient magnitude are discussed in the main text.
