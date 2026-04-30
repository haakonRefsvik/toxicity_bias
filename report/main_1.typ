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
    #text(size: 24pt, weight: "bold")[Biases in toxicity ratings amongst different social groups]
    #v(6pt)
    #text(size: 11pt)[
      IT University of Copenhagen \
      Alorithmic Fariness
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

= Theory

==

==

= Analysis

= Discussion

= Conclusion

// -----------------------------------------------
// BIBLIOGRAPHY
// -----------------------------------------------
#bibliography("references.bib", style: "apa")
