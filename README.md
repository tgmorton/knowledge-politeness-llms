# Grace Project

Research project analyzing politeness in speaker responses.

## Directory Structure

```
.
├── README.md                 # This file
├── data/                     # Raw and processed data files
│   ├── study1.csv
│   ├── study1_adjusted.csv
│   ├── study1_answered_20241031_104324.csv
│   └── study2.csv
├── scripts/                  # Python scripts
│   ├── prompt.py
│   └── query_openai.py
└── Analysis/                 # Analysis files and outputs
    ├── R_scripts/           # R analysis scripts
    │   ├── Study1.R
    │   └── Study2.R
    ├── data/                # Analysis-specific data files
    │   ├── study1_annotated.csv
    │   ├── study1_annotated.xlsx
    │   └── updated_file.csv
    ├── visualizations/      # Plots and figures
    │   ├── Rplot01.png
    │   ├── Rplot02.png
    │   ├── Rplot03.png
    │   ├── Rplot04.png
    │   ├── Rplot05.png
    │   ├── Rplot06.png
    │   └── response_visualization.pdf
    └── output/              # Analysis outputs and reports
        ├── 02_polite_speaker_ana.Rmd
        ├── analyses_section.md
        ├── procedures_section.tex
        └── Rhistory.txt
```

## Description

This project contains:
- **Data**: Study datasets (study1 and study2) with various versions
- **Scripts**: Python scripts for data processing and OpenAI API queries
- **Analysis**: R-based statistical analysis and visualization
