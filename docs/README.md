# Documentation

This directory contains all documentation for the ComfyUI Quadtree Tile project.

## Directory Structure

```
docs/
├── README.md                       # This file
├── PARAMETER_TUNING_GUIDE.md      # User guide for tuning parameters
├── bug-analysis/                  # Historical bug analysis
├── implementation/                # Implementation guides and technical details
├── reports/                       # Fix reports and QA reports
└── research/                      # Research and evaluation documents
```

## Quick Navigation

### For Users
- **[Parameter Tuning Guide](PARAMETER_TUNING_GUIDE.md)** - Learn how to optimize quadtree diffusion parameters for your use case

### For Developers

#### Implementation Documentation
- **[Implementation Guide](implementation/IMPLEMENTATION_GUIDE.md)** - Overview of the quadtree diffusion implementation
- **[Gradient Metric Implementation](implementation/GRADIENT_METRIC_IMPLEMENTATION.md)** - Details on gradient-based subdivision metrics
- **[Square Quadtree Implementation](implementation/SQUARE_QUADTREE_IMPLEMENTATION_REPORT.md)** - Report on square quadtree tile implementation

#### Research & Evaluation
- **[Quadtree Research Report](research/QUADTREE_RESEARCH_REPORT.md)** - Comprehensive research on quadtree approaches
- **[Research Summary](research/RESEARCH_SUMMARY.md)** - Summary of research findings
- **[Visual Comparison](research/QUADTREE_VISUAL_COMPARISON.md)** - Visual comparisons of different approaches
- **[Variance Metrics Evaluation](research/VARIANCE_METRICS_EVALUATION.md)** - Evaluation of variance-based metrics

#### Bug Analysis (Historical)
- **[Skip Tile Velocity Bug](bug-analysis/SKIP_TILE_VELOCITY_BUG.md)** - Complete analysis of the film negative bug in skipped tiles (consolidated from 3 separate analyses)

#### Reports
- **[Changes Summary](reports/CHANGES_SUMMARY.md)** - Summary of changes made to the codebase
- **[Gaussian Variance Fix Report](reports/FIX_REPORT_gaussian_variance.md)** - Report on Gaussian variance fix
- **[Coverage Gap Fix QA Report](reports/QA_REPORT_COVERAGE_GAP_FIX.md)** - QA report for coverage gap fix

## Document Categories

### Implementation Guides
Technical documentation describing how various features are implemented. Essential reading for developers who want to understand or modify the codebase.

### Research Documents
Research findings, evaluations, and comparisons that informed implementation decisions. Useful for understanding the rationale behind design choices.

### Bug Analysis
Detailed analyses of historical bugs, their root causes, and resolutions. Preserved for learning and reference purposes.

### Reports
Summaries of fixes, QA testing results, and change logs. Useful for tracking project evolution and understanding what has changed.

## Contributing Documentation

When adding new documentation:

1. **Choose the right category**:
   - Implementation details → `implementation/`
   - Research or evaluation → `research/`
   - Bug analysis → `bug-analysis/`
   - Fix or QA reports → `reports/`
   - User-facing guides → root of `docs/`

2. **Use descriptive filenames**:
   - Good: `GRADIENT_METRIC_IMPLEMENTATION.md`
   - Bad: `doc1.md`

3. **Update this README**: Add a link to your new document in the appropriate section

4. **Follow markdown conventions**:
   - Use clear headings
   - Include code examples where relevant
   - Add links to related documents
   - Include date and author information for technical analyses
