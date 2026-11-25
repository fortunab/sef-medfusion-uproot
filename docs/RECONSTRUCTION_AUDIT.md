# Reconstruction audit

The supplied PDF has 16 pages. Its ZIP contains LaTeX, a bibliography, and screenshots, with no executable implementation, model checkpoints, split files, random seeds, or raw metric records. Reconstruction on public data was explicitly authorized. The manuscript was rewritten to describe the implementation and measurements delivered here.

| Initial claim or ambiguity | Delivered treatment |
| --- | --- |
| 96% accuracy and comparisons with six CNN, MIL, and transformer systems | Replaced with actual, common-protocol measurements of three fully specified compact CNNs. No cross-task state-of-the-art ranking is retained. |
| 87.3% adversarial detection and 6.1% FPR | Replaced with per-image attack evaluations, clean calibration, three seeds, and explicit successful-attack denominators. |
| Gradient loss label unspecified at inference | Uses a detached predicted label, requiring no ground-truth label when serving predictions. |
| Maximum gradient denominator unspecified | Fits it once on a disjoint clean calibration partition and clips normalized values. |
| Proxy selection direction and candidate space absent | Minimizes the stated parameter-to-gradient ratio over widths 8, 16, and 32; does not claim replication of the separate activation-diversity AD-ZeroNAS paper. |
| Cervical categories include diagnostic classes unavailable in the cited data | Uses the five categories actually provided by SIPaKMeD. |
| Cell-to-TNM mapping and staging-alignment percentages | Removed as experimental claims. TNM is unavailable in these datasets and is never inferred from class/confidence. The API only preserves externally supplied records with evidence metadata. |
| Semantic, staging, and monitoring capability charts with hand-entered scores | Replaced with measured graph size, validation outcomes, negative controls, query/event equality, and declared latency measurements. |
| Privacy-preserving federated deployment | Implemented and tested as a three-client, single-process FedAvg simulation. No privacy or institutional-deployment claim is retained. |
| Screenshots presented as scalability evidence | Replaced with generated figures and measured records. A local monitor is included as an operational prototype. |
| Clinically deployable framework | Narrowed to a research implementation evaluated on public patch/cell datasets. |

Author names, ORCIDs, institutional information, and the funding acknowledgment come from the supplied manuscript. The authors should confirm these details and any publisher-required declarations before submission. The source documents are not redistributed in the repository.

