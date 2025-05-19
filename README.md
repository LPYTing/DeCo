## DeCo: Defect-Aware Modeling with Contrasting Matching for Optimizing Task Assignment in Online IC Testing (IJCAI 2025)


The **DeCo** framework (**De**fect-Aware Modeling with **Co**ntrasting Matching for Optimizing Task Assignment in Online IC Testing) leverages graph-based learning to improve IC testing task assignment. 


## Abstract

In the semiconductor industry, integrated circuit (IC) processes play a vital role, as the rising complexity and market expectations necessitate improvements in yield. Identifying IC defects and assigning IC testing tasks to the right engineers improves efficiency and reduces losses. 

We propose DeCo, an innovative approach for optimizing task assignment in IC testing. DeCo constructs a novel defect-aware graph from IC testing reports, capturing co-failure relationships to enhance defect differentiation. Additionally, it formulates defect-aware representations for engineers and tasks, reinforced by local and global structure modeling on the defect-aware graph. Finally, a contrasting-based assignment mechanism pairs testing tasks with QA engineers by considering their skill level and current workload.

Experiments on a real-world dataset from a seminconductor demonstrate that DeCo achieves the highest task-handling success rates in different scenarios, while also maintaining balanced workloads on both scarce or expanded defect data.


## Experiments
The example of running the task assignment is as follows:
```bash
python deco_main.py --structure global --lambda_weight 0.5
```
Note that due to dataset privacy, we only release the main part of the assignment module in our model. We are currently organizing and cleaning up the releasable parts of the implementation.

## Citations

```
@article{ting2025deco,
  title={DeCo: Defect-Aware Modeling with Contrasting Matching for Optimizing Task Assignment in Online IC Testing},
  author={Ting, Lo Pang-Yun and Chiang, Yu-Hao and Tsai, Yi-Tung and Lai, Hsu-Chao and Chuang, Kun-Ta},
  journal={arXiv preprint arXiv:2505.00278},
  year={2025}
}
```


## Contributors
This codebase is co-developed with the following members from [NetDB](https://ncku-ccs.github.io/netdb-web/), NCKU
- [Yu-Hao Chiang](https://github.com/Hunk0724)
- [Yi-Tung Tsai]()