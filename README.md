# BoundClassifier-LDP

Code for paper: [PETS'26] Quantifying Classifier Utility under Local Differential Privacy

Contributions:
- *(Quantification framework)* The first framework that provides analytical utility quantification for classifiers under LDP mechanisms. This framework bridges the concentration analysis of LDP mechanisms with the robustness analysis of classifiers, enabling systematic evaluation of utility.
- *(Refinement techniques)* Two refinement techniques to enhance utility quantification. 
- *(Comparison and case studies)* Case studies on typical classifiers, including logistic regression, random forests, and neural networks, under various LDP mechanisms.

<p align="center">
    <img src="/others/readme_poster.png" alt="Framework Overview" width="800"/>
</p>

## Reproduction

To reproduce the results in the paper, first navigate to the `experiments/` directory and run the corresponding scripts in the `stroke_pred/`, `bank_attrition/`, and `mnist/` folders.

This project uses `uv` for environment management and execution. After installing `uv` and initializing the environment, navigate to the root directory of the project and run the experiments using:

```bash
uv run experiments/stroke_pred/lr_empirical_theo.py
```

This command runs the first experiment (Figure 6a in the paper) and saves the results to the `lr_accuracy.csv` file. The results can then be visualized using the provided `draw_accuracy.py` script.

For complete reproduction instructions, please refer to `Artifact-Appendix.md`.



## Code Structure

The code is organized as follows:
- `src/` contains the source code for the framework.
    - `cdf_ldp_mechanisms_at_x.py` implements the concentration analysis for LDP mechanisms at a given point x.
    - `ldp_mechanisms.py` & `samples_from_mechanism.py` implement the LDP mechanisms and the sampling methods.
    - `robust_radius_sklearn.py` & `robust_radius_torch.py` implement the robustness analysis for sklearn and PyTorch classifiers, respectively.
- `experiments/` contains the code for the experiments in the paper.


## Freedom of Usage

This project is licensed under the MIT License for freedom of usage and distribution.
Hope this paper and code can help you in your research or work.